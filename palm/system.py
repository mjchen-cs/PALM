"""
PQC-aware NOMA-MEC system: per-frame coefficient builder, critic (evaluates a
binary offloading candidate by solving the resource allocation exactly or
approximately), and the environment (queues, PQC sessions, non-idealities).
"""
import numpy as np
from . import solver as _solver_mod
from . import params as P
from .solver import (solve_exact, solve_greedy_val, solve_sca, solve_anns, y_to_power,
                     upper_bound_interference_free, solve_order_search)

LN2 = np.log(2.0)


class Devices:
    """Static per-device parameters (vectorised)."""

    def __init__(self, p, profile=None):
        N = p["N"]
        self.N = N
        cls = p["classes"]
        assert len(cls) == N, "len(classes) must equal N"
        self.cls = list(cls)
        g = lambda k, d=0.0: np.array([P.DEVICE_CLASSES[c].get(k, d) for c in cls], float)
        self.fmax, self.kappa, self.phi = g("fmax"), g("kappa"), g("phi")
        self.Pmax, self.gamma = g("Pmax"), g("gamma")
        self.gpu = np.array([P.DEVICE_CLASSES[c].get("gpu", False) for c in cls])
        self.R_gpu, self.P_gpu0, self.zeta = g("R_gpu"), g("P_gpu0"), g("zeta_gpu")
        pq = [P.pqc_costs(p, c, profile) for c in cls]
        self.t_hs = np.array([q["t_hs"] for q in pq])
        self.E_hs = np.array([q["E_hs"] for q in pq])
        self.B_hs = np.array([q["B_hs"] for q in pq])
        self.eps_aead = np.array([q["eps_aead"] for q in pq])
        self.P_static = np.array([q["P_static"] for q in pq])
        es = P.es_pqc_costs(p)
        self.E_hs_es = es["E_hs"]
        self.eta = P.eta_aead(p)
        if p["c_weights"] == "odd1.5":
            self.c = np.array([1.5 if (i + 1) % 2 == 1 else 1.0 for i in range(N)])
        else:
            self.c = np.ones(N)
        self.d = np.linspace(p["d_min"], p["d_max"], N)
        self.pl = p["Ad"] * (3e8 / (4 * np.pi * p["fc"] * self.d)) ** p["de"]
        self.N0 = P.noise_power(p)


def rician_power(rng, K_dB, size):
    K = 10 ** (K_dB / 10)
    los = np.sqrt(K / (K + 1))
    nl = np.sqrt(1 / (2 * (K + 1))) * (rng.standard_normal(size) + 1j * rng.standard_normal(size))
    return np.abs(los + nl) ** 2


# --------------------------------------------------------------------------
class Coeffs:
    """Per-frame, per-device coefficients independent of the offloading set."""
    pass


def build_coeffs(p, dev, h_hat, Q, Y, Y0, need_hs, pqc_aware=True, tau_dec=None, ps=None):
    N = dev.N
    T = p["T"]
    tau_dec = p["tau_dec"] if tau_dec is None else tau_dec
    nu, nu0 = p["nu"], p["nu0"]
    b = nu * Y
    b0 = nu0 * Y0
    a = Q + p["V"] * dev.c
    C = Coeffs()
    C.a, C.b, C.b0 = a, b, b0
    # ---------------- local computing (closed form) -------------------------
    kb_per_ghz = 1e3 * T / dev.phi                   # kbit per frame per GHz
    with np.errstate(divide="ignore", invalid="ignore"):
        f_star = np.where(b > 0, np.sqrt(a * 1e3 / (3 * dev.phi * np.maximum(b, 1e-300) * dev.kappa)), dev.fmax)
    f_cap = np.minimum(dev.fmax, Q / kb_per_ghz)
    f = np.clip(f_star, 0, f_cap)
    D_cpu = kb_per_ghz * f
    E_cpu = dev.kappa * f ** 3 * T
    v_cpu = a * D_cpu - b * E_cpu
    # GPU: energy linear in processed data, E = (P0/R + zeta) D
    e_per_kb = np.where(dev.gpu, dev.P_gpu0 / np.maximum(dev.R_gpu, 1e-9) + dev.zeta, 0.0)
    D_gpu = np.where(dev.gpu, np.minimum(Q, dev.R_gpu * T), 0.0)
    coef = a - b * e_per_kb
    D_gpu = np.where(coef > 0, D_gpu, 0.0)
    E_gpu = np.where(dev.gpu, e_per_kb * D_gpu, 0.0)
    v_gpu = np.where(dev.gpu, coef * D_gpu, -np.inf)
    use_gpu = v_gpu > v_cpu
    C.loc_val = np.where(use_gpu, v_gpu, v_cpu)
    C.loc_D = np.where(use_gpu, D_gpu, D_cpu)
    C.loc_E = np.where(use_gpu, E_gpu, E_cpu)
    C.loc_f = np.where(use_gpu, 0.0, f)
    C.loc_gpu = use_gpu
    # ---------------- offloading coefficients ------------------------------
    if pqc_aware:
        t_hs = np.where(need_hs, dev.t_hs, 0.0)
        B = np.where(need_hs, dev.B_hs, 0.0)
        E_fix = np.where(need_hs, dev.E_hs + dev.P_static * T, 0.0)
        E_fix_es = np.where(need_hs, dev.E_hs_es, 0.0)
        eps = dev.eps_aead
        eta = dev.eta
    else:  # PQC-agnostic planner
        t_hs = np.zeros(N); B = np.zeros(N); E_fix = np.zeros(N); E_fix_es = np.zeros(N)
        eps = np.zeros(N); eta = 1.0
    T_air = T - tau_dec - t_hs
    a_eff = a - b * eps - b0 * p["es_eps_mJ_per_kbit"]
    kr = eta * np.maximum(T_air, 1e-9) * p["W"] / p["v_u"]          # kbit per (bit/s/Hz)
    C.T_air, C.B, C.kr, C.a_eff, C.eta = T_air, B, kr, a_eff, eta
    C.alpha = a_eff * kr / LN2
    C.beta = b * np.maximum(T_air, 0) * dev.N0 / h_hat
    with np.errstate(over="ignore"):
        C.L = np.exp2(B / kr)
        C.U = np.exp2(np.minimum((Q + B) / kr, 60.0))   # SINR>2^60 is never reachable
    C.pbar = h_hat * dev.Pmax / dev.N0
    om = p["amortize"] if p["amortize"] is not None else 1.0 / max(p["rekey_frames"], 1)
    C.fix = om * (a_eff * B + b * E_fix + b0 * E_fix_es)
    C.E_fix, C.E_fix_es, C.t_hs = E_fix, E_fix_es, t_hs
    C.can_off = (T_air > 0) & (a_eff > 0) & (Q > 1e-6)
    C.h_hat = h_hat
    C.Q = Q
    C.om = om
    # ---------------- proactive session establishment (PALM-PS) ------------
    # Mode H: a WD that does not offload runs the handshake now (CPU busy for
    # t_hs, B_hs bits sent on the NOMA uplink) so that its next offloading frame
    # needs neither the handshake airtime nor the handshake bits.
    C.ps = False
    if ps is not None and pqc_aware:
        C.ps = True
        airmin = p.get("ps_airmin", 5.0)
        hs_rem = ps.get("hs_rem", np.zeros(N))
        pend = hs_rem > 0                               # handshake in progress
        tH = np.where(pend, hs_rem, dev.t_hs)           # remaining handshake computation
        TairH = T - tau_dec - tH
        # multi-frame handshake: if too little airtime would remain, this frame is
        # spent on computation only and the B_hs bits are sent in a later frame
        split = TairH < airmin
        C.H_pend, C.H_teff, C.H_split = pend, tH, split

        def loc_reduced(tl):
            """local value when the CPU is available only for tl ms (GPU unaffected)"""
            kbg = 1e3 * np.maximum(tl, 0) / dev.phi
            fH = np.clip(f_star, 0, np.minimum(dev.fmax, Q / np.maximum(kbg, 1e-12)))
            fH = np.where(tl > 0, fH, 0.0)
            D_ = kbg * fH; E_ = dev.kappa * fH ** 3 * np.maximum(tl, 0)
            v_ = a * D_ - b * E_
            g_ = v_gpu > v_
            return np.where(g_, v_gpu, v_), np.where(g_, D_gpu, D_), np.where(g_, E_gpu, E_)

        C.locH_val, C.locH_D, C.locH_E = loc_reduced(T - np.minimum(tH, T))
        krH = eta * np.maximum(TairH, 1e-9) * p["W"] / p["v_u"]
        C.H_T_air, C.H_kr = TairH, krH
        with np.errstate(over="ignore"):
            C.H_L = np.where(split, 1.0, np.exp2(dev.B_hs / krH))
        C.H_beta = b * np.maximum(TairH, 0) * dev.N0 / h_hat

        def solo(Tair, Bh):
            """stand-alone (interference-free) offloading value, Proposition 3(b)"""
            kr_ = eta * np.maximum(Tair, 1e-9) * p["W"] / p["v_u"]
            al = np.maximum(a_eff, 0) * kr_ / LN2
            be = b * np.maximum(Tair, 0) * dev.N0 / h_hat
            lo = np.exp2(Bh / kr_) - 1
            hi = np.minimum(np.exp2(np.minimum((Q + Bh) / kr_, 60.0)) - 1, h_hat * dev.Pmax / dev.N0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(be > 0, al / np.maximum(be, 1e-300) - 1, hi)
            r = np.clip(r, lo, np.maximum(hi, lo))
            v = al * np.log1p(r) - be * r - np.maximum(a_eff, 0) * Bh
            return np.where((Tair > 0) & (hi >= lo), v, -np.inf)

        C.H_vno = solo(np.full(N, T - tau_dec), np.zeros(N))   # offloading value with a live session
        # on-demand handshakes are impossible if a whole handshake does not fit a frame
        ondemand_ok = (T - tau_dec - dev.t_hs) >= airmin
        # Renewal rule (Proposition 3): a session covers Le frames; a proactive one
        # spends k = 1 (or 2 if split) of them without offloading.
        Le = max(min(p["idle_timeout"], p["rekey_frames"]), 1)
        nfr = np.where(split & ~pend, 2, 1)
        p_use = np.clip(ps["p_off"], 0, 1)
        # WDs that can never offload on demand: use the fraction of frames in which
        # offloading would beat local computing if a session were alive
        p_use = np.where(ondemand_ok, p_use, np.maximum(p_use, np.clip(ps.get("w_off", 0 * p_use), 0, 1)))
        kap = p_use * np.maximum(Le - nfr, 0)
        cost = b * (dev.E_hs + dev.P_static * T) + b0 * dev.E_hs_es
        Tfull = max(T - tau_dec, 1e-9)
        scale = Tfull / np.maximum(Tfull - dev.t_hs, 1e-9)
        D0, E0 = np.nan_to_num(ps["D_on"][0]), np.nan_to_num(ps["E_on"][0])
        D1 = np.where(np.isnan(ps["D_on"][1]), D0 * scale + dev.B_hs, ps["D_on"][1])
        E1 = np.where(np.isnan(ps["E_on"][1]), E0 * scale, ps["E_on"][1])
        d_emp = np.maximum(np.maximum(a_eff, 0) * (D1 - D0) - b * (E1 - E0), 0.0)
        # without on-demand handshakes, the alternative to a session frame is local computing
        d_inf = np.maximum(np.where(np.isfinite(C.H_vno), C.H_vno, 0.0) - C.loc_val, 0.0)
        dval = np.where(ondemand_ok, d_emp, d_inf)
        # expected loss in the second frame of a split handshake
        rem2 = np.maximum(tH - (T - tau_dec), 0.0)
        Tair2 = T - tau_dec - rem2
        kr2 = eta * np.maximum(Tair2, 1e-9) * p["W"] / p["v_u"]
        with np.errstate(over="ignore"):
            L2 = np.exp2(dev.B_hs / kr2)
        l2 = (C.loc_val - loc_reduced(T - rem2)[0]) + b * np.maximum(Tair2, 0) * dev.N0 / h_hat * (L2 - 1)
        C.H_l2 = np.where(split & ~pend, np.maximum(l2, 0.0), 0.0)
        C.H_kappa, C.H_Delta = kap, dval
        C.H_psi = p.get("ps_zeta", 1.0) * kap / (1.0 + kap) * (cost + dval)
        C.H_fix = cost + C.H_l2
        early = p.get("ps_early", 1) + split.astype(int)
        elig = ps["need_hs"] | (ps["alive"] & (ps["age"] >= p["rekey_frames"] - early))
        C.H_elig = (elig | pend) & (np.isfinite(C.H_L) | split) & (Tair2 > 0)
        C.H_elig_raw = C.H_elig.copy()
        # robustness: a proactive session must be expected to cover >= kappa_min offloading frames
        C.H_elig &= (kap >= p.get("ps_kmin", 1.0)) | pend
        # a WD with a handshake in progress cannot offload in this frame
        C.can_off = C.can_off & ~pend
    return C


# --------------------------------------------------------------------------
def _tdma(p, dev, C, idx, tau_dec):
    """Orthogonal (TDMA) allocation for offloading set idx: per-unit-time optimal
    power, then greedy time filling (exact for the linear time-sharing LP)."""
    T_av = p["T"] - tau_dec
    a_e, b = C.a_eff[idx], C.b[idx]
    h = C.h_hat[idx]
    rate_coef = C.eta * p["W"] / p["v_u"]
    with np.errstate(divide="ignore", invalid="ignore"):
        Pst = np.where(b > 0, a_e * rate_coef / (LN2 * np.maximum(b, 1e-300)) - dev.N0 / h, dev.Pmax[idx])
    Pst = np.clip(Pst, 0, dev.Pmax[idx])
    # handshake-only users (no payload value) still have to deliver their B_hs bits
    Pst = np.where((a_e <= 0) & (C.B[idx] > 0), dev.Pmax[idx], Pst)
    r = rate_coef * np.log2(1 + Pst * h / dev.N0)          # kbit/ms
    g = a_e * r - b * Pst
    if np.any(r <= 0):
        return -np.inf, None
    tmin = C.B[idx] / r
    tmax = np.minimum((C.Q[idx] + C.B[idx]) / r, C.T_air[idx])
    if np.any(tmin > tmax) or tmin.sum() > T_av:
        return -np.inf, None
    tau = tmin.copy()
    rem = T_av - tau.sum()
    for j in np.argsort(-g):
        if g[j] <= 0 or rem <= 0:
            break
        add = min(tmax[j] - tau[j], rem)
        tau[j] += add
        rem -= add
    val = np.sum(tau * g) - np.sum(a_e * C.B[idx])
    D = tau * r - C.B[idx]
    return val, dict(tau=tau, P=Pst, D=D, rate=r)


def evaluate(p, dev, C, x, order="weight", solver="exact", access="noma", tau_dec=None,
             return_alloc=False, partial=False):
    """Critic: value of the per-frame DPP subproblem for candidate x."""
    tau_dec = p["tau_dec"] if tau_dec is None else tau_dec
    x = np.asarray(x).astype(bool) & C.can_off
    loc = ~x
    val = C.loc_val[loc].sum()
    idx = np.flatnonzero(x)
    alloc = None
    if idx.size:
        val -= C.fix[idx].sum()
        if access == "tdma":
            v, al = _tdma(p, dev, C, idx, tau_dec)
            val += v
            if return_alloc and al is not None:
                alloc = dict(kind="tdma", idx=idx, **al)
        else:
            if solver == "search" and idx.size > 12:
                solver = "exact"          # real-time rule: order search only for M<=12
            if solver == "search":
                done = False
                tol = p.get("cert_tol", 0.0)
                if tol > 0:
                    # certificate (Proposition 4): if the weight-order optimum is within tol
                    # of the interference-free upper bound, no order can do better -> skip search
                    o = idx[np.argsort(C.alpha[idx], kind="stable")]
                    y, v, st = solve_exact(C.alpha[o], C.beta[o], C.L[o], C.U[o], C.pbar[o])
                    ub = upper_bound_interference_free(C.alpha[o], C.beta[o], C.L[o], C.U[o], C.pbar[o])
                    done = np.isfinite(v) and (ub - v) <= tol * max(abs(ub), 1e-9)
                if not done:
                    _solver_mod._PREDICT[0] = bool(p.get("rt_pred", p.get("rt_fast", False)))
                    perm, y, v = solve_order_search(C.alpha[idx], C.beta[idx], C.L[idx], C.U[idx], C.pbar[idx],
                                                    budget=p.get("ref_budget", 3e-3))
                    o = idx[perm]
                st = 0
            elif order == "weight":
                o = idx[np.argsort(C.alpha[idx], kind="stable")]
            else:  # strongest channel decoded first
                o = idx[np.argsort(-C.h_hat[idx], kind="stable")]
            args = (C.alpha[o], C.beta[o], C.L[o], C.U[o], C.pbar[o])
            if solver == "search":
                pass
            elif solver == "exact":
                if order != "weight":
                    raise ValueError("exact solver requires weight order")
                y, v, st = solve_exact(*args)
            elif solver == "greedy":
                y, v, st = solve_greedy_val(*args)
            elif solver == "sca":
                y, v, st, _ = solve_sca(*args)
            elif solver == "anns":
                y, v, st, _ = solve_anns(*args)
            else:
                raise ValueError(solver)
            # constant from y_{M+1}=1 is already inside objective_y
            xi = p["sic_residual"] if p.get("sic_aware", False) else 0.0
            if xi > 0 and np.isfinite(v):
                # SIC-aware re-evaluation: rates with residual interference xi from
                # already-decoded users (rate adaptation), then robust access selection
                yn = np.append(y[1:], 1.0)
                qn = y - yn
                before = np.concatenate([[0.0], np.cumsum(qn)[:-1]])
                sinr = qn / (yn + xi * before)
                bits = np.minimum(C.kr[o] * np.log2(1 + sinr), C.Q[o] + C.B[o])
                ok = bits >= C.B[o] * (1 - 1e-9)
                v_adj = np.sum(np.where(ok, C.a_eff[o] * bits, 0.0) - C.beta[o] * qn)
                vt, alt = _tdma(p, dev, C, idx, tau_dec)
                if vt > v_adj:
                    val += vt
                    if return_alloc and alt is not None:
                        alloc = dict(kind="tdma", idx=idx, **alt)
                    if return_alloc:
                        return val, x, alloc
                    return val
                val += v_adj
                if return_alloc:
                    Pw = y_to_power(y, C.h_hat[o], dev.N0)
                    D = np.where(ok, bits - C.B[o], 0.0)
                    alloc = dict(kind="noma", idx=o, y=y, P=Pw, D=D, sinr=sinr)
                    return val, x, alloc
                return val
            val += v
            if return_alloc and np.isfinite(v):
                Pw = y_to_power(y, C.h_hat[o], dev.N0)
                yn = np.append(y[1:], 1.0)
                D = C.kr[o] * np.log2(y / yn) - C.B[o]
                alloc = dict(kind="noma", idx=o, y=y, P=Pw, D=D, sinr=y / yn - 1)
    if return_alloc:
        return val, x, alloc
    return val


def solo_gains(C):
    """Exact single-WD offloading gains J(e_i)-J(0) in closed form (M=1: no
    interference, P3 is a scalar concave problem)."""
    lo = C.L - 1.0
    hi = np.minimum(C.U - 1.0, C.pbar)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(C.beta > 0, C.alpha / np.maximum(C.beta, 1e-300) - 1.0, hi)
    r = np.clip(r, lo, np.maximum(hi, lo))
    v = C.alpha * np.log1p(r) - C.beta * r - C.fix - C.loc_val
    return np.where(C.can_off & (hi >= lo), v, -np.inf)


def _with_H(C, h):
    """Shallow copy of C in which the WDs in h are entered as handshake-only
    NOMA users (alpha=0, exactly B_hs bits, value of mode H in `fix`)."""
    import copy
    D = copy.copy(C)
    for k in ("alpha", "beta", "L", "U", "fix", "can_off", "a_eff", "kr", "T_air", "B", "Q"):
        setattr(D, k, getattr(C, k).copy())
    D.alpha[h] = 0.0
    D.beta[h] = C.H_beta[h]
    D.L[h] = C.H_L[h]
    D.U[h] = C.H_L[h] * (1.0 + 1e-3)
    D.a_eff[h] = 0.0
    D.kr[h] = C.H_kr[h]
    D.T_air[h] = C.H_T_air[h]
    D.B[h] = np.log2(C.H_L[h]) * C.H_kr[h]
    D.Q[h] = 0.0
    # evaluate() adds -fix for every scheduled WD and omits its local value
    D.fix[h] = C.H_fix[h] - C.locH_val[h] - C.H_psi[h]
    D.can_off[h] = True
    return D


def h_gain(C, x):
    """Stand-alone (interference-free) gain of mode H for each non-offloading WD."""
    g = C.locH_val - C.loc_val - C.H_fix + C.H_psi - np.where(C.H_split, 0.0, C.H_beta * (C.H_L - 1.0))
    return np.where(C.H_elig & ~np.asarray(x, bool), g, -np.inf)


def evaluate_ps(p, dev, C, x, order="weight", solver="exact", access="noma", tau_dec=None,
                return_alloc=False, rule="dpp", v0=None, mode_hint=None):
    """Critic with proactive sessions: for offloading vector x, also decides the
    set H of non-offloading WDs that establish a session now.  Two variants are
    solved exactly: H = {} and H = {i: stand-alone gain > 0}; the better is kept."""
    x = np.asarray(x).astype(bool)
    if v0 is not None and not return_alloc:
        r0 = v0                       # value without proactive handshakes already known
    elif mode_hint is True and return_alloc and getattr(C, "ps", False) and access == "noma":
        r0 = (-np.inf, x & C.can_off, None)   # mode H known to win: refine only that branch
    else:
        r0 = evaluate(p, dev, C, x, order, solver, access, tau_dec, return_alloc=return_alloc)
    v0 = r0[0] if return_alloc else r0
    if not getattr(C, "ps", False) or access != "noma":
        return (v0, x & C.can_off, r0[2], np.zeros_like(x)) if return_alloc else v0
    xo = x & C.can_off
    g = h_gain(C, xo)
    h = (g > 0) if rule == "dpp" else (C.H_elig_raw & ~xo)    # "always": keep-alive baseline
    h = h | C.H_pend                  # a handshake in progress is always continued
    if mode_hint is False and return_alloc and not C.H_pend.any():
        h = np.zeros_like(h)
    if not h.any():
        return (v0, xo, r0[2], np.zeros_like(x)) if return_alloc else v0
    hn = h & ~C.H_split               # handshake bits sent in this frame (NOMA users)
    hc = h & C.H_split                # computation-only frame of a multi-frame handshake
    D = _with_H(C, hn)
    r1 = evaluate(p, dev, D, xo | hn, order, solver, access, tau_dec, return_alloc=return_alloc)
    v1 = r1[0] if return_alloc else r1
    v1 = v1 + np.sum((C.locH_val - C.loc_val - C.H_fix + C.H_psi)[hc])
    if C.H_pend.any() and np.isfinite(v1):   # forced continuation of a handshake in progress
        v0 = -np.inf
    if rule != "dpp":                 # keep-alive: use H whenever it is feasible
        v0 = -np.inf if np.isfinite(v1) else v0
    if not return_alloc:
        return max(v0, v1)
    if mode_hint is True and not np.isfinite(v1):        # fall back to the no-H branch
        return evaluate_ps(p, dev, C, x, order, solver, access, tau_dec, return_alloc=True, rule=rule)
    if v1 > v0:
        return v1, xo, r1[2], h
    return v0, xo, r0[2], np.zeros_like(x)


# --------------------------------------------------------------------------

def robust_bits(planned, B, hs, Pw, h_hat, N0, sigma, kr, kappa_hs, kappa_pl, xi=0.0):
    """Robust rate selection for fixed-rate transmission under CSI error
    h = h_hat/(1+e), e ~ N(0, sigma^2).  The controller knows only h_hat and sigma.
    For the k-th decoded user, the rate is set for a pessimistic own gain
    g_lo = 1/(1+kappa*sigma) and a pessimistic interference (mean plus kappa standard
    deviations of the sum of the later-decoded users' received powers).  Handshake
    frames (hs) use the larger margin kappa_hs, because an outage there forces a new
    handshake; payload-only frames use kappa_pl.  Returns the bits to transmit:
    min(planned, robust capacity), but at least B for a handshake (attempt)."""
    q = Pw * h_hat / N0                                # planned received SNR contributions
    M = q.shape[0]
    if kappa_pl is None:                               # automatic margins (see robust_bits_auto)
        return robust_bits_auto(planned, B, hs, q, sigma, kr, xi)
    kap = np.where(hs, kappa_hs, kappa_pl)
    mu_g = 1.0 + sigma ** 2                            # E[1/(1+e)] (second order)
    out = planned.copy()
    for k in range(M):
        g_lo = 1.0 / (1.0 + kap[k] * sigma)
        qi = q[k + 1:]
        I = mu_g * qi.sum() + kap[k] * sigma * np.sqrt(np.sum(qi ** 2)) + xi * q[:k].sum()
        cap = kr[k] * np.log2(1.0 + q[k] * g_lo / (1.0 + I))
        out[k] = min(planned[k], cap)
        if hs[k]:
            out[k] = max(out[k], B[k])
    return out


_KGRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


def robust_bits_auto(planned, B, hs, q, sigma, kr, xi=0.0):
    """Per-user margins: a payload-only user maximizes its expected delivered bits
    cap(kappa) * Phi(kappa)^2 (own gain and interference both within kappa standard
    deviations); a handshake user takes the largest margin whose capacity still carries
    the handshake bits B (the handshake succeeds with the highest probability), and
    sends payload only within that capacity."""
    from math import erf, sqrt
    Phi = np.array([0.5 * (1 + erf(k / sqrt(2))) for k in _KGRID])
    mu_g = 1.0 + sigma ** 2
    out = planned.copy()
    M = q.shape[0]
    for k in range(M):
        qi = q[k + 1:]
        I0 = mu_g * qi.sum() + xi * q[:k].sum()
        sd = sigma * np.sqrt(np.sum(qi ** 2))
        caps = kr[k] * np.log2(1.0 + q[k] / (1.0 + _KGRID * sigma) / (1.0 + I0 + _KGRID * sd))
        caps = np.minimum(caps, planned[k])
        if hs[k]:
            okk = np.flatnonzero(caps >= B[k] * (1 - 1e-9))
            out[k] = caps[okk[-1]] if okk.size else B[k]
        else:
            out[k] = caps[np.argmax(caps * Phi ** 2)]
    return out

class Env:
    def __init__(self, p, profile=None, seed=None):
        self.p = p
        self.dev = Devices(p, profile)
        self.rng = np.random.default_rng(p["seed"] if seed is None else seed)
        N = p["N"]
        self.Q = np.zeros(N)
        self.Y = np.zeros(N)
        self.Y0 = 0.0
        self.alive = np.zeros(N, bool)
        self.age = np.zeros(N, int)
        self.idle = np.zeros(N, int)
        # statistics used to value a proactive session (PALM-PS)
        self.p_off = np.zeros(N)            # EMA of the offloading indicator
        self.r_bar = np.zeros(N)            # EMA of the realized payload rate (kbit/ms airtime)
        self.P_bar = np.zeros(N)            # EMA of the transmit power while offloading (W)
        # EMA of payload (kbit) and transmit energy (mJ) per offloading frame,
        # separately for frames with [0] and without [1] an on-demand handshake
        self.D_on = np.full((2, N), np.nan)
        self.E_on = np.full((2, N), np.nan)
        self.hs_rem = np.zeros(N)           # remaining computation of a multi-frame handshake (ms)
        self.w_off = np.zeros(N)            # EMA of 1{offloading with a live session would beat local}
        self.t = 0
        self.lam = np.full(N, p["lam"]) if np.isscalar(p["lam"]) else np.asarray(p["lam"], float)
        self._draw_channel()

    def _draw_channel(self):
        p = self.p
        self.h = self.dev.pl * rician_power(self.rng, p["rician_K_dB"], self.dev.N)
        if p["csi_err_std"] > 0:
            e = self.rng.standard_normal(self.dev.N) * p["csi_err_std"]
            self.h_hat = np.maximum(self.h * (1 + e), 1e-3 * self.h)
        else:
            self.h_hat = self.h.copy()

    def need_hs(self):
        return (~self.alive) | (self.age >= self.p["rekey_frames"])

    def state(self):
        return dict(h_hat=self.h_hat * (1 - self.p["backoff"]), Q=self.Q.copy(), Y=self.Y.copy(),
                    Y0=self.Y0, need_hs=self.need_hs(), alive=self.alive.copy())

    def ps_state(self):
        return dict(need_hs=self.need_hs(), alive=self.alive.copy(), age=self.age.copy(),
                    p_off=self.p_off.copy(), r_bar=self.r_bar.copy(), P_bar=self.P_bar.copy(),
                    D_on=self.D_on.copy(), E_on=self.E_on.copy(), hs_rem=self.hs_rem.copy(),
                    w_off=self.w_off.copy())

    def coeffs(self, pqc_aware=True, tau_dec=None, ps=None):
        s = self.state()
        return build_coeffs(self.p, self.dev, s["h_hat"], s["Q"], s["Y"], s["Y0"], s["need_hs"],
                            pqc_aware, tau_dec, ps=ps)

    # ------------------------------------------------------------------
    def step(self, x, alloc, C_true, tau_dec=None, hmode=None):
        """Execute decision. C_true: PQC-aware coefficients (true costs) used for
        local allocation; alloc from evaluate(..., return_alloc=True)."""
        p, dev = self.p, self.dev
        tau_dec = p["tau_dec"] if tau_dec is None else tau_dec
        N, T = dev.N, p["T"]
        need = self.need_hs()
        D = np.zeros(N)
        E = np.zeros(N)
        E_es = 0.0
        x = np.asarray(x, bool)
        H = np.zeros(N, bool) if hmode is None else (np.asarray(hmode, bool) & ~x)
        hsf = need | H                     # WDs that run a handshake if scheduled
        ps_on = getattr(C_true, "ps", False)
        Hc = H & C_true.H_split if ps_on else np.zeros(N, bool)    # computation-only frame
        teff = np.where(H, C_true.H_teff, dev.t_hs) if ps_on else dev.t_hs
        efrac = np.where(H, teff / np.maximum(dev.t_hs, 1e-12), 1.0)  # share of E_hs spent now
        loc = ~x
        # local
        D[loc] = np.minimum(C_true.loc_D[loc], self.Q[loc])
        E[loc] = C_true.loc_E[loc]
        if H.any():                        # local computing shares the CPU with the handshake
            D[H] = np.minimum(C_true.locH_D[H], self.Q[H])
            E[H] = C_true.locH_E[H]
        hs_ok = np.zeros(N, bool)
        if Hc.any():                       # computation part of a multi-frame handshake
            c = np.minimum(T - tau_dec, teff[Hc])
            E[Hc] += dev.E_hs[Hc] * c / dev.t_hs[Hc]
        if alloc is not None and len(alloc["idx"]):
            idx = alloc["idx"]
            t_hs = np.where(hsf[idx], teff[idx], 0.0)
            B = np.where(hsf[idx], dev.B_hs[idx], 0.0)
            T_air = T - tau_dec - t_hs
            if alloc["kind"] == "noma":
                Pw = np.maximum(alloc["P"], 0)
                rx = Pw * self.h[idx]
                M = len(idx)
                sinr = np.empty(M)
                for k in range(M):
                    interf = rx[k + 1:].sum() + p["sic_residual"] * rx[:k].sum()
                    sinr[k] = rx[k] / (dev.N0 + interf)
                cap = dev.eta * T_air * p["W"] / p["v_u"] * np.log2(1 + sinr)  # kbit
                txE = Pw * T_air
            else:
                Pw = alloc["P"]
                tau = alloc["tau"]
                r_act = dev.eta * p["W"] / p["v_u"] * np.log2(1 + Pw * self.h[idx] / dev.N0)
                cap = tau * r_act
                sinr = Pw * self.h[idx] / dev.N0
                txE = Pw * tau
            Hi = H[idx]
            planned = np.where(Hi, B, alloc["D"] + B)     # planned total bits incl. handshake
            if p["rate_mode"] == "fixed" and p.get("robust_rate", False) and alloc["kind"] == "noma" \
                    and p["csi_err_std"] > 0:
                planned = robust_bits(planned, B, hsf[idx], Pw, self.h_hat[idx], dev.N0, p["csi_err_std"],
                                      dev.eta * T_air * p["W"] / p["v_u"], p.get("kappa_hs", 3.0),
                                      p.get("kappa_pl", None), p["sic_residual"])
            if p["rate_mode"] == "fixed":
                ok = cap >= planned * (1 - 1e-9)
                deliv = np.where(ok, planned, 0.0)
            else:
                deliv = np.minimum(cap, np.where(Hi, B, self.Q[idx] + B))
            payload = np.where(Hi, 0.0, np.maximum(deliv - B, 0.0))
            hs_ok[idx] = deliv >= B * (1 - 1e-9)
            Dl = np.where(Hi, D[idx], 0.0)                 # local data of mode-H WDs
            El = np.where(Hi, E[idx], 0.0)
            D[idx] = Dl + np.minimum(payload, self.Q[idx])
            E[idx] = El + txE + dev.eps_aead[idx] * np.minimum(payload, self.Q[idx]) \
                + np.where(hsf[idx], dev.E_hs[idx] * efrac[idx], 0.0)
            E_es = np.sum(p["es_eps_mJ_per_kbit"] * np.minimum(payload, self.Q[idx])
                          + np.where(hsf[idx], dev.E_hs_es, 0.0))
            # realized payload rate (kbit per ms of airtime) of offloading WDs
            off_i = idx[~Hi]
            if off_i.size:
                ta = np.maximum(T_air[~Hi], 1e-9)
                rr = np.minimum(payload[~Hi], self.Q[off_i]) / ta
                bw = p.get("ps_ema", 0.05)
                first = self.r_bar[off_i] == 0
                self.r_bar[off_i] = np.where(first, rr, (1 - bw) * self.r_bar[off_i] + bw * rr)
                pp = txE[~Hi] / ta
                self.P_bar[off_i] = np.where(first, pp, (1 - bw) * self.P_bar[off_i] + bw * pp)
                dd = np.minimum(payload[~Hi], self.Q[off_i]); ee = txE[~Hi]
                k = np.where(hsf[off_i], 0, 1)
                for j, i in enumerate(off_i):
                    if np.isnan(self.D_on[k[j], i]):
                        self.D_on[k[j], i], self.E_on[k[j], i] = dd[j], ee[j]
                    else:
                        self.D_on[k[j], i] += bw * (dd[j] - self.D_on[k[j], i])
                        self.E_on[k[j], i] += bw * (ee[j] - self.E_on[k[j], i])
        # static secure-module power while a session is alive
        E += np.where(self.alive | hs_ok, dev.P_static * T, 0.0)
        # ---------------- queues -----------------------------------------
        A = self.rng.exponential(self.lam * T)
        self.Q = self.Q - D + A
        self.Y = np.maximum(self.Y + p["nu"] * (E - dev.gamma * T), 0.0) / 1.0
        self.Y0 = max(self.Y0 + p["nu0"] * (E_es - p["es_gamma"] * T), 0.0)
        # ---------------- sessions ----------------------------------------
        started = (x | H) & hs_ok & hsf
        if ps_on:
            new_rem = np.where(Hc, np.maximum(teff - (T - tau_dec), 1e-3), 0.0)
            self.hs_rem = new_rem          # completed, failed or aborted handshakes reset to 0
            bw0 = p.get("ps_ema", 0.05)
            wv = np.isfinite(C_true.H_vno) & (C_true.H_vno > C_true.loc_val)
            self.w_off = (1 - bw0) * self.w_off + bw0 * wv
        self.age = np.where(started, 0, np.where(self.alive, self.age + 1, 0))
        self.alive = (self.alive & ~(x & need & ~hs_ok)) | started
        self.idle = np.where(x | started, 0, self.idle + 1)
        bw = p.get("ps_ema", 0.05)
        self.p_off = (1 - bw) * self.p_off + bw * x
        drop = self.idle > p["idle_timeout"]
        self.alive &= ~drop
        self.t += 1
        self._draw_channel()
        return dict(D=D, E=E, E_es=E_es, x=x, wrate=np.sum(dev.c * D) / T,
                    hs=int(started.sum()), hpro=int((H & started).sum()))
