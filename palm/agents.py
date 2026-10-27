"""
Online agents.

* LyDROO-style learning-assisted framework (actor DNN + noisy order-preserving
  quantiser + model-based critic + replay-based imitation update).  This is the
  "PALM" framework of the paper when the critic uses the exact convex solver
  with weight-ordered SIC.
* Oracle: exhaustive search over all 2^N offloading vectors (small N).
* DDPG: model-free DRL baseline acting directly on (offload, power, freq).
* Simple heuristics (all-local, all-offload, random).
"""
import gc
import time
import itertools
import numpy as np
import torch
import torch.nn as nn

from . import solver as _solver
from .system import solo_gains, evaluate_ps, Env, evaluate, build_coeffs

torch.set_num_threads(1)


class RunningNorm:
    def __init__(self, dim, eps=1e-6):
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.ones(dim)
        self.eps = eps

    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)

    def __call__(self, x):
        std = np.sqrt(self.M2 / max(self.n, 1)) + self.eps
        return np.clip((x - self.mean) / std, -5, 5)


def features(env, s):
    return np.concatenate([np.log10(s["h_hat"] / env.dev.pl), np.log1p(s["Q"]),
                           np.log1p(s["Y"]), s["need_hs"].astype(float),
                           np.full(1, np.log1p(s["Y0"]))])


class Actor(nn.Module):
    def __init__(self, din, N, hidden=(256, 128)):
        super().__init__()
        layers, d = [], din
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, N)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def nop_quantize(xh, K, rng, noise_std=0.15):
    """Noisy order-preserving quantisation (LyDROO): K/2 order-preserving
    candidates + K/2 Gaussian-perturbed candidates. Always adds all-local."""
    N = xh.shape[0]
    cands = [(xh > 0.5).astype(np.int8)]
    Kop = max(1, K // 2)
    order = np.argsort(np.abs(xh - 0.5))
    for k in range(1, min(Kop, N + 1)):
        th = xh[order[k - 1]]
        if th > 0.5:
            c = (xh > th).astype(np.int8)
        else:
            c = (xh >= th).astype(np.int8)
        cands.append(c)
    while len(cands) < K:
        cands.append((xh + rng.normal(0, noise_std, N) > 0.5).astype(np.int8))
    cands.append(np.zeros(N, np.int8))
    # unique
    seen, out = set(), []
    for c in cands:
        key = c.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


SCHEMES = {
    # name: critic configuration.  refine: re-solve the selected candidate with
    # SIC-order search; prior: add model-based singleton-ranking candidates.
    "PALM":          dict(order="weight", solver="exact", access="noma", pqc_aware=True, refine=True, prior=True),
    "PALM-G":        dict(order="weight", solver="greedy", access="noma", pqc_aware=True, refine=False, prior=True),
    "PALM-noprior":  dict(order="weight", solver="exact", access="noma", pqc_aware=True, refine=True, prior=False),
    "PALM-norefine": dict(order="weight", solver="exact", access="noma", pqc_aware=True, refine=False, prior=True),
    "LyDROO-SCA":    dict(order="channel", solver="sca", access="noma", pqc_aware=True, refine=False, prior=True),
    "LyDROO-Greedy": dict(order="channel", solver="greedy", access="noma", pqc_aware=True, refine=False, prior=True),
    "LyDROO-TDMA":   dict(order="weight", solver="exact", access="tdma", pqc_aware=True, refine=False, prior=True),
    "PQC-agnostic":  dict(order="weight", solver="exact", access="noma", pqc_aware=False, refine=True, prior=True),
    "LyDROO-orig":   dict(order="channel", solver="greedy", access="noma", pqc_aware=True, refine=False, prior=False),
    # recent baselines re-implemented in our setting (Section VII)
    "NOMA-ANNS":     dict(order="channel", solver="anns", access="noma", pqc_aware=True, refine=False, prior=True),
    "LyDiff":        dict(order="weight", solver="exact", access="noma", pqc_aware=True, refine=False, prior=False),
    "CEA":           dict(order="weight", solver="exact", access="noma", pqc_aware=True, refine=False, prior=False),
    # proactive session establishment (this revision)
    "PALM-PS":       dict(order="weight", solver="exact", access="noma", pqc_aware=True, refine=True, prior=True,
                          proactive="dpp"),
    "PALM-PS-RT":    dict(order="weight", solver="exact", access="noma", pqc_aware=True, refine=True, prior=True,
                          proactive="dpp", rt_fast=True, screen_tol=1e-4),
    "PALM-RT":       dict(order="weight", solver="exact", access="noma", pqc_aware=True, refine=True, prior=True,
                          rt_fast=True, screen_tol=1e-4),
    "PALM-G-PS":     dict(order="weight", solver="greedy", access="noma", pqc_aware=True, refine=False, prior=True,
                          proactive="dpp"),
    "KeepAlive":     dict(order="weight", solver="exact", access="noma", pqc_aware=True, refine=True, prior=True,
                          proactive="always"),
}
GENERATOR = {"LyDiff": "diffusion", "CEA": "ga"}


class DiffusionActor(nn.Module):
    """Conditional denoising-diffusion policy over relaxed offloading vectors
    (generative actor in the spirit of Lyapunov-guided diffusion DRL, Liu et al.,
    IEEE TMC 2025). eps_theta(x_t, t, s); a few denoising steps."""

    def __init__(self, din, N, steps=5, hidden=256):
        super().__init__()
        self.N, self.steps = N, steps
        self.net = nn.Sequential(nn.Linear(din + N + steps, hidden), nn.Mish(), nn.Linear(hidden, hidden),
                                 nn.Mish(), nn.Linear(hidden, N))
        betas = torch.linspace(0.05, 0.5, steps)
        alphas = 1 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", torch.cumprod(alphas, 0))

    def eps(self, x, t, s):
        te = nn.functional.one_hot(t, self.steps).float()
        return self.net(torch.cat([x, te, s], -1))

    def loss(self, s, x0):
        B = x0.shape[0]
        t = torch.randint(0, self.steps, (B,))
        n = torch.randn_like(x0)
        ab = self.abar[t].unsqueeze(1)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * n
        return nn.functional.mse_loss(self.eps(xt, t, s), n)

    @torch.no_grad()
    def sample(self, s, K, gen=None):
        x = torch.randn(K, self.N, generator=gen)
        S = s.expand(K, -1)
        for k in range(self.steps - 1, -1, -1):
            t = torch.full((K,), k, dtype=torch.long)
            e = self.eps(x, t, S)
            a, ab, b = self.alphas[k], self.abar[k], self.betas[k]
            x = (x - b / (1 - ab).sqrt() * e) / a.sqrt()
            if k > 0:
                x = x + b.sqrt() * torch.randn(K, self.N, generator=gen)
        return x


class LyDROO:
    def __init__(self, env, scheme="PALM", K0=None, mem=1024, batch=128, train_every=10,
                 lr=1e-3, delta_K=32, seed=0, tau_dec=None, generator="dnn"):
        self.env = env
        self.cfg = dict(SCHEMES[scheme]) if isinstance(scheme, str) else dict(scheme)
        N = env.dev.N
        self.N = N
        self.rng = np.random.default_rng(seed + 123)
        torch.manual_seed(seed)
        din = 4 * N + 1
        self.norm = RunningNorm(din)
        self.actor = Actor(din, N)
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.09, 0.999), weight_decay=1e-4)
        self.mem_x = np.zeros((mem, din), np.float32)
        self.mem_y = np.zeros((mem, N), np.float32)
        self.mem_n = 0
        self.mem = mem
        self.batch = batch
        self.train_every = train_every
        self.K = K0 or N
        self.delta_K = delta_K
        self.best_k_hist = []
        self.tau_dec = tau_dec
        self.generator = generator
        if generator == "oracle":
            self.all_x = np.array(list(itertools.product([0, 1], repeat=N)), np.int8)
        if generator == "diffusion":
            self.diff = DiffusionActor(din, N)
            self.opt = torch.optim.Adam(self.diff.parameters(), lr=lr)
            self.tgen = torch.Generator().manual_seed(seed + 5)
        self.ga_best = None
        # run-time options of a scheme (e.g. the real-time configuration PALM-PS-RT)
        self.rp = dict(env.p, **{k: self.cfg[k] for k in ("rt_fast", "screen_tol") if k in self.cfg})

    def decide(self):
        env, cfg = self.env, self.cfg
        P_ = self.rp
        s = env.state()
        feat = features(env, s)
        self.norm.update(feat)
        z = self.norm(feat).astype(np.float32)
        t0 = time.perf_counter()
        pro = cfg.get("proactive", False)
        _solver._EXACT_TOL[0] = P_.get("screen_tol", 1e-7)     # candidate screening
        C_plan = build_coeffs(P_, env.dev, s["h_hat"], s["Q"], s["Y"], s["Y0"], s["need_hs"],
                              cfg["pqc_aware"], self.tau_dec, ps=env.ps_state() if pro else None)
        ev = lambda c, sv=None: evaluate(P_, env.dev, C_plan, c, cfg["order"], sv or cfg["solver"],
                                          cfg["access"], self.tau_dec)
        ev_ps = (lambda c, v0=None: evaluate_ps(P_, env.dev, C_plan, c, cfg["order"], cfg["solver"],
                                                cfg["access"], self.tau_dec, rule=pro, v0=v0)) if pro else None
        if self.generator == "dnn":
            with torch.no_grad():
                xh = self.actor(torch.from_numpy(z)).numpy()
            cands = nop_quantize(xh, self.K, self.rng)
            if cfg.get("prior", False):
                # model-based prior: rank WDs by their stand-alone DPP gain, add prefixes
                if P_.get("fast_prior", False) and cfg["access"] == "noma":
                    gain = solo_gains(C_plan)       # closed form, identical to the M=1 critic
                else:
                    f0 = C_plan.loc_val.sum()
                    gain = np.full(self.N, -np.inf)
                    for i in np.flatnonzero(C_plan.can_off):
                        e = np.zeros(self.N, np.int8); e[i] = 1
                        gain[i] = ev(e) - f0
                rk = [i for i in np.argsort(-gain) if gain[i] > 0]
                for m in range(1, min(len(rk), 10) + 1):      # at most 10 prefixes (inactive for N<=10)
                    c = np.zeros(self.N, np.int8); c[rk[:m]] = 1
                    cands.append(c)
        elif self.generator == "diffusion":
            xs = self.diff.sample(torch.from_numpy(z).unsqueeze(0), max(self.K, 2), self.tgen).numpy()
            cands = [(x > 0).astype(np.int8) for x in xs] + [np.zeros(self.N, np.int8)]
        elif self.generator == "ga":
            cands = self._ga(ev)
        elif self.generator == "oracle":
            cands = self.all_x
        elif self.generator == "alllocal":
            cands = [np.zeros(self.N, np.int8)]
        elif self.generator == "alloff":
            cands = [np.ones(self.N, np.int8), np.zeros(self.N, np.int8)]
        elif self.generator == "random":
            cands = [self.rng.integers(0, 2, self.N).astype(np.int8), np.zeros(self.N, np.int8)]
        else:
            cands = self.generator(env, C_plan)          # external generator (e.g. QAOA)
        best, bv, bi = None, -np.inf, 0
        vals = []
        seen = {}
        rt = P_.get("rt_fast", False)
        if P_.get("rt_bb", rt) and self.generator == "dnn" and cfg["access"] == "noma" and not P_.get("sic_aware", False):
            # bound-and-prune screening (exact): J(x) <= sum_i loc_val_i + sum_{i in x} g_i, with the
            # closed-form stand-alone gains g_i (interference-free bound).  Candidates are visited in
            # order of decreasing bound and skipped when the bound is below the k-th best exact value
            # (k = 1, or n_ps for PALM-PS so that the re-scored set is unchanged).
            sg = solo_gains(C_plan)
            sg = np.where(C_plan.can_off, sg, 0.0)
            f0 = C_plan.loc_val.sum()
            X = np.asarray(cands, np.int8)
            with np.errstate(invalid="ignore"):
                ub = f0 + (X * sg).sum(1, where=X.astype(bool))
            ub = np.where(np.isnan(ub), -np.inf, ub)
            kk = cfg.get("n_ps", 3) if pro else 1
            top = []                                        # k best exact values (unique candidates)
            vals = [-np.inf] * len(cands)
            budget = P_.get("dec_budget")
            for i in np.argsort(-ub, kind="stable"):
                key = X[i].tobytes()
                if key in seen:
                    vals[i] = seen[key]
                    continue
                thr = top[-1] if len(top) >= kk else -np.inf
                if ub[i] < thr:
                    break                                   # all remaining bounds are smaller
                if budget is not None and top and time.perf_counter() - t0 > budget:
                    break                                   # anytime: keep the best found so far
                v = ev(cands[i])
                seen[key] = v
                vals[i] = v
                top = sorted(top + [v], reverse=True)[:kk]
            for i, v in enumerate(vals):
                if v > bv:
                    best, bv, bi = cands[i], v, i
            if best is None:
                best, bi = cands[int(np.argmax(ub))], int(np.argmax(ub))
        else:
            for i, c in enumerate(cands):
                key = np.asarray(c, np.int8).tobytes()
                if key not in seen:                 # duplicate candidates are evaluated once
                    seen[key] = ev(c)
                v = seen[key]
                vals.append(v)
                if v > bv:
                    best, bv, bi = c, v, i
        if pro:
            # proactive sessions: re-score the n_ps best candidates with mode H
            bv = -np.inf
            order_ = np.argsort(-np.asarray(vals), kind="stable")
            uniq, keys_ = [], set()
            for i in order_:
                k_ = np.asarray(cands[i], np.int8).tobytes()
                if k_ not in keys_:
                    keys_.add(k_); uniq.append(i)
            hint = None
            for i in uniq[:cfg.get("n_ps", 3)]:
                if not np.isfinite(vals[i]) and rt:
                    continue
                v = ev_ps(cands[i], v0=vals[i])
                if v > bv:
                    best, bv, bi = cands[i], v, i
                    hint = v > vals[i]          # proactive handshakes improve this candidate
            if best is None:
                best = cands[bi]
        _solver._EXACT_TOL[0] = 1e-7                               # selected candidate: full accuracy
        sv = "search" if cfg.get("refine", False) and cfg["access"] == "noma" else cfg["solver"]
        if pro:
            v, xr, alloc, hm = evaluate_ps(P_, env.dev, C_plan, best, cfg["order"], sv, cfg["access"],
                                           self.tau_dec, return_alloc=True, rule=pro,
                                           mode_hint=hint if P_.get("rt_hint", rt) else None)
        else:
            v, xr, alloc = evaluate(P_, env.dev, C_plan, best, cfg["order"], sv, cfg["access"],
                                    self.tau_dec, return_alloc=True)
            hm = None
        self.hmode = hm
        if sv == "search":
            bv = max(bv, v)
        dt = time.perf_counter() - t0
        return z, best, xr, alloc, bv, bi, dt, len(cands)

    def _ga(self, ev, pop=16, gens=6, elite=2, pm=None):
        """Evolutionary search over offloading vectors with a convex resource-allocation
        step as fitness (convex-evolutionary alternating method, Shi et al., IEEE TMC 2026)."""
        N, rng = self.N, self.rng
        pm = pm or 1.0 / N
        P = [rng.integers(0, 2, N).astype(np.int8) for _ in range(pop - 2)]
        P.append(np.zeros(N, np.int8))
        P.append(self.ga_best.copy() if self.ga_best is not None else rng.integers(0, 2, N).astype(np.int8))
        cache = {}
        def fit(c):
            k = c.tobytes()
            if k not in cache:
                cache[k] = ev(c)
            return cache[k]
        F = np.array([fit(c) for c in P])
        for _ in range(gens):
            order = np.argsort(-F)
            newP = [P[i].copy() for i in order[:elite]]
            while len(newP) < pop:
                i1, i2 = rng.integers(0, pop, 2); a = P[i1] if F[i1] >= F[i2] else P[i2]
                i1, i2 = rng.integers(0, pop, 2); b = P[i1] if F[i1] >= F[i2] else P[i2]
                m = rng.random(N) < 0.5
                c = np.where(m, a, b).astype(np.int8)
                flip = rng.random(N) < pm
                c[flip] ^= 1
                newP.append(c)
            P = newP
            F = np.array([fit(c) for c in P])
        order = np.argsort(-F)
        self.ga_best = P[order[0]].copy()
        return [P[i] for i in order[:4]]

    def train_step(self):
        if self.generator == "diffusion":
            n = min(self.mem_n, self.mem)
            if n < self.batch:
                return
            ids = self.rng.choice(n, self.batch, replace=False)
            s_ = torch.from_numpy(self.mem_x[ids]); x0 = torch.from_numpy(self.mem_y[ids]) * 2 - 1
            loss = self.diff.loss(s_, x0)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            return
        n = min(self.mem_n, self.mem)
        if n < self.batch:
            return
        ids = self.rng.choice(n, self.batch, replace=False)
        xb = torch.from_numpy(self.mem_x[ids])
        yb = torch.from_numpy(self.mem_y[ids])
        pred = self.actor(xb)
        loss = nn.functional.binary_cross_entropy(pred, yb)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def step(self):
        env = self.env
        rtgc = self.rp.get("rt_fast", False)
        if rtgc:
            gc.disable()                    # garbage collection outside the decision path
        z, best, xr, alloc, bv, bi, dt, nc = self.decide()
        if rtgc:
            gc.enable()
            if env.t % 50 == 0:
                gc.collect(0)
        # true (PQC-aware) coefficients for execution of local part
        C_true = env.coeffs(pqc_aware=True, tau_dec=self.tau_dec,
                            ps=env.ps_state() if self.cfg.get("proactive", False) else None)
        if self.generator in ("dnn", "diffusion"):
            j = self.mem_n % self.mem
            self.mem_x[j] = z
            self.mem_y[j] = best
            self.mem_n += 1
            if self.mem_n % self.train_every == 0:
                self.train_step()
            self.best_k_hist.append(bi if bi < self.K else 0)
            if len(self.best_k_hist) >= self.delta_K and env.t % self.delta_K == 0:
                self.K = min(max(max(self.best_k_hist[-self.delta_K:]) + 2, self.N // 2, 2), self.N)
        info = env.step(xr, alloc, C_true, self.tau_dec, hmode=self.hmode)
        info["dt"] = dt
        info["ncand"] = nc
        info["dpp"] = bv
        return info


# --------------------------------------------------------------------------
class DDPG:
    """Model-free DDPG baseline: action = [offload score, power frac, cpu frac] per WD.
    NOMA with strongest-channel-first SIC, reward = per-frame DPP value."""

    def __init__(self, env, seed=0, lr_a=1e-4, lr_c=1e-3, gamma=0.9, tau=0.005, mem=20000, batch=128,
                 noise=0.2):
        N = env.dev.N
        self.env, self.N = env, N
        din = 4 * N + 1
        self.norm = RunningNorm(din)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed + 7)
        mk = lambda i, o, act: nn.Sequential(nn.Linear(i, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                                             nn.Linear(128, o), *( [nn.Sigmoid()] if act else []))
        self.A, self.At = mk(din, 3 * N, True), mk(din, 3 * N, True)
        self.C, self.Ct = mk(din + 3 * N, 1, False), mk(din + 3 * N, 1, False)
        self.At.load_state_dict(self.A.state_dict()); self.Ct.load_state_dict(self.C.state_dict())
        self.oa = torch.optim.Adam(self.A.parameters(), lr=lr_a)
        self.oc = torch.optim.Adam(self.C.parameters(), lr=lr_c)
        self.g, self.tau, self.batch, self.noise = gamma, tau, batch, noise
        self.buf = dict(s=np.zeros((mem, din), np.float32), a=np.zeros((mem, 3 * N), np.float32),
                        r=np.zeros(mem, np.float32), s2=np.zeros((mem, din), np.float32))
        self.n, self.mem = 0, mem
        self.rscale = None

    def _exec(self, a):
        env, p, dev = self.env, self.env.p, self.env.dev
        N, T = self.N, p["T"]
        s = env.state()
        C = env.coeffs(pqc_aware=True)
        x = (a[:N] > 0.5) & C.can_off
        Pw = a[N:2 * N] * dev.Pmax
        fr = a[2 * N:]
        # local part: CPU at fraction of fmax (or GPU fraction of R)
        D = np.zeros(N); E = np.zeros(N)
        f = fr * dev.fmax
        Dc = np.minimum(1e3 * T * f / dev.phi, s["Q"])
        Ec = dev.kappa * f ** 3 * T
        Dg = np.minimum(fr * dev.R_gpu * T, s["Q"])
        Eg = (dev.P_gpu0 / np.maximum(dev.R_gpu, 1e-9) + dev.zeta) * Dg
        useg = dev.gpu
        D = np.where(~x, np.where(useg, Dg, Dc), 0.0)
        E = np.where(~x, np.where(useg, Eg, Ec), 0.0)
        idx = np.flatnonzero(x)
        o = idx[np.argsort(-s["h_hat"][idx])]
        alloc = None
        if o.size:
            rx = Pw[o] * s["h_hat"][o]
            interf = np.concatenate([np.cumsum(rx[::-1])[::-1][1:], [0.0]])
            sinr = rx / (dev.N0 + interf)
            cap = C.kr[o] * np.log2(1 + sinr)
            Dp = np.clip(cap - C.B[o], 0, s["Q"][o])
            alloc = dict(kind="noma", idx=o, P=Pw[o], D=Dp)
        # reward: DPP value with the planned quantities
        val = np.sum((C.a * D - C.b * E)[~x])
        if o.size:
            val += np.sum(C.a_eff[o] * alloc["D"] - C.b[o] * Pw[o] * C.T_air[o]) - C.fix[o].sum()
        # execution (env applies true channel); local quantities passed via a fake C
        C.loc_D = np.where(useg, Dg, Dc); C.loc_E = np.where(useg, Eg, Ec)
        info = env.step(x, alloc, C)
        return val, info

    def step(self):
        env = self.env
        s = env.state()
        feat = features(env, s); self.norm.update(feat)
        z = self.norm(feat).astype(np.float32)
        t0 = time.perf_counter()
        with torch.no_grad():
            a = self.A(torch.from_numpy(z)).numpy()
        dt = time.perf_counter() - t0
        a = np.clip(a + self.rng.normal(0, self.noise, a.shape), 0, 1)
        r, info = self._exec(a)
        if self.rscale is None:
            self.rscale = max(abs(r), 1.0)
        rn = r / self.rscale
        s2 = env.state(); f2 = self.norm(features(env, s2)).astype(np.float32)
        j = self.n % self.mem
        self.buf["s"][j], self.buf["a"][j], self.buf["r"][j], self.buf["s2"][j] = z, a, rn, f2
        self.n += 1
        self.noise = max(0.02, self.noise * 0.9997)
        if self.n >= self.batch and self.n % 2 == 0:
            self._train()
        info["dt"] = dt; info["ncand"] = 1; info["dpp"] = r
        return info

    def _train(self):
        n = min(self.n, self.mem)
        ids = self.rng.choice(n, self.batch, replace=False)
        S = torch.from_numpy(self.buf["s"][ids]); A = torch.from_numpy(self.buf["a"][ids])
        R = torch.from_numpy(self.buf["r"][ids]).unsqueeze(1); S2 = torch.from_numpy(self.buf["s2"][ids])
        with torch.no_grad():
            y = R + self.g * self.Ct(torch.cat([S2, self.At(S2)], 1))
        lc = nn.functional.mse_loss(self.C(torch.cat([S, A], 1)), y)
        self.oc.zero_grad(); lc.backward(); self.oc.step()
        la = -self.C(torch.cat([S, self.A(S)], 1)).mean()
        self.oa.zero_grad(); la.backward(); self.oa.step()
        with torch.no_grad():
            for t, s_ in ((self.At, self.A), (self.Ct, self.C)):
                for pt, ps in zip(t.parameters(), s_.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * ps)


# --------------------------------------------------------------------------
class _AttnActor(nn.Module):
    def __init__(self, N, d=64):
        super().__init__()
        self.N = N
        self.inp = nn.Linear(5, d)
        self.enc = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.0, batch_first=True)
        self.out = nn.Linear(d, 3)

    def tokens(self, z):
        N = self.N
        per = z[..., :4 * N].reshape(*z.shape[:-1], 4, N).transpose(-1, -2)
        g = z[..., 4 * N:].unsqueeze(-2).expand(*per.shape[:-1], 1)
        return torch.cat([per, g], -1)

    def forward(self, z):
        h = self.enc(self.inp(self.tokens(z)))
        a = torch.sigmoid(self.out(h))                   # (B, N, 3)
        return a.transpose(-1, -2).reshape(*z.shape[:-1], 3 * self.N)


class _AttnCritic(nn.Module):
    def __init__(self, N, d=64):
        super().__init__()
        self.N = N
        self.inp = nn.Linear(8, d)
        self.enc = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.0, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))
        self.tok = _AttnActor(N).tokens

    def forward(self, za):
        N = self.N
        z, a = za[..., :4 * N + 1], za[..., 4 * N + 1:]
        at = a.reshape(*a.shape[:-1], 3, N).transpose(-1, -2)
        h = self.enc(self.inp(torch.cat([self.tok(z), at], -1)))
        return self.head(h.mean(-2))


class LyDDPGAtt(DDPG):
    """Lyapunov-guided DDPG with self-attention over device tokens (in the spirit of
    Chen et al., IEEE TMC 2026); reward = per-frame drift-plus-penalty value."""

    def __init__(self, env, seed=0, **kw):
        super().__init__(env, seed=seed, **kw)
        N = self.N
        torch.manual_seed(seed)
        self.A, self.At = _AttnActor(N), _AttnActor(N)
        self.C, self.Ct = _AttnCritic(N), _AttnCritic(N)
        self.At.load_state_dict(self.A.state_dict()); self.Ct.load_state_dict(self.C.state_dict())
        self.oa = torch.optim.Adam(self.A.parameters(), lr=1e-4)
        self.oc = torch.optim.Adam(self.C.parameters(), lr=1e-3)


# --------------------------------------------------------------------------
def run(p, scheme="PALM", frames=10000, seed=0, profile=None, generator="dnn", tau_dec=None,
        log_every=1):
    env = Env(p, profile, seed=seed)
    if scheme == "DDPG":
        ag = DDPG(env, seed=seed)
    elif scheme == "LyDDPG-Att":
        ag = LyDDPGAtt(env, seed=seed)
    else:
        if isinstance(scheme, str) and scheme in GENERATOR and generator == "dnn":
            generator = GENERATOR[scheme]
        ag = LyDROO(env, scheme, seed=seed, generator=generator, tau_dec=tau_dec)
    N = p["N"]
    keys = ["Q", "Y", "E", "wrate", "off", "dt", "Ees", "hs", "D", "hpro"]
    log = {k: [] for k in keys}
    perdev_E = np.zeros(N); perdev_D = np.zeros(N); perdev_off = np.zeros(N)
    for t in range(frames):
        info = ag.step()
        log["Q"].append(env.Q.mean())
        log["Y"].append(env.Y.mean())
        log["E"].append(info["E"].mean() / p["T"])            # W
        log["Ees"].append(info["E_es"] / p["T"])              # W
        log["wrate"].append(info["wrate"])                   # Mbps
        log["off"].append(info["x"].mean())
        log["dt"].append(info["dt"])
        log["hs"].append(info["hs"])
        log["hpro"].append(info.get("hpro", 0))
        log["D"].append(info["D"].sum() / p["T"])
        perdev_E += info["E"]; perdev_D += info["D"]; perdev_off += info["x"]
    out = {k: np.asarray(v) for k, v in log.items()}
    out["perdev_P"] = perdev_E / (frames * p["T"])           # W
    out["perdev_Jbit"] = perdev_E * 1e-3 / np.maximum(perdev_D * 1e3, 1e-9)  # J/bit
    out["perdev_off"] = perdev_off / frames
    out["perdev_D"] = perdev_D / (frames * p["T"])           # Mbps
    out["Qfinal"] = env.Q.copy()
    return out
