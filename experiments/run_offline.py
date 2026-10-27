"""
Offline (per-frame) experiments on states sampled from an online PALM run.
    S   : sample states (+ exhaustive-search benchmark of the offloading decision)
    E2  : optimality of the power-allocation solvers (incl. all SIC orders, small M)
    E9  : binary vs partial offloading (per-frame DPP value)
    E10 : QAOA candidate generation (state-vector + noisy pyqpanda3 simulation)
    E4  : runtime scaling of solvers and of the full per-frame decision
"""
import sys, os, pickle, time, itertools, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from palm.params import default_params, with_
from palm.system import Env, evaluate, build_coeffs
from palm.agents import LyDROO, features
from palm import solver as S
from palm.solver import warmup

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
BASE = default_params(); BASE["lam"] = 2.5
warmup()


def sample_states(p=BASE, frames=5000, every=20, start=1000, seed=7, exhaustive=True):
    env = Env(p, seed=seed)
    ag = LyDROO(env, "PALM", seed=seed)
    allx = np.array(list(itertools.product([0, 1], repeat=p["N"])), np.int8)
    rec = []
    for t in range(frames):
        if t >= start and t % every == 0:
            s = env.state()
            C = build_coeffs(p, env.dev, s["h_hat"], s["Q"], s["Y"], s["Y0"], s["need_hs"])
            z, best, xr, alloc, bv, bi, dt, nc = ag.decide()
            r = dict(state=s, palm_x=best, palm_v=bv, ncand=nc)
            if exhaustive:
                vals = np.array([evaluate(p, env.dev, C, x) for x in allx])
                # refine the 8 best alpha-order candidates with SIC-order search
                top = np.argsort(-vals)[:8]
                ref = np.array([evaluate(p, env.dev, C, allx[j], solver="search") for j in top])
                r["opt_v"] = max(vals.max(), ref.max())
                r["opt_x"] = allx[top[ref.argmax()]] if ref.max() >= vals.max() else allx[vals.argmax()]
                r["allvals"] = vals
                # decisions of channel-ordered greedy / SCA critics evaluated exactly
                for nm, (o, sv) in dict(greedy=("channel", "greedy"), sca=("channel", "sca")).items():
                    vv = np.array([evaluate(p, env.dev, C, x, o, sv) for x in allx])
                    xb = allx[np.nanargmax(np.where(np.isfinite(vv), vv, -np.inf))]
                    r[f"{nm}_x"] = xb; r[f"{nm}_true_v"] = evaluate(p, env.dev, C, xb)
            rec.append(r)
        ag.step()
    with open(os.path.join(OUT, "states.pkl"), "wb") as f:
        pickle.dump(dict(p=p, rec=rec), f)
    gap = np.array([(r["opt_v"] - r["palm_v"]) / abs(r["opt_v"]) for r in rec])
    print("PALM decision vs exhaustive: mean rel gap %.2e, 95%% %.2e, exact-hit %.3f" %
          (gap.mean(), np.quantile(gap, .95), np.mean(gap < 1e-9)))


def _rand_start(L, U, pbar, rng):
    M = len(L); y = np.empty(M); nxt = 1.0
    for k in range(M - 1, -1, -1):
        lo = L[k] * nxt; hi = min(U[k] * nxt, nxt + pbar[k])
        if hi < lo:
            return None
        y[k] = lo + rng.uniform(0.01, 0.99) * (hi - lo); nxt = y[k]
    return y


def E2(n_inst=400, seed=0):
    """Power allocation for a fixed offloading set: exact (weight order) vs
    greedy / SCA (channel order) vs best over ALL M! decoding orders
    (multi-start SCA per order)."""
    d = pickle.load(open(os.path.join(OUT, "states.pkl"), "rb"))
    p = d["p"]; env = Env(p); dev = env.dev
    rng = np.random.default_rng(seed)
    rows = []
    for it in range(n_inst):
        r = d["rec"][rng.integers(len(d["rec"]))]
        s = r["state"]
        C = build_coeffs(p, dev, s["h_hat"], s["Q"], s["Y"], s["Y0"], s["need_hs"])
        cand = np.flatnonzero(C.can_off)
        M = int(rng.integers(2, 7))
        if len(cand) < M:
            continue
        idx = rng.choice(cand, M, replace=False)
        def coef(o):
            return (C.alpha[o], C.beta[o], C.L[o], C.U[o], C.pbar[o])
        ow = idx[np.argsort(C.alpha[idx])]
        oc = idx[np.argsort(-C.h_hat[idx])]
        _, v_ex, st = S.solve_exact(*coef(ow))
        if not np.isfinite(v_ex):
            continue
        _, v_gw, _ = S.solve_greedy_val(*coef(ow))
        _, v_gc, _ = S.solve_greedy_val(*coef(oc))
        _, v_sc, _, its = S.solve_sca(*coef(oc))
        _, v_an, _, _ = S.solve_anns(*coef(oc))
        _, _, v_srch = S.solve_order_search(C.alpha[idx], C.beta[idx], C.L[idx], C.U[idx], C.pbar[idx])
        _, _, v_full = S.solve_order_search(C.alpha[idx], C.beta[idx], C.L[idx], C.U[idx], C.pbar[idx], 3, None)
        ob = idx[np.argsort(C.beta[idx])]
        _, v_beta, _, _ = S.solve_sca(*coef(ob))
        ub = S.upper_bound_interference_free(*coef(ow))
        # global over all orders
        best_any = -np.inf
        for perm in itertools.permutations(idx):
            o = np.array(perm)
            a = coef(o)
            _, v, _, _ = S.solve_sca(*a)
            best_any = max(best_any, v)
            for _ in range(2):
                y0 = _rand_start(a[2], a[3], a[4], rng)
                if y0 is None:
                    continue
                y, v2, _ = S._sca_iterations(*[np.asarray(x, float) for x in a], y0, 100, 1e-7)
                best_any = max(best_any, v2)
        best_any = max(best_any, v_srch, v_full)
        rows.append(dict(M=M, exact=v_ex, greedy_w=v_gw, greedy_c=v_gc, sca_c=v_sc, anns_c=v_an, sca_its=its,
                         beta_sca=v_beta, search=v_srch, search_full=v_full, ub=ub, global_best=best_any))
    with open(os.path.join(OUT, "E2.pkl"), "wb") as f:
        pickle.dump(rows, f)
    g = lambda k: np.array([(x["global_best"] - x[k]) / abs(x["global_best"]) for x in rows])
    print("E2 n=%d" % len(rows))
    for k in ["exact", "search", "search_full", "beta_sca", "greedy_w", "greedy_c", "sca_c", "anns_c"]:
        v = g(k); v = v[np.isfinite(v)]
        print(f"  {k:9s} mean gap {v.mean():.3e} max {v.max():.3e}  within1e-6 {np.mean(v < 1e-6):.3f}")


def evaluate_partial(p, dev, C, x, iters=4):
    """Partial offloading: devices with x=1 may compute locally AND offload.
    Block-coordinate ascent between the (convex) NOMA block and the local block."""
    x = np.asarray(x).astype(bool) & C.can_off
    idx = np.flatnonzero(x)
    base = evaluate(p, dev, C, x)
    if idx.size == 0 or not np.isfinite(base):
        return base
    o = idx[np.argsort(C.alpha[idx])]
    Q = C.Q.copy()
    T = p["T"]
    DL = np.zeros(len(o))
    best = base
    for _ in range(iters):
        Ue = np.exp2(np.minimum((np.maximum(Q[o] - DL, 0) + C.B[o]) / C.kr[o], 60.0))
        Ue = np.maximum(Ue, C.L[o] * (1 + 1e-9))
        y, v, st = S.solve_exact(C.alpha[o], C.beta[o], C.L[o], Ue, C.pbar[o])
        if not np.isfinite(v):
            break
        yn = np.append(y[1:], 1.0)
        DO = C.kr[o] * np.log2(y / yn) - C.B[o]
        # local block with remaining queue (CPU or GPU, closed form)
        a, b = C.a[o], C.b[o]
        rem = np.maximum(Q[o] - DO, 0)
        kb = 1e3 * T / dev.phi[o]
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(b > 0, np.sqrt(a * 1e3 / (3 * dev.phi[o] * np.maximum(b, 1e-300) * dev.kappa[o])), dev.fmax[o])
        f = np.clip(f, 0, np.minimum(dev.fmax[o], rem / kb))
        vc = a * kb * f - b * dev.kappa[o] * f ** 3 * T
        epk = np.where(dev.gpu[o], dev.P_gpu0[o] / np.maximum(dev.R_gpu[o], 1e-9) + dev.zeta[o], 0)
        dg = np.where(dev.gpu[o], np.minimum(rem, dev.R_gpu[o] * T), 0)
        vg = np.where(dev.gpu[o] & (a - b * epk > 0), (a - b * epk) * dg, -np.inf)
        vl = np.maximum(vc, np.maximum(vg, 0))
        DL = np.where(vg > vc, dg, kb * f)
        val = C.loc_val[~x].sum() - C.fix[o].sum() + v + vl.sum()
        best = max(best, val)
    return best


def E9(n=150):
    d = pickle.load(open(os.path.join(OUT, "states.pkl"), "rb"))
    p = d["p"]; dev = Env(p).dev
    allx = np.array(list(itertools.product([0, 1], repeat=p["N"])), np.int8)
    rows = []
    for r in d["rec"][:n]:
        s = r["state"]
        C = build_coeffs(p, dev, s["h_hat"], s["Q"], s["Y"], s["Y0"], s["need_hs"])
        # restrict partial search to the 64 best binary candidates (sufficient in practice)
        top = allx[np.argsort(-r["allvals"])[:64]]
        vp = max(evaluate_partial(p, dev, C, x) for x in top)
        rows.append(dict(binary=r["opt_v"], partial=vp))
    with open(os.path.join(OUT, "E9.pkl"), "wb") as f:
        pickle.dump(rows, f)
    g = np.array([(x["partial"] - x["binary"]) / abs(x["binary"]) for x in rows])
    print("E9 partial vs binary DPP gain: mean %.3e  95%% %.3e  max %.3e" % (g.mean(), np.quantile(g, .95), g.max()))


def E10(n=60, Nq=8, shots=1024, K=8, seed=0):
    """QAOA candidates on N=8 sub-problems (first 8 WDs of sampled states)."""
    from palm import qaoa as QA
    d = pickle.load(open(os.path.join(OUT, "states.pkl"), "rb"))
    p = with_(d["p"])
    dev = Env(p).dev
    rng = np.random.default_rng(seed)
    allx8 = np.array(list(itertools.product([0, 1], repeat=Nq)), np.int8)
    rows = []
    for r in d["rec"][:n]:
        s = r["state"]
        C = build_coeffs(p, dev, s["h_hat"], s["Q"], s["Y"], s["Y0"], s["need_hs"])
        idx = np.arange(Nq)
        def full(xs):  # embed: remaining WDs compute locally
            x = np.zeros(p["N"], np.int8); x[idx] = xs; return x
        vals = np.array([evaluate(p, dev, C, full(x)) for x in allx8])
        opt = vals.max(); worst = vals[np.isfinite(vals)].min()
        Q = QA.build_qubo(p, dev, C, idx)
        sv = QA.qubo_value(Q, allx8)
        sur_x = allx8[np.argmax(sv)]
        hz, J, c0 = QA.ising(Q)
        row = dict(opt=opt, worst=worst, f0=Q["f0"], qubo_argmax_true=evaluate(p, dev, C, full(sur_x)))
        for pl in [1, 2, 3]:
            g, b, _ = QA.train_qaoa(hz, J, pl, restarts=6, seed=seed)
            bits = QA.sample_statevector(hz, J, g, b, shots, rng)
            cands = QA.candidates_from_samples(bits, K=K, Q=Q)
            row[f"qaoa_p{pl}"] = max(evaluate(p, dev, C, full(x)) for x in cands)
            # probability of sampling the true optimum
            psi, _ = QA.statevector_qaoa(hz, J, g, b)
            kopt = int(sum(int(v) << i for i, v in enumerate(allx8[vals.argmax()])))
            row[f"popt_p{pl}"] = float(np.abs(psi[kopt]) ** 2)
            if pl == 2:
                for dp in [0.001, 0.005, 0.01]:
                    bb, w = QA.sample_pyqpanda(hz, J, g, b, shots, depol=dp)
                    cn = QA.candidates_from_samples(bb, w, K=K, Q=Q)
                    row[f"noisy_{dp}"] = max(evaluate(p, dev, C, full(x)) for x in cn)
        rnd = rng.integers(0, 2, (K, Nq))
        row["random"] = max(evaluate(p, dev, C, full(x)) for x in rnd)
        rows.append(row)
    with open(os.path.join(OUT, "E10.pkl"), "wb") as f:
        pickle.dump(rows, f)
    nr = lambda k: np.array([(x[k] - x["f0"]) / (x["opt"] - x["f0"]) if x["opt"] > x["f0"] else 1.0 for x in rows])
    for k in ["qubo_argmax_true", "qaoa_p1", "qaoa_p2", "qaoa_p3", "noisy_0.001", "noisy_0.005", "noisy_0.01", "random"]:
        v = nr(k); print(f"E10 {k:18s} normalised gain mean {v.mean():.3f}  hit-opt {np.mean(v > 1 - 1e-9):.2f}")
    for pl in [1, 2, 3]:
        print("  P(opt) p=%d: %.3f (uniform %.4f)" % (pl, np.mean([x[f'popt_p{pl}'] for x in rows]), 1 / 2 ** Nq))


def E4(reps=30):
    """Runtime: solvers vs M; full decision vs N."""
    import cvxpy as cp
    rng = np.random.default_rng(0)
    if os.environ.get("SKIP_SOLVERS"):
        res = json.load(open(os.path.join(OUT, "E4.json")))
        Ms = []
    else:
        res = None
    res = res or dict(M=[], exact=[], greedy=[], sca_numba=[], sca_cvx=[], convex_cvx=[], sca_its=[], search=[])
    p0 = BASE
    for M in ([] if os.environ.get("SKIP_SOLVERS") else [5, 10, 15, 20, 25, 30, 35, 40, 50]):
        te, tg, ts, tc, tcv, its, tsr = [], [], [], [], [], [], []
        for r in range(reps):
            N = M
            p = with_(p0, N=N, classes=["iot"] * N)
            env = Env(p, seed=r)
            dev = env.dev
            Q = rng.uniform(50, 400, N); Y = rng.uniform(0, 300, N)
            C = build_coeffs(p, dev, env.h_hat, Q, Y, 0.0, np.zeros(N, bool))
            idx = np.arange(N)
            ow = idx[np.argsort(C.alpha)]; oc = idx[np.argsort(-C.h_hat)]
            a = [np.ascontiguousarray(z) for z in (C.alpha[ow], C.beta[ow], C.L[ow], C.U[ow], C.pbar[ow])]
            ac = [np.ascontiguousarray(z) for z in (C.alpha[oc], C.beta[oc], C.L[oc], C.U[oc], C.pbar[oc])]
            t = time.perf_counter(); S.solve_exact(*a); te.append(time.perf_counter() - t)
            t = time.perf_counter(); S.solve_greedy_val(*ac); tg.append(time.perf_counter() - t)
            t = time.perf_counter(); _, _, _, k = S.solve_sca(*ac); ts.append(time.perf_counter() - t); its.append(k)
            t = time.perf_counter(); S.solve_order_search(C.alpha, C.beta, C.L, C.U, C.pbar); tsr.append(time.perf_counter() - t)
            if r < 8:
                # SCA with a generic conic solver (as in the original submission), P-domain,
                # DPP-compliant CVXPY model compiled once and re-solved with Clarabel.
                h = C.h_hat[oc]; al = C.alpha[oc]; be = C.beta[oc]; N0 = dev.N0
                Ainc = np.triu(np.tile(h / N0, (N, 1)))          # row k: sum_{j>=k}
                Astr = np.triu(np.tile(h / N0, (N, 1)), 1)       # row k: sum_{j>k}
                Pv = cp.Variable(N); g = cp.Parameter(N, nonneg=True)
                Lc, Uc = C.L[oc] - 1, np.minimum(C.U[oc], 1e12) - 1
                obj = al @ cp.log(1 + Ainc @ Pv) - cp.multiply(al, g) @ (Astr @ Pv) - (be * h / N0) @ Pv
                cons = [Pv >= 0, Pv <= dev.Pmax[oc],
                        cp.multiply(h / N0, Pv) <= cp.multiply(Uc, 1 + Astr @ Pv),
                        cp.multiply(h / N0, Pv) >= cp.multiply(Lc, 1 + Astr @ Pv)]
                prob = cp.Problem(cp.Maximize(obj), cons)
                y0, _ = S._feasible_start(*[np.ascontiguousarray(z) for z in (C.L[oc], C.U[oc], C.pbar[oc])])
                P0 = S.y_to_power(y0, h, N0)
                t = time.perf_counter()
                prev = -np.inf
                for it in range(100):
                    g.value = 1.0 / (1.0 + Astr @ np.clip(P0, 0, None))
                    prob.solve(solver="CLARABEL")
                    if Pv.value is None:
                        break
                    P0 = Pv.value
                    if abs(prob.value - prev) < 1e-6 * max(1, abs(prev)):
                        break
                    prev = prob.value
                tc.append(time.perf_counter() - t)
                # the convex (weight-ordered) problem with a generic solver
                yv = cp.Variable(N); w = np.r_[a[0][0], np.diff(a[0])]; cc = np.r_[a[1][0], np.diff(a[1])]
                yn = cp.hstack([yv[1:], np.ones(1)])
                pr = cp.Problem(cp.Maximize(w @ cp.log(yv) - cc @ yv),
                                [yv >= cp.multiply(a[2], yn), yv <= cp.multiply(np.minimum(a[3], 1e12), yn), yv - yn <= a[4]])
                t = time.perf_counter(); pr.solve(solver="CLARABEL"); tcv.append(time.perf_counter() - t)
        res["M"].append(M)
        for k, v in [("exact", te), ("greedy", tg), ("sca_numba", ts), ("sca_cvx", tc), ("convex_cvx", tcv), ("sca_its", its), ("search", tsr)]:
            res[k].append((float(np.median(v)), float(np.std(v))))
        print(M, {k: res[k][-1] for k in res if k != "M"}, flush=True)
    # full per-frame decision time of the online agent vs N
    dec = {}
    for sch in ["PALM", "PALM-G", "LyDROO-SCA", "LyDROO-TDMA"]:
        dec[sch] = {}
        for N in [10, 20, 30, 40, 50]:
            heavy = sch == "LyDROO-SCA"
            if heavy and N > 30:
                continue
            p = with_(p0, N=N, classes=(["iot"] * 6 + ["jetson"] * 4) * (N // 10), lam=2.5 * 10 / N)
            env = Env(p, seed=1); ag = LyDROO(env, sch, seed=1)
            dts = []
            nf, warm = (120, 60) if heavy else (400, 200)
            for t in range(nf):
                info = ag.step()
                if t >= warm:
                    dts.append(info["dt"])
            dec[sch][str(N)] = (float(np.median(dts) * 1e3), float(np.percentile(dts, 99) * 1e3))
        print(sch, dec[sch], flush=True)
    res["decision_ms"] = dec
    with open(os.path.join(OUT, "E4.json"), "w") as f:
        json.dump(res, f, indent=1)
    prof = {s: {n: v[0] for n, v in d_.items()} for s, d_ in dec.items()}
    prof["platform"] = "cloud Xeon 2.1GHz, 1 thread"
    prof["platform_factor"] = 1.0
    with open(os.path.join(OUT, "latency_profile.json"), "w") as f:
        json.dump(prof, f, indent=1)


def E4b(reps=30):
    """Runtime of the recent baselines added in the revision: ANNS solver vs M and
    per-frame decision time of NOMA-ANNS, LyDiff, CEA, and LyDDPG-Att vs N."""
    from palm.agents import LyDDPGAtt, GENERATOR
    res = json.load(open(os.path.join(OUT, "E4.json")))
    res["anns"] = []
    for M in res["M"]:
        ta = []
        for r in range(reps):
            p = with_(BASE, N=M, classes=["iot"] * M)
            env = Env(p, seed=r); dev = env.dev
            rng = np.random.default_rng(1000 + r)
            Q = rng.uniform(50, 400, M); Y = rng.uniform(0, 300, M)
            C = build_coeffs(p, dev, env.h_hat, Q, Y, 0.0, np.zeros(M, bool))
            oc = np.argsort(-C.h_hat)
            ac = [np.ascontiguousarray(z) for z in (C.alpha[oc], C.beta[oc], C.L[oc], C.U[oc], C.pbar[oc])]
            t = time.perf_counter(); S.solve_anns(*ac); ta.append(time.perf_counter() - t)
        res["anns"].append((float(np.median(ta)), float(np.std(ta))))
    print("anns", res["anns"], flush=True)
    dec = res["decision_ms"]
    for sch in ["LyDiff", "CEA", "LyDDPG-Att"]:
        dec[sch] = {}
        for N in [10, 20, 30, 40, 50]:
            p = with_(BASE, N=N, classes=(["iot"] * 6 + ["jetson"] * 4) * (N // 10), lam=2.5 * 10 / N)
            env = Env(p, seed=1)
            ag = LyDDPGAtt(env, seed=1) if sch == "LyDDPG-Att" else LyDROO(env, sch, seed=1, generator=GENERATOR.get(sch, "dnn"))
            dts = []
            for t in range(400):
                info = ag.step()
                if t >= 200:
                    dts.append(info["dt"])
            dec[sch][str(N)] = (float(np.median(dts) * 1e3), float(np.percentile(dts, 99) * 1e3))
        print(sch, dec[sch], flush=True)
    with open(os.path.join(OUT, "E4.json"), "w") as f:
        json.dump(res, f, indent=1)


def E4c(frames=400, warm=200):
    """Per-frame decision time vs N for the main configuration of the revision
    (1 s sessions, closed-form prior, 1 ms order-refinement budget), all schemes.
    Run alone (no other load).  Also writes results/latency_profile_v2.json with
    Jetson latencies estimated as Jetson(measured, earlier implementation) x
    Xeon(new)/Xeon(earlier) per scheme and N."""
    from palm.agents import LyDDPGAtt, GENERATOR
    B2 = with_(BASE, rekey_frames=20); B2["fast_prior"] = True; B2["ref_budget"] = 1e-3
    res = json.load(open(os.path.join(OUT, "E4.json")))
    dec = {}
    for sch in ["PALM-PS", "PALM-G-PS", "PALM", "PALM-G", "LyDROO-SCA", "LyDROO-TDMA", "LyDiff", "CEA", "LyDDPG-Att"]:
        dec[sch] = {}
        for N in [10, 20, 30, 40, 50]:
            if sch == "LyDROO-SCA" and N > 30:
                continue
            p = with_(B2, N=N, classes=(["iot"] * 6 + ["jetson"] * 4) * (N // 10), lam=2.5 * 10 / N)
            env = Env(p, seed=1)
            ag = LyDDPGAtt(env, seed=1) if sch == "LyDDPG-Att" else LyDROO(env, sch, seed=1, generator=GENERATOR.get(sch, "dnn"))
            dts = []
            for t in range(frames):
                info = ag.step()
                if t >= warm:
                    dts.append(info["dt"])
            dec[sch][str(N)] = (float(np.median(dts) * 1e3), float(np.percentile(dts, 99) * 1e3))
        print(sch, dec[sch], flush=True)
    res["decision_ms_v1"] = res.get("decision_ms_v1", res["decision_ms"])
    old = res["decision_ms_v1"]
    res["decision_ms"] = dec
    with open(os.path.join(OUT, "E4.json"), "w") as f:
        json.dump(res, f, indent=1)
    jet = json.load(open(os.path.join(OUT, "latency_profile_jetson.json")))
    prof = {"platform_factor": 1.0, "note": "Jetson latency estimated: Jetson(earlier impl.) x Xeon(new)/Xeon(earlier)"}
    for sch, d in dec.items():
        base = sch.replace("-PS", "")
        prof[sch] = {}
        for N, (med, _) in d.items():
            if base in jet and N in jet[base] and base in old and N in old[base]:
                prof[sch][N] = jet[base][N] * med / old[base][N][0]
    with open(os.path.join(OUT, "latency_profile_v2.json"), "w") as f:
        json.dump(prof, f, indent=1)



def E4d(frames=400, warm=200):
    """Decision time vs N of the real-time configuration (PALM-PS-RT, PALM-RT) on this
    machine; merged into results/E4.json["decision_ms"]."""
    B2 = with_(BASE, rekey_frames=20); B2["fast_prior"] = True; B2["ref_budget"] = 1e-3
    res = json.load(open(os.path.join(OUT, "E4.json")))
    for sch in ["PALM-PS-RT", "PALM-RT"]:
        d = {}
        for N in [10, 20, 30, 40, 50]:
            p = with_(B2, N=N, classes=(["iot"] * 6 + ["jetson"] * 4) * (N // 10), lam=2.5 * 10 / N)
            env = Env(p, seed=1); ag = LyDROO(env, sch, seed=1)
            dts = []
            for t in range(frames):
                info = ag.step()
                if t >= warm:
                    dts.append(info["dt"])
            d[str(N)] = (float(np.median(dts) * 1e3), float(np.percentile(dts, 99) * 1e3))
        res["decision_ms"][sch] = d
        print(sch, d, flush=True)
    with open(os.path.join(OUT, "E4.json"), "w") as f:
        json.dump(res, f, indent=1)

if __name__ == "__main__":
    for e in sys.argv[1:]:
        if e == "S":
            sample_states()
        else:
            globals()[e]()
