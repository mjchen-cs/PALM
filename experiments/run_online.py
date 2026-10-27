"""
Time-domain (online) experiments.  Usage:
    python experiments/run_online.py E1   (convergence / baseline comparison)
    python experiments/run_online.py E3   (load sweep)
    ...
Results are stored in results/<exp>.pkl  (list of (config, summary, traces)).
"""
import sys, os, pickle, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from palm.params import default_params, with_
from palm.parallel import pmap, summarize

FR = int(os.environ.get("FRAMES", 10000))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3").split(",")]
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
BASE = default_params()
BASE["lam"] = 2.5
# main configuration of the revision: 1 s sessions, PALM-PS as the main method,
# closed-form prior (exact, faster) and a 1 ms order-refinement budget
BASE["rekey_frames"] = 20
BASE["fast_prior"] = True
BASE["ref_budget"] = 1e-3
MAIN = ["PALM-PS", "PALM-G-PS", "PALM", "PALM-G", "LyDROO-SCA", "LyDROO-Greedy", "LyDROO-TDMA"]


def go(name, cfgs, keep_traces=False):
    """cfgs: list of (tag dict, params, scheme, kwargs)"""
    jobs, tags = [], []
    prof_path = os.environ.get("PROFILE")
    prof = json.load(open(prof_path)) if prof_path else None
    for tag, p, sch, kw in cfgs:
        if prof is not None and "profile" not in kw:
            kw = dict(kw, profile=prof)
        for sd in SEEDS:
            jobs.append((p, sch, FR, sd, kw)); tags.append(dict(tag, seed=sd, scheme=sch))
    t0 = time.time()
    res = pmap(jobs, procs=int(os.environ.get("PROCS", 2)))
    out = []
    for tg, o in zip(tags, res):
        s = summarize(o)
        s.update(perdev_P=o["perdev_P"], perdev_Jbit=o["perdev_Jbit"], perdev_off=o["perdev_off"],
                 perdev_D=o["perdev_D"])
        tr = None
        if keep_traces:
            tr = {k: o[k] for k in ("Q", "Y", "E", "wrate", "off", "Ees", "dt")}
        out.append((tg, s, tr))
    with open(os.path.join(OUT, f"{name}.pkl"), "wb") as f:
        pickle.dump(out, f)
    print(name, "done in %.0fs" % (time.time() - t0), flush=True)


def E1():   # convergence & comparison with baselines incl. DDPG
    cfgs = [({}, BASE, s, {}) for s in
            MAIN + ["PQC-agnostic", "KeepAlive", "DDPG", "LyDiff", "CEA", "LyDDPG-Att"]]
    go("E1", cfgs, keep_traces=True)


def E3():   # load sweep
    cfgs = []
    for lam in [1.5, 2.0, 2.5, 2.75, 3.0, 3.25]:
        for s in MAIN + ["KeepAlive", "LyDiff", "CEA", "LyDDPG-Att"]:
            cfgs.append((dict(lam=lam), with_(BASE, lam=lam), s, {}))
    go("E3", cfgs)


def E6():   # PQC policy: security level x session lifetime, PQC-aware vs agnostic
    cfgs = []
    for lev in [1, 3, 5]:
        for rk in [1, 5, 20, 200]:
            for s in ["PALM-PS", "PALM", "PQC-agnostic"]:
                cfgs.append((dict(sec=lev, rekey=rk), with_(BASE, sec_level=lev, rekey_frames=rk), s, {}))
    go("E6", cfgs)


def E6b():  # AEAD packet size (eta), static secure-module power, amortisation
    cfgs = []
    for pk in [128, 256, 512, 1500]:
        cfgs.append((dict(pkt=pk), with_(BASE, pkt_bytes=pk, rekey_frames=20), "PALM-PS", {}))
    for ps in [0.0, 0.005, 0.01, 0.02, 0.04]:
        p = with_(BASE, rekey_frames=20)
        cfgs.append((dict(pst=ps), p, "PALM-PS", {"profile": {"iot": {"3": {"P_static": ps}}}}))
    for om in [1.0, 0.2, 0.05, 0.0]:
        cfgs.append((dict(om=om), with_(BASE, amortize=om, rekey_frames=20), "PALM-PS", {}))
    go("E6b", cfgs)


def E7():   # robustness: imperfect SIC, CSI error, heterogeneity, ES budget
    cfgs = []
    for xi in [0.0, 0.01, 0.02, 0.05, 0.1]:
        for rm in ["adaptive", "fixed"]:
            cfgs.append((dict(xi=xi, rm=rm), with_(BASE, sic_residual=xi, rate_mode=rm), "PALM-PS", {}))
    for sd in [0.0, 0.05, 0.1, 0.2]:
        for bo in [0.0, 1.0]:
            cfgs.append((dict(csi=sd, bo=bo), with_(BASE, csi_err_std=sd, rate_mode="fixed", backoff=bo * 2 * sd), "PALM-PS", {}))
        # robust rate selection (per-user margins, handshake-aware) with a planning back-off of sigma
        cfgs.append((dict(csi=sd, bo=2.0), with_(BASE, csi_err_std=sd, rate_mode="fixed", backoff=sd,
                                                 robust_rate=True), "PALM-PS", {}))
    for mix in [10, 6, 2]:
        cls = ["iot"] * mix + ["jetson"] * (10 - mix)
        for s in ["PALM-PS", "PALM", "LyDROO-Greedy", "LyDROO-TDMA"]:
            cfgs.append((dict(mix=mix), with_(BASE, classes=cls), s, {}))
    for eg in [1.0, 2.0, 4.0, 8.0]:
        cfgs.append((dict(esg=eg), with_(BASE, es_gamma=eg), "PALM-PS", {}))
    go("E7", cfgs)


def E7b():  # small residual-SIC levels and SIC-aware robust access selection
    cfgs = []
    for xi in [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]:
        cfgs.append((dict(xi=xi, aware=0), with_(BASE, sic_residual=xi), "PALM-PS", {}))
        cfgs.append((dict(xi=xi, aware=1), with_(BASE, sic_residual=xi, sic_aware=True), "PALM-PS", {}))
    cfgs.append((dict(xi=-1, aware=0), BASE, "LyDROO-TDMA", {}))
    go("E7b", cfgs)


def E8():   # V x nu sweep
    cfgs = []
    for V in [5, 20, 50, 200, 500]:
        for nu in [2, 5, 20, 50, 200]:
            cfgs.append((dict(V=V, nu=nu), with_(BASE, V=V, nu=nu), "PALM-PS", {}))
    go("E8", cfgs)


def E5():   # deadline-aware: decision latency consumes airtime (T sweep, N=20)
    import json
    # prefer decision latencies measured on the Jetson Orin Nano (edge controller)
    pj = os.path.join(OUT, "latency_profile_jetson.json")
    pv2 = os.path.join(OUT, "latency_profile_jetson_v2.json")      # measured on the Jetson (v2 code)
    if not os.path.exists(pv2):
        pv2 = os.path.join(OUT, "latency_profile_v2.json")          # estimate
    prof = json.load(open(pv2 if os.path.exists(pv2) else (pj if os.path.exists(pj) else os.path.join(OUT, "latency_profile.json"))))
    cfgs = []
    for T in [50.0, 100.0, 200.0]:
        prt = os.path.join(OUT, "latency_profile_jetson_rt.json")    # real-time configuration (Jetson)
        if os.path.exists(prt):
            prof.update({k: v for k, v in json.load(open(prt)).items() if isinstance(v, dict)})
        for s in ["PALM-PS", "PALM-PS-RT", "PALM-G-PS", "PALM", "PALM-RT", "PALM-G", "LyDROO-SCA", "LyDROO-TDMA",
                  "PALM-PS-ideal"]:
            sch = "PALM-PS" if s == "PALM-PS-ideal" else s
            if sch not in prof:
                continue
            td = 0.0 if s == "PALM-PS-ideal" else prof[sch]["20"] * prof.get("platform_factor", 1.0)
            p = with_(BASE, N=20, classes=["iot"] * 12 + ["jetson"] * 8, T=T, lam=1.5,
                      rekey_frames=int(1000 / T), idle_timeout=int(2000 / T))
            cfgs.append((dict(T=T, tau=td, label=s), p, sch, {"tau_dec": td}))
    go("E5", cfgs)


def E11():  # proactive sessions: session lifetime, zeta, coverage condition
    cfgs = []
    S = ["PALM", "PALM-PS", "KeepAlive"]
    for rk in [1, 2, 5, 10, 20, 50, 200]:                     # (a) session lifetime
        for s in S:
            cfgs.append((dict(part="a", rekey=rk), with_(BASE, rekey_frames=rk), s, {}))
    for z in [0.25, 0.5, 2.0]:                               # (d) session-value weight zeta
        p = with_(BASE, rekey_frames=5); p["ps_zeta"] = z
        cfgs.append((dict(part="d", zeta=z), p, "PALM-PS", {}))
    for rk in [2, 5]:                                        # (e) coverage condition
        for km in [0.0, 2.0]:
            p = with_(BASE, rekey_frames=rk); p["ps_kmin"] = km
            cfgs.append((dict(part="e", rekey=rk, kmin=km), p, "PALM-PS", {}))
    go("E11", cfgs)


def E12():  # all methods under short sessions (where session management matters)
    cfgs = []
    S = ["PALM-PS", "PALM-G-PS", "PALM", "PALM-G", "LyDROO-SCA", "LyDROO-Greedy", "LyDROO-TDMA",
         "PQC-agnostic", "DDPG", "LyDiff", "CEA", "LyDDPG-Att", "KeepAlive"]
    for rk in [5]:
        for s in S:
            cfgs.append((dict(rekey=rk), with_(BASE, rekey_frames=rk), s, {}))
    go("E12", cfgs)



def E13():  # real-time configuration in the main setting (cost of the latency reductions)
    cfgs = [(dict(rekey=20), BASE, s, {}) for s in ["PALM-PS-RT", "PALM-RT"]]
    cfgs.append((dict(rekey=5), with_(BASE, rekey_frames=5), "PALM-PS-RT", {}))
    go("E13", cfgs, keep_traces=True)

if __name__ == "__main__":
    for e in sys.argv[1:]:
        globals()[e]()
