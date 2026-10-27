#!/usr/bin/env python3
"""
Per-frame decision latency of the online scheduler on the Jetson (edge AP side).
Runs the PALM framework (actor inference + candidate screening + SIC-order
refinement) and baselines for N = 10..50 WDs and records median / p99 latency
and the power drawn while scheduling.

Output: hardware/jetson/results/latency_profile_jetson_v2.json (default, configuration of
        the revision: 1 s sessions, PALM-PS) or latency_profile_jetson.json (--v1).

Dependencies: numpy, scipy, numba, torch (CPU is enough), the palm package
(run from the repository root:  python3 hardware/jetson/bench_scheduler.py).
"""
import os, sys, json, time
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from palm.params import default_params, with_
from palm.system import Env
from palm.agents import LyDROO, LyDDPGAtt, GENERATOR
from palm.solver import warmup

try:
    from bench_pqc_energy import INA3221, PowerSampler
except Exception:
    INA3221 = None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", action="store_true",
                    help="earlier configuration (10 s sessions, iterative prior, 3 ms refinement)")
    ap.add_argument("--schemes", default="PALM-PS,PALM-G-PS,PALM,PALM-G,LyDROO-SCA,LyDROO-TDMA")
    ap.add_argument("--N", default="10,20,30,40,50")
    ap.add_argument("--rt", action="store_true",
                    help="real-time configuration: bound-and-prune screening, screening tolerance 1e-4, "
                         "predictive refinement budget, garbage collection outside the decision")
    ap.add_argument("--out", default=None, help="output file name")
    ap.add_argument("--baselines", action="store_true",
                    help="recent baselines and the earlier greedy algorithm (LyDiff, CEA, LyDDPG-Att, LyDROO-Greedy, PQC-agnostic)")
    args = ap.parse_args()
    warmup()
    ina = INA3221() if INA3221 else None
    rail = next((r for r in (ina.rails if ina else {}) if "VDD_IN" in r), None)
    base = default_params()
    if not args.v1:
        # main configuration of the revision: 1 s sessions, PALM-PS, closed-form prior, 1 ms refinement
        base["rekey_frames"] = 20
        base["fast_prior"] = True
        base["ref_budget"] = 1e-3
    if args.baselines and args.schemes == ap.get_default("schemes"):
        args.schemes = "LyDiff,CEA,LyDDPG-Att,LyDROO-Greedy,PQC-agnostic"
    if args.rt and args.schemes == ap.get_default("schemes"):
        args.schemes = "PALM-PS-RT,PALM-RT"     # real-time configuration (schemes carry the options)
    out = {"platform": os.uname().nodename + " " + os.uname().machine, "platform_factor": 1.0,
           "config": ("v1" if args.v1 else "v2") + ("-rt" if args.rt else "")}
    Ns = [int(n) for n in args.N.split(",")]
    for sch in args.schemes.split(","):
        out[sch] = {}
        for N in Ns:
            if sch == "LyDROO-SCA" and N > 30:
                continue
            p = with_(base, N=N, classes=(["iot"] * 6 + ["jetson"] * 4) * (N // 10), lam=2.5 * 10 / N)
            env = Env(p, seed=1)
            ag = LyDDPGAtt(env, seed=1) if sch == "LyDDPG-Att" else \
                LyDROO(env, sch, seed=1, generator=GENERATOR.get(sch, "dnn"))
            nf, warm = (120, 60) if sch in ("LyDROO-SCA", "CEA") else (400, 200)
            dts = []
            ps = PowerSampler(ina) if (ina and rail) else None
            for t in range(nf):
                if t == warm and ps:
                    ps.start()
                info = ag.step()
                if t >= warm:
                    dts.append(info["dt"])
            pw = None
            if ps:
                ps.stop(); pw = ps.mean(rail)
            out[sch][str(N)] = float(np.median(dts) * 1e3)
            out[sch][f"{N}_p99"] = float(np.percentile(dts, 99) * 1e3)
            out[sch][f"{N}_power_W"] = pw
            print(sch, N, "median %.3f ms  p99 %.3f ms" % (out[sch][str(N)], out[sch][f"{N}_p99"]), flush=True)
    od = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(od, exist_ok=True)
    name = args.out or ("latency_profile_jetson.json" if args.v1 else
                        ("latency_profile_jetson_rt.json" if args.rt else
         ("latency_profile_jetson_base.json" if args.baselines else "latency_profile_jetson_v2.json")))
    json.dump(out, open(os.path.join(od, name), "w"), indent=1)
    print("saved", os.path.join(od, name))


if __name__ == "__main__":
    main()
