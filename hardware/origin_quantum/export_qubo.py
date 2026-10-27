#!/usr/bin/env python3
"""
Export per-frame offloading QUBO instances (n = 6 or 8 qubits) from states
sampled in simulation (results/states.pkl) to a self-contained JSON file that
run_wukong.py can execute on the Origin Quantum superconducting QPU.

Each instance stores: Ising coefficients (hz, J), the QUBO surrogate, and the
*true* per-frame DPP value of all 2^n offloading vectors (exact critic), so
hardware samples can be scored without the simulator.
"""
import os, sys, json, pickle, itertools, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from palm.system import Env, build_coeffs, evaluate
from palm import qaoa as QA
from palm.solver import warmup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "instances.json"))
    a = ap.parse_args()
    warmup()
    d = pickle.load(open(os.path.join(ROOT, "results", "states.pkl"), "rb"))
    p = d["p"]; dev = Env(p).dev
    allx = np.array(list(itertools.product([0, 1], repeat=a.n)), np.int8)
    inst = []
    for r in d["rec"][: a.count]:
        s = r["state"]
        C = build_coeffs(p, dev, s["h_hat"], s["Q"], s["Y"], s["Y0"], s["need_hs"])
        idx = np.arange(a.n)
        full = lambda xs: np.r_[xs, np.zeros(p["N"] - a.n, np.int8)]
        vals = [float(evaluate(p, dev, C, full(x))) for x in allx]
        Q = QA.build_qubo(p, dev, C, idx)
        hz, J, c0 = QA.ising(Q)
        g, b, _ = QA.train_qaoa(hz, J, 1, restarts=8)
        g2, b2, _ = QA.train_qaoa(hz, J, 2, restarts=8)
        # key: bit i of the integer = qubit i = WD i
        inst.append(dict(n=a.n, hz=hz.tolist(), J=J.tolist(), const=c0,
                         true_vals={str(int(sum(int(v) << i for i, v in enumerate(x)))): vals[k] for k, x in enumerate(allx)},
                         f0=float(Q["f0"]), qaoa_p1=dict(gamma=list(g), beta=list(b)),
                         qaoa_p2=dict(gamma=list(g2), beta=list(b2))))
    json.dump(inst, open(a.out, "w"))
    print("exported", len(inst), "instances to", a.out)


if __name__ == "__main__":
    main()
