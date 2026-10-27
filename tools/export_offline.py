"""Export the offline experiments (E2, E9, E10, sampled states S) to CSV/NPZ/JSON in data/offline.
Usage: python3 tools/export_offline.py"""
import os, sys, csv, json, pickle
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results"); O = os.path.join(ROOT, "data", "offline"); os.makedirs(O, exist_ok=True)

def tolist(x):
    if isinstance(x, dict): return {str(k): tolist(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [tolist(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)): return x.item()
    return x

def table(rows, fn):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(os.path.join(O, fn), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(tolist(v)) if isinstance(v, (list, tuple, dict, np.ndarray)) else tolist(v)) for k, v in r.items()})

# E2: power allocation and SIC order, 400 instances (values of every method and the best over all M! orders)
table(pickle.load(open(os.path.join(R, "E2.pkl"), "rb")), "E2_sic_order_gap.csv")
# E9: binary vs. partial offloading, per-frame DPP value
table(pickle.load(open(os.path.join(R, "E9.pkl"), "rb")), "E9_partial_offloading.csv")
# E10: QAOA candidates in simulation
json.dump(tolist(pickle.load(open(os.path.join(R, "E10.pkl"), "rb"))), open(os.path.join(O, "E10_qaoa_simulation.json"), "w"))
# S: 200 sampled states with all 2^10 offloading values, PALM's choice, greedy and SCA choices
d = pickle.load(open(os.path.join(R, "states.pkl"), "rb"))
arr = {}
rows = []
for i, r in enumerate(d["rec"]):
    for k, v in r["state"].items():
        arr[f"s{i:03d}/state_{k}"] = np.asarray(v)
    arr[f"s{i:03d}/allvals"] = np.asarray(r["allvals"])
    rows.append({k: r[k] for k in r if k not in ("state", "allvals")})
np.savez_compressed(os.path.join(O, "S_states_allvals.npz"), **arr)
table(rows, "S_states_decisions.csv")
json.dump(tolist({k: v for k, v in d["p"].items() if not callable(v)}), open(os.path.join(O, "S_params.json"), "w"), indent=1)
print("offline data exported to", O)
