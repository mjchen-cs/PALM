#!/usr/bin/env python3
"""
Dump the calibration data of an Origin Quantum backend (T1/T2, single-qubit gate
fidelity, read-out fidelity, two-qubit gate fidelity) to chip_info_<backend>.json
and print chip-wide medians for the paper.

    $env:ORIGIN_QCLOUD_API_KEY = "..."
    python chip_info.py --backend WK_C180

pyqpanda3 has changed the accessor names between versions, so every object is
read through introspection: all zero-argument getters are called and stored.
"""
import os, json, argparse, statistics, time


def obj_to_dict(o, depth=0):
    """Call every public zero-argument method/attribute of a pybind object."""
    if depth > 3:
        return str(o)
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    if isinstance(o, dict):
        return {str(k): obj_to_dict(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [obj_to_dict(v, depth + 1) for v in o]
    out = {}
    for name in dir(o):
        if name.startswith("_"):
            continue
        try:
            a = getattr(o, name)
            v = a() if callable(a) else a
        except Exception:
            continue
        out[name] = obj_to_dict(v, depth + 1)
    return out if out else str(o)


def find_numbers(d, keys):
    """Collect all numeric values whose key contains one of `keys` (case-insensitive)."""
    vals = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and any(s in k.lower() for s in keys):
                    vals.append(float(v))
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(d)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="WK_C180")
    a = ap.parse_args()
    from pyqpanda3.qcloud import QCloudService
    key = os.environ.get("ORIGIN_QCLOUD_API_KEY")
    assert key, "set ORIGIN_QCLOUD_API_KEY"
    be = QCloudService(key).backend(a.backend)
    info = None
    for name in ("chip_info", "get_chip_info", "chip_config"):
        if hasattr(be, name):
            try:
                info = getattr(be, name)()
                break
            except Exception as e:
                print(name, "failed:", e)
    if info is None:
        print("No chip-info accessor found. Available methods:", [n for n in dir(be) if not n.startswith("_")])
        return
    d = obj_to_dict(info)
    d["_queried_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"chip_info_{a.backend}.json")
    json.dump(d, open(path, "w"), indent=1, ensure_ascii=False)
    print("saved", path)
    print("top-level fields:", list(d.keys()))
    for lab, ks in [("T1", ["t1"]), ("T2", ["t2"]),
                    ("single-qubit gate fidelity", ["single_gate_fidelity", "gate_fidelity"]),
                    ("read-out fidelity", ["readout"]),
                    ("two-qubit / CZ fidelity", ["double", "cz", "two"])]:
        v = [x for x in find_numbers(d, ks) if x > 0]
        if v:
            print(f"{lab:28s} n={len(v):4d}  median={statistics.median(v):.4g}  min={min(v):.4g}  max={max(v):.4g}")


if __name__ == "__main__":
    main()
