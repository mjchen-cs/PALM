#!/usr/bin/env python3
"""
Run QAOA offloading-candidate generation on the Origin Quantum (本源量子)
superconducting QPU through pyqpanda3's QCloud interface, or on the local
noisy simulator for a dry run.

  pip install pyqpanda3            (tested with 0.5.0)
  export ORIGIN_QCLOUD_API_KEY=...  (from the Origin Quantum cloud console)

  # dry run on local simulator with depolarising noise
  python run_wukong.py --backend local --depol 0.005
  # real chip (check the exact backend name with --list)
  python run_wukong.py --list
  python run_wukong.py --backend origin_wukong --shots 1000 --layers 1

For each instance the script reports
  * P(opt): probability of sampling the true optimum,
  * best-of-K: normalised DPP gain of the best of the K most frequent bitstrings
    re-scored by the exact critic ( (F(x)-F(0)) / (F*-F(0)) ),
  * the same metric for K uniformly random bitstrings (baseline),
  * queue/execution time reported by the cloud.
Results: results_<backend>.json
"""
import os, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def build_prog(inst, layers, n_logical, phys=None, decompose=False):
    from pyqpanda3.core import QProg, H, RZ, RZZ, RX, CNOT, measure
    hz, J = np.array(inst["hz"]), np.array(inst["J"])
    par = inst[f"qaoa_p{layers}"]
    q = phys or list(range(n_logical))
    prog = QProg()
    for i in range(n_logical):
        prog << H(q[i])
    for g, b in zip(par["gamma"], par["beta"]):
        for i in range(n_logical):
            for j in range(i + 1, n_logical):
                if abs(J[i, j]) > 1e-12:
                    if decompose:   # RZZ(t) = CNOT . RZ_target(t) . CNOT
                        prog << CNOT(q[i], q[j]) << RZ(q[j], 2 * g * J[i, j]) << CNOT(q[i], q[j])
                    else:
                        prog << RZZ(q[i], q[j], 2 * g * J[i, j])
        for i in range(n_logical):
            prog << RZ(q[i], 2 * g * hz[i])
        for i in range(n_logical):
            prog << RX(q[i], 2 * b)
    prog << measure(q, list(range(n_logical)))
    return prog


def counts_to_ints(counts, n):
    out = {}
    for k, v in counts.items():
        k = str(k).strip()
        if k.lower().startswith("0x"):
            ival = int(k, 16) & ((1 << n) - 1)
        else:
            ival = int(k[-n:].rjust(n, "0"), 2)   # right-most char = c-bit 0 = qubit 0 = WD 0
        out[ival] = out.get(ival, 0) + v
    return out


def score(inst, ints_counts, K, rng):
    tv = {int(k): v for k, v in inst["true_vals"].items()}
    f0, opt = inst["f0"], max(tv.values())
    kopt = max(tv, key=tv.get)
    tot = sum(ints_counts.values())
    top = sorted(ints_counts, key=lambda k: -ints_counts[k])[:K]
    best = max(tv[k] for k in top)
    rnd = max(tv[int(k)] for k in rng.integers(0, 2 ** inst["n"], K))
    norm = lambda v: (v - f0) / (opt - f0) if opt > f0 else 1.0
    return dict(p_opt=ints_counts.get(kopt, 0) / tot, best_of_K=norm(best), random_K=norm(rnd))


def _state(job):
    try:
        return str(job.status()).upper()
    except Exception as e:
        return f"UNKNOWN({e})"


def wait_result(job, poll=15, timeout=6 * 3600):
    """Poll the cloud until the job is finished or failed, then fetch the result."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = _state(job)
        if "FINISH" in st or "COMPLETE" in st or "SUCCESS" in st:
            break
        if "FAIL" in st or "ERROR" in st or "CANCEL" in st:
            print("job state:", st, flush=True)
            break
        print(f"  [{time.strftime('%H:%M:%S')}] state {st}, waiting...", flush=True)
        time.sleep(poll)
    return job.result()


def _dump(out):
    info = {}
    for name in ("job_status", "error_message", "get_counts_list", "get_probs_list"):
        try:
            info[name] = str(getattr(out, name)())[:500]
        except Exception as e:
            info[name] = f"<{type(e).__name__}: {e}>"
    return info


def extract_counts(out, idx, shots):
    """Counts of circuit idx; falls back to probabilities (read-out-corrected results)."""
    for getter, is_prob in (("get_counts_list", False), ("get_probs_list", True)):
        try:
            lst = getattr(out, getter)()
        except Exception:
            continue
        if lst and len(lst) > idx and lst[idx]:
            d = dict(lst[idx])
            return {k: v * shots for k, v in d.items()} if is_prob else d
    if idx == 0:
        for getter, is_prob in (("get_counts", False), ("get_probs", True)):
            try:
                d = dict(getattr(out, getter)())
            except Exception:
                continue
            if d:
                return {k: v * shots for k, v in d.items()} if is_prob else d
    raise RuntimeError("empty result from cloud: " + json.dumps(_dump(out), ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--instances", default=os.path.join(HERE, "instances.json"))
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=1, choices=[1, 2])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--depol", type=float, default=0.0)
    ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--decompose", action="store_true", help="use CNOT-RZ-CNOT instead of RZZ")
    ap.add_argument("--batch", action="store_true", help="submit all instances in one cloud job")
    ap.add_argument("--job", default=None,
                    help="fetch existing job id(s) instead of submitting; comma-separated, one per instance "
                         "(or one id for a --batch job)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    insts = json.load(open(a.instances))[: a.max]
    res = []
    if a.backend == "local":
        from pyqpanda3.core import CPUQVM, NoiseModel, depolarizing_error, GateType
        for inst in insts:
            prog = build_prog(inst, a.layers, inst["n"], decompose=a.decompose)
            qvm = CPUQVM()
            if a.depol > 0:
                nm = NoiseModel()
                nm.add_all_qubit_quantum_error(depolarizing_error(a.depol), [GateType.RX, GateType.RZ, GateType.H])
                nm.add_all_qubit_quantum_error(depolarizing_error(10 * a.depol), [GateType.RZZ])
                qvm.run(prog, a.shots, nm)
            else:
                qvm.run(prog, a.shots)
            r = score(inst, counts_to_ints(qvm.result().get_counts(), inst["n"]), a.K, rng)
            res.append(r); print(r)
    else:
        from pyqpanda3.qcloud import QCloudService, QCloudOptions, QCloudJob
        key = os.environ.get("ORIGIN_QCLOUD_API_KEY")
        assert key, "set ORIGIN_QCLOUD_API_KEY"
        svc = QCloudService(key)
        if a.list:
            print(svc.backends()); return
        be = svc.backend(a.backend)
        opt = QCloudOptions()
        opt.set_mapping(True)          # let the compiler map to the coupling graph
        opt.set_optimization(True)
        opt.set_amend(True)            # read-out error mitigation
        if a.batch:
            progs = [build_prog(inst, a.layers, inst["n"], decompose=a.decompose) for inst in insts]
            t0 = time.time()
            if a.job:
                job = QCloudJob(a.job.split(",")[0].strip())
            else:
                job = be.run(progs, a.shots, opt)
            print("batch job", job.job_id(), flush=True)
            out = wait_result(job)
            wall = time.time() - t0
            for i, inst in enumerate(insts):
                counts = extract_counts(out, i, a.shots)
                r = score(inst, counts_to_ints(counts, inst["n"]), a.K, rng)
                r["job_id"] = job.job_id()
                r["wall_s_batch"] = wall
                res.append(r); print(r, flush=True)
        else:
            ids = [s.strip() for s in a.job.split(",")] if a.job else None
            if ids:
                insts = insts[: len(ids)]
            for k, inst in enumerate(insts):
                t0 = time.time()
                if ids:
                    job = QCloudJob(ids[k])
                else:
                    prog = build_prog(inst, a.layers, inst["n"], decompose=a.decompose)
                    job = be.run([prog], a.shots, opt)
                print("job", job.job_id(), "instance", k, flush=True)
                out = wait_result(job)
                counts = extract_counts(out, 0, a.shots)
                r = score(inst, counts_to_ints(counts, inst["n"]), a.K, rng)
                r["job_id"] = job.job_id()
                r["wall_s"] = time.time() - t0
                try:
                    r["timing"] = str(out.timing_info())
                except Exception:
                    pass
                res.append(r); print(r, flush=True)
    summ = {k: float(np.mean([r[k] for r in res])) for k in ("p_opt", "best_of_K", "random_K")}
    print("SUMMARY", a.backend, "layers", a.layers, summ)
    json.dump(dict(args=vars(a), summary=summ, rows=res),
              open(os.path.join(HERE, f"results_{a.backend}_p{a.layers}{'_dec' if a.decompose else ''}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
