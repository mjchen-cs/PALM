#!/usr/bin/env python3
"""
Local-computing energy model calibration on Jetson Orin Nano.

(a) GPU model  E_gpu = (P0/R + zeta) * D   (D = input data, kbit)
    Runs a ResNet-18 inference workload on 224x224x3 uint8 images on the GPU
    (and on the CPU for comparison) and measures
        R    : processed input data rate (Mbps)
        P0   : rail power above idle while the GPU is busy minus dynamic part
        zeta : dynamic energy per kbit (mJ/kbit)
(b) CPU DVFS model  P_cpu(f) = kappa * f^3 (+ static)
    If run as root, sweeps scaling_max_freq of one core running a fixed
    integer workload and fits kappa (W/GHz^3) by least squares.

Output: results/local_compute.json  (and prints the simulator parameters
        R_gpu, P_gpu0, zeta_gpu, kappa for the 'jetson' device class)

Dependencies: PyTorch + torchvision for Jetson (NVIDIA wheels), numpy.
"""
import argparse, json, os, time, glob
import numpy as np
from bench_pqc_energy import INA3221, PowerSampler, idle_power, OUTDIR


def gpu_model(ina, rail, seconds=8.0, batch=8):
    import torch, torchvision
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = torchvision.models.resnet18(weights=None).eval().to(dev)
    if dev == "cuda":
        m = m.half()
    x8 = torch.randint(0, 255, (batch, 3, 224, 224), dtype=torch.uint8)
    bits_per_batch = x8.numel() * 8
    def step():
        x = x8.to(dev, non_blocking=True)
        x = x.half() / 255 if dev == "cuda" else x.float() / 255
        with torch.no_grad():
            m(x)
        if dev == "cuda":
            torch.cuda.synchronize()
    for _ in range(10):
        step()
    ps = PowerSampler(ina); ps.start()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        step(); n += 1
    dt = time.perf_counter() - t0
    ps.stop()
    P = ps.mean(rail)
    R = n * bits_per_batch / dt / 1e6                       # Mbps of raw input data
    return dev, R, P


def cpu_dvfs(ina, rail, core=2, seconds=3.0):
    base = f"/sys/devices/system/cpu/cpu{core}/cpufreq"
    try:
        freqs = sorted(int(f) for f in open(f"{base}/scaling_available_frequencies").read().split())
    except OSError:
        print("cpufreq not accessible; skipping DVFS fit"); return None
    os.sched_setaffinity(0, {core})
    orig = open(f"{base}/scaling_max_freq").read().strip()
    orig_min = open(f"{base}/scaling_min_freq").read().strip()
    out = []
    try:
        # jetson_clocks pins min=max; release the floor so that max can be lowered
        open(f"{base}/scaling_min_freq", "w").write(str(freqs[0]))
        for f in freqs[:: max(1, len(freqs) // 6)]:
            open(f"{base}/scaling_max_freq", "w").write(str(f))
            time.sleep(0.5)
            P_id = idle_power(ina, 2.0)[rail]
            ps = PowerSampler(ina); ps.start()
            t0 = time.perf_counter(); x = 0
            while time.perf_counter() - t0 < seconds:
                for i in range(20000):
                    x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            ps.stop()
            out.append((f / 1e6, ps.mean(rail) - P_id))     # GHz, W dynamic
    finally:
        open(f"{base}/scaling_max_freq", "w").write(orig)
        open(f"{base}/scaling_min_freq", "w").write(orig_min)
    f = np.array([o[0] for o in out]); P = np.array([o[1] for o in out])
    kappa = float(np.sum(P * f ** 3) / np.sum(f ** 6))
    return dict(points=out, kappa_W_per_GHz3=kappa)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rail", default=None)
    a = ap.parse_args()
    ina = INA3221()
    rail = a.rail or next((r for r in ina.rails if "VDD_IN" in r), next(iter(ina.rails), None))
    P_idle = idle_power(ina)[rail]
    dev, R, P = gpu_model(ina, rail)
    P_dyn = P - P_idle
    # with E = (P0/R + zeta) D and a busy GPU: energy per kbit = P_dyn / R  (W / Mbps = mJ/kbit)
    e_kbit = P_dyn / R
    res = dict(rail=rail, P_idle=P_idle, device=dev, R_mbps=R, P_busy=P, e_per_kbit=e_kbit,
               note="simulator uses P_gpu0/R_gpu + zeta_gpu = e_per_kbit; split P_gpu0 = static share")
    res["cpu_dvfs"] = cpu_dvfs(ina, rail)
    json.dump(res, open(os.path.join(OUTDIR, "local_compute.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
