#!/usr/bin/env python3
"""
PQC latency + energy benchmark on NVIDIA Jetson Orin Nano (JetPack 5/6).

Measures, for NIST security categories 1/3/5:
  ML-KEM-{512,768,1024}: keygen / encaps / decaps
  ML-DSA-{44,65,87}    : keygen / sign / verify
  AES-256-GCM          : encryption throughput (payload protection)
and integrates the on-board INA3221 power rails (VDD_IN = whole module,
VDD_CPU_GPU_CV) during each benchmark loop.  Energy per operation is reported
both as total energy (P_avg * t) and as *dynamic* energy ((P_avg - P_idle) * t).

Output:  results/pqc_jetson_<mode>.json   and a profile file
         results/profile_jetson.json  that the simulator consumes:
         python experiments/run_online.py ... with PROFILE=hardware/jetson/results/profile_jetson.json

Usage (run as root for stable clocks):
  sudo nvpmodel -m 0 && sudo jetson_clocks          # fixed max clocks ("ap" mode)
  taskset -c 2 python3 bench_pqc_energy.py --mode ap
  # emulate a low-end WD: limit CPU frequency (e.g. 729.6 MHz) and one core
  sudo sh -c 'echo 729600 > /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq'
  taskset -c 2 python3 bench_pqc_energy.py --mode lowfreq

Dependencies: pip3 install liboqs-python cryptography numpy
(liboqs-python builds liboqs on first import; needs cmake, gcc, git.)
"""
import argparse, glob, json, os, threading, time, statistics
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results")
os.makedirs(OUTDIR, exist_ok=True)

LEVELS = {1: ("ML-KEM-512", "ML-DSA-44"), 3: ("ML-KEM-768", "ML-DSA-65"), 5: ("ML-KEM-1024", "ML-DSA-87")}


# ---------------------------------------------------------------- power rails
class INA3221:
    """Reads Jetson INA3221 rails via hwmon sysfs (auto-discovery)."""

    def __init__(self):
        self.rails = {}
        cands = glob.glob("/sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*") + \
                glob.glob("/sys/bus/i2c/drivers/ina3221x/*/iio:device*")
        for d in cands:
            for lab in glob.glob(os.path.join(d, "in*_label")) + glob.glob(os.path.join(d, "rail_name_*")):
                try:
                    name = open(lab).read().strip()
                except OSError:
                    continue
                if "in" in os.path.basename(lab) and lab.endswith("_label"):
                    ch = os.path.basename(lab)[2:-6]                 # "1"
                    v = os.path.join(d, f"in{ch}_input")             # mV
                    c = os.path.join(d, f"curr{ch}_input")           # mA
                    if os.path.exists(v) and os.path.exists(c):
                        self.rails[name] = ("vi", v, c)
                else:   # older iio layout
                    ch = lab.rsplit("_", 1)[-1]
                    pw = os.path.join(d, f"in_power{ch}_input")      # mW
                    if os.path.exists(pw):
                        self.rails[name] = ("p", pw, None)
        if not self.rails:
            print("WARNING: no INA3221 rails found -> energy will be NaN (use tegrastats instead)")

    def read(self):
        out = {}
        for n, (kind, a, b) in self.rails.items():
            try:
                if kind == "vi":
                    out[n] = float(open(a).read()) * float(open(b).read()) / 1e6   # W
                else:
                    out[n] = float(open(a).read()) / 1e3
            except OSError:
                pass
        return out


class PowerSampler(threading.Thread):
    def __init__(self, ina, period=0.005):
        super().__init__(daemon=True)
        self.ina, self.period, self.samples, self.run_flag = ina, period, [], True

    def run(self):
        while self.run_flag:
            self.samples.append((time.perf_counter(), self.ina.read()))
            time.sleep(self.period)

    def stop(self):
        self.run_flag = False
        self.join()

    def mean(self, rail):
        v = [s[1].get(rail, np.nan) for s in self.samples]
        return float(np.nanmean(v)) if v else float("nan")


def measure(fn, ina, seconds=3.0, rail=None):
    """Run fn repeatedly for `seconds`, return (median op time [s], mean power [W] per rail)."""
    for _ in range(20):
        fn()
    ps = PowerSampler(ina); ps.start()
    ts = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ps.stop()
    pw = {r: ps.mean(r) for r in ina.rails}
    return statistics.median(ts), float(np.mean(ts)), len(ts), pw


def idle_power(ina, seconds=5.0):
    ps = PowerSampler(ina); ps.start(); time.sleep(seconds); ps.stop()
    return {r: ps.mean(r) for r in ina.rails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ap", help="label of the CPU configuration (ap / lowfreq / ...)")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--rail", default=None, help="rail used for energy (default: VDD_IN if present)")
    args = ap.parse_args()

    import oqs
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    ina = INA3221()
    rail = args.rail or next((r for r in ina.rails if "VDD_IN" in r), next(iter(ina.rails), None))
    print("power rails:", list(ina.rails), "-> using", rail)
    P_idle = idle_power(ina)
    print("idle power:", P_idle)

    res = dict(mode=args.mode, rail=rail, idle=P_idle, liboqs=oqs.oqs_version(), levels={})
    for lev, (kem, sig) in LEVELS.items():
        r = {}
        with oqs.KeyEncapsulation(kem) as srv, oqs.KeyEncapsulation(kem) as cli:
            pk = srv.generate_keypair()
            ct, _ = cli.encap_secret(pk)
            r["kem_keygen"] = measure(lambda: oqs.KeyEncapsulation(kem).generate_keypair(), ina, args.seconds)
            r["encaps"] = measure(lambda: cli.encap_secret(pk), ina, args.seconds)
            r["decaps"] = measure(lambda: srv.decap_secret(ct), ina, args.seconds)
            r["sizes_kem"] = dict(pk=len(pk), ct=len(ct))
        with oqs.Signature(sig) as s:
            spk = s.generate_keypair(); m = os.urandom(64); sg = s.sign(m)
            r["sign"] = measure(lambda: s.sign(m), ina, args.seconds)
            r["verify"] = measure(lambda: s.verify(m, sg, spk), ina, args.seconds)
            r["sizes_sig"] = dict(pk=len(spk), sig=len(sg))
        res["levels"][lev] = r
        print(kem, sig, {k: round(v[0] * 1e6, 1) for k, v in r.items() if isinstance(v, tuple)}, "us")

    # AES-256-GCM per-kbit cost (1500-byte packets)
    key = AESGCM.generate_key(256); aes = AESGCM(key); nonce = os.urandom(12); pkt = os.urandom(1500)
    t_med, t_mean, n, pw = measure(lambda: aes.encrypt(nonce, pkt, None), ina, args.seconds)
    res["aes_gcm_1500B"] = (t_med, t_mean, n, pw)

    # ---- derive simulator profile: WD-side handshake = encaps + sign + verify
    prof = {}
    for lev, r in res["levels"].items():
        t_hs = (r["encaps"][0] + r["sign"][0] + r["verify"][0]) * 1e3            # ms
        e_tot = sum(r[k][0] * r[k][3].get(rail, np.nan) for k in ("encaps", "sign", "verify")) * 1e3   # mJ
        e_dyn = sum(r[k][0] * (r[k][3].get(rail, np.nan) - P_idle.get(rail, 0)) for k in ("encaps", "sign", "verify")) * 1e3
        prof[str(lev)] = dict(t_hs=t_hs, E_hs=e_dyn, E_hs_total=e_tot)
    aes_kbit_s = res["aes_gcm_1500B"][0] / 12.0                                 # 1500 B = 12 kbit
    eps = aes_kbit_s * (res["aes_gcm_1500B"][3].get(rail, np.nan) - P_idle.get(rail, 0)) * 1e3   # mJ/kbit
    for v in prof.values():
        v["eps_aead"] = eps
    res["profile"] = prof
    fn = os.path.join(OUTDIR, f"pqc_jetson_{args.mode}.json")
    json.dump(res, open(fn, "w"), indent=1, default=float)
    json.dump({"jetson": prof}, open(os.path.join(OUTDIR, f"profile_jetson_{args.mode}.json"), "w"), indent=1)
    print("saved", fn)
    print("profile:", json.dumps(prof, indent=1))


if __name__ == "__main__":
    main()
