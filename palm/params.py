"""
System / PQC parameters.

Unit convention (chosen so that rates in Mbps equal kbit per ms):
    time  : ms          data  : kbit        rate : Mbps (= kbit/ms)
    power : W           energy: mJ (= W * ms)
    CPU frequency: GHz  (cycles per ms = f * 1e6)

PQC profiles
------------
Each NIST security category s in {1,3,5} pairs ML-KEM (FIPS 203) for key
establishment with ML-DSA (FIPS 204) for authentication; bulk payload is
protected by AES-256-GCM.  A secure session is (re)established by a
handshake: the WD encapsulates to the ES's ML-KEM key, signs the transcript
with ML-DSA and verifies the ES signature.  Object sizes are the standard
FIPS 203/204 sizes.  Timing / energy per device class:

  * ES (x86 server)       : measured with liboqs 0.16 on Intel Xeon @2.1GHz
                            (scripts/bench_liboqs.py).
  * class 'iot'           : PQC on a Cortex-M4 class secure co-processor
                            (168 MHz); cycle counts are representative of the
                            pqm4 benchmark ranges -> REPLACE with your own
                            measurement if you have one.
  * class 'jetson'        : PQC on the Jetson Orin Nano Cortex-A78AE CPU.
                            Default numbers are PLACEHOLDERS (x86 * 4.0) and
                            are overwritten by hardware/jetson/results/*.json
                            produced by hardware/jetson/bench_pqc_energy.py.
"""
import json
import os
import copy
import numpy as np

# ---------------------------------------------------------------- PQC sizes
PQC_SIZES = {   # bytes: (ML-KEM ciphertext, ML-DSA signature)
    1: dict(kem="ML-KEM-512", sig="ML-DSA-44", ct=768, sigb=2420),
    3: dict(kem="ML-KEM-768", sig="ML-DSA-65", ct=1088, sigb=3309),
    5: dict(kem="ML-KEM-1024", sig="ML-DSA-87", ct=1568, sigb=4627),
}
HS_EXTRA_BYTES = 64          # nonces, ids, transcript hash
AEAD_OVH_BYTES = 28          # 12 B nonce + 16 B GCM tag per packet

# liboqs 0.16 on Xeon @2.1 GHz (measured in this work, median of 300 runs, us)
ES_X86_US = {
    1: dict(encaps=9.8, decaps=9.9, sign=59.8, verify=21.8),
    3: dict(encaps=13.2, decaps=14.3, sign=89.8, verify=34.4),
    5: dict(encaps=16.8, decaps=18.6, sign=110.2, verify=49.4),
}
# representative Cortex-M4 cycle counts (Mcycles), pqm4-style -> verify
IOT_M4_MCYC = {
    1: dict(encaps=0.55, sign=2.3, verify=0.9),
    3: dict(encaps=0.85, sign=3.7, verify=1.5),
    5: dict(encaps=1.30, sign=5.0, verify=2.5),
}

DEVICE_CLASSES = {
    # low-end IoT node, CPU-only, PQC on a secure co-processor
    "iot": dict(fmax=0.3, kappa=10.0, phi=1000.0, Pmax=0.1, gamma=0.08,
                gpu=False,
                pqc_mcu_mhz=168.0, pqc_mcu_power=0.10,     # W while busy
                P_static=0.002,                            # W while a session is alive
                aead_cyc_per_byte=30.0),
    # Jetson Orin Nano-class device with an integrated GPU.
    # Calibrated on a Jetson Orin Nano (JetPack 6.2, 15 W mode, jetson_clocks), see
    # hardware/jetson/results: ResNet-18 needs 334.8 CPU cycles/bit (1 A78AE core,
    # 1.498 GHz, 0.792 W dynamic) and runs at 459.5 Mbps / 13.39 nJ/bit on the GPU.
    # Scaled to the simulated workload (phi = 1000 cycles/bit, factor 1000/334.8):
    #   R_gpu = 459.5/2.987 = 153.8 Mbps,  rho = 13.39*2.987 = 40.0 nJ/bit,
    #   CPU: kappa*1.498^3 = 0.792 W -> kappa = 0.236 W/GHz^3.
    "jetson": dict(fmax=1.5, kappa=0.236, phi=1000.0, Pmax=0.2, gamma=0.25,
                   gpu=True, R_gpu=153.8, P_gpu0=0.0, zeta_gpu=0.0400,   # Mbps, W, mJ/kbit
                   pqc_cpu_power=1.5, cpu_slowdown_vs_x86=4.0,          # only used if no measurement
                   P_static=0.0,
                   aead_mJ_per_kbit=3.8e-5),    # OpenSSL AES-256-GCM: 1.2525 GB/s at 0.381 W
}

# Jetson Orin Nano measurements (liboqs 0.16, 1 core, median of 4 runs).
# WD handshake = Encaps + Sign + Verify; E_hs is dynamic energy above idle (VDD_IN).
JETSON_MEASURED = {
    1: dict(t_hs=0.371, E_hs=0.184),
    3: dict(t_hs=0.582, E_hs=0.294),
    5: dict(t_hs=0.856, E_hs=0.446),
}


def default_params():
    p = dict(
        N=10, T=50.0, W=2.0, v_u=1.1,
        N0_dBm_per_Hz=-174.0,
        # path loss (free-space-type, as in LyDROO): h = Ad (c/(4 pi fc d))^de * |g|^2
        Ad=4.11, fc=915e6, de=3.0, d_min=120.0, d_max=255.0, rician_K_dB=3.0,
        lam=2.0,                      # Mbps mean arrival per WD
        c_weights="odd1.5",
        classes=["iot"] * 6 + ["jetson"] * 4,
        sec_level=3,
        pkt_bytes=1500,               # AEAD packet size -> eta
        rekey_frames=200,             # session lifetime (frames)
        idle_timeout=40,              # session torn down after idle frames
        amortize=None,                # None -> 1/rekey_frames (handshake cost amortised over session); 1 = myopic
        # Lyapunov
        V=50.0, nu=20.0, nu0=2.0,
        # edge server
        es_eps_mJ_per_kbit=0.5,       # 500 nJ/bit: GPU-accelerated ES running the offloaded DNN workload
        es_gamma=10.0,                # W average budget of ES compute
        es_power_pqc=10.0,            # W while doing handshake crypto
        # non-idealities
        sic_residual=0.0,             # xi
        csi_err_std=0.0,              # relative std of |h| estimate
        rate_mode="adaptive",         # 'adaptive' or 'fixed' (outage)
        backoff=0.0,                  # planning back-off on h estimate
        tau_dec=0.0,                  # decision latency (ms) eating airtime
        seed=0,
    )
    return p


def noise_power(p):
    return 10 ** ((p["N0_dBm_per_Hz"] - 30) / 10) * p["W"] * 1e6   # W


def pqc_costs(p, cls_name, profile_override=None):
    """Return dict with per-handshake device time (ms), energy (mJ),
    handshake uplink bits (kbit), AEAD per-kbit energy (mJ/kbit), static power (W)."""
    s = p["sec_level"]
    sz = PQC_SIZES[s]
    bits_hs = (sz["ct"] + sz["sigb"] + HS_EXTRA_BYTES) * 8 / 1000.0
    c = DEVICE_CLASSES[cls_name]
    if cls_name == "iot":
        mc = IOT_M4_MCYC[s]
        cyc = (mc["encaps"] + mc["sign"] + mc["verify"]) * 1e6
        t_ms = cyc / (c["pqc_mcu_mhz"] * 1e3)
        e_mJ = t_ms * c["pqc_mcu_power"]
        aead = c["aead_cyc_per_byte"] * 125.0 / (c["pqc_mcu_mhz"] * 1e3) * c["pqc_mcu_power"]
    elif cls_name == "jetson" and s in JETSON_MEASURED:
        t_ms, e_mJ = JETSON_MEASURED[s]["t_hs"], JETSON_MEASURED[s]["E_hs"]
        aead = c["aead_mJ_per_kbit"]
    else:
        x = ES_X86_US[s]
        t_ms = (x["encaps"] + x["sign"] + x["verify"]) * c["cpu_slowdown_vs_x86"] / 1000.0
        e_mJ = t_ms * c["pqc_cpu_power"]
        aead = c["aead_mJ_per_kbit"]
    out = dict(t_hs=t_ms, E_hs=e_mJ, B_hs=bits_hs, eps_aead=aead, P_static=c["P_static"])
    if profile_override and cls_name in profile_override and str(s) in profile_override[cls_name]:
        out.update(profile_override[cls_name][str(s)])
    return out


def es_pqc_costs(p):
    x = ES_X86_US[p["sec_level"]]
    t_ms = (x["decaps"] + x["verify"] + x["sign"]) / 1000.0
    return dict(t_hs=t_ms, E_hs=t_ms * p["es_power_pqc"])


def eta_aead(p):
    return p["pkt_bytes"] / (p["pkt_bytes"] + AEAD_OVH_BYTES)


def load_profile(path):
    """Measured overrides, e.g. produced on the Jetson:
       {"jetson": {"3": {"t_hs": .., "E_hs": .., "eps_aead": ..}}}"""
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def with_(p, **kw):
    q = copy.deepcopy(p)
    q.update(kw)
    return q
