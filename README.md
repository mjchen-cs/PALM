# PALM / PALM-PS: PQC-Aware Online Offloading and NOMA Resource Allocation

Code and **all data** for the paper

> Y. Yao, W. Xiao, M. Chen, A. Liu, Z. Liu, M. Chen, A. Farouk, and H. H. Song,
> "PQC-Aware Online Offloading and NOMA Resource Allocation for Quantum-Safe Mobile Edge Computing,"
> submitted to *IEEE Transactions on Mobile Computing*.

PALM is a Lyapunov-guided, learning-assisted online controller for NOMA-based mobile edge computing
whose uplink is protected by standardized post-quantum cryptography (ML-KEM/ML-DSA handshakes and
AES-256-GCM, NIST categories 1/3/5). PALM-PS adds proactive session management: devices may establish
PQC sessions in frames without offloading, also over two frames, when a renewal argument shows that
this pays off.

## Repository layout
```
palm/            core library
  params.py        system, device-class, and PQC parameters
  solver.py        Algorithm 1 (exact chain interior-point solver), Algorithm 2 (SIC-order refinement),
                   greedy and SCA baselines, interference-free upper bound
  system.py        PQC-aware coefficients (Lemma 1), critic, TDMA, environment (queues, sessions,
                   imperfect SIC, CSI error), proactive sessions (Eq. (23), Proposition 3),
                   robust rate selection
  agents.py        PALM / PALM-PS / PALM-G / PALM-PS-RT / keep-alive, LyDROO-type baselines,
                   DDPG, LyDiff, CEA, LyDDPG-Att, run()
  qaoa.py          QUBO surrogate and QAOA candidate generation
  parallel.py      parallel runs with an on-disk cache
experiments/
  run_online.py    online experiments E1, E3, E5-E8, E11-E13
  run_offline.py   sampled states S, E2 (SIC-order gap), E4 (runtime), E9 (partial offloading), E10 (QAOA)
  plots.py         every figure and table of the paper -> figs/
  fig_testbed.py   hardware testbed figure
hardware/
  jetson/          PQC, local-computing, and decision-latency benchmarks for the NVIDIA Jetson Orin Nano
  origin_quantum/  QAOA on the Origin Wukong 180 superconducting processor (pyqpanda3)
  photos/          photos used in the testbed figure
tools/             export of all data to open formats (data/)
results/           results used by plots.py (Python pickles / JSON)
data/              ALL data of the paper in open formats (see data/README.md)
figs/              all figures of the paper (PDF)
reproduce_all.sh   reproduces every result from scratch
```

## Installation
```
python3 -m pip install -r requirements.txt
# optional: python3 -m pip install liboqs-python pyqpanda3
```
Tested with Python 3.13, NumPy 2, Numba 0.61, PyTorch 2.5. Units inside the code: time in ms,
data in kbit, rate in Mbps (= kbit/ms), power in W, energy in mJ.

## Reproducing the paper
* Figures and tables from the stored results (minutes): `python3 experiments/plots.py`
* Everything from scratch (about 5-6 h on 2 cores): `./reproduce_all.sh`
* A single experiment, e.g., the main comparison:
  `FRAMES=10000 SEEDS=1,2,3 PROCS=2 python3 experiments/run_online.py E1`

| Experiment | Paper | Content |
|---|---|---|
| S, E2 | Sec. V-B, Table VI, Fig. 3 | gap to all SIC orders and to all offloading vectors |
| E9 | Sec. V-B | binary versus partial offloading |
| E4 | Sec. V-C, Fig. 4 | solver and decision times (incl. real-time configuration) |
| E1, E13 | Sec. V-C/D, Table VII, Figs. 5-6 | comparison of all schemes, 1 s sessions |
| E3 | Sec. V-E, Fig. 7 | load sweep and stability region |
| E6, E6b | Sec. V-F, Fig. 8 | security level, session lifetime, packet size, static power |
| E11, E12 | Sec. V-G, Table VIII, Fig. 8(d) | proactive sessions; all schemes with 0.25 s sessions |
| E7, E7b | Sec. V-H, Fig. 9 | imperfect SIC, CSI error (incl. robust rate selection), device mix, ES budget |
| E8 | Sec. V-I, Fig. 10 | V x nu grid |
| E5 | Sec. V-J, Fig. 11 | deadline-aware evaluation with measured Jetson latencies |
| H1-H5 | Sec. V-K/L, Tables IV-V, IX, Fig. 12 | Jetson and quantum-processor measurements |

## Main options
* **Proactive sessions (PALM-PS):** scheme `PALM-PS`; parameters `ps_zeta` (default 1), `ps_kmin` (1),
  `ps_ema` (0.05), `ps_airmin` (5 ms, two-frame handshakes). `KeepAlive` renews every session when possible.
* **Real-time configuration:** schemes `PALM-PS-RT` / `PALM-RT` (bound-based pruning, screening tolerance
  1e-4, predictive refinement budget, garbage collection outside the decision).
* **Robust rate selection under CSI error:** `robust_rate=True` with `rate_mode="fixed"` and `csi_err_std>0`.
* **SIC-aware variant:** `sic_aware=True` with `sic_residual>0` (PALM-PS-R).
* **Measured hardware profiles:** `PROFILE=hardware/jetson/results/profile_jetson_ap.json` before
  `run_online.py`; Jetson decision latencies: `python3 hardware/jetson/bench_scheduler.py [--rt|--baselines]`.

## Data
`data/` contains every number behind the paper: the frame-by-frame time series of all 541 online runs,
the per-run summaries, the offline experiments, the timing measurements, and the raw hardware
measurements (Jetson Orin Nano and Origin Wukong 180). See `data/README.md` for the formats.

## License
The license file (`LICENSE`) is chosen by the authors before publication.
