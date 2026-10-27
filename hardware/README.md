# Hardware Measurements (NVIDIA Jetson Orin Nano and Origin Quantum Wukong 180)

This folder contains the scripts that produced the hardware measurements of the paper and their
raw results. The measured values used in the paper are already stored in `jetson/results/`,
`origin_quantum/`, and `../data/hardware/`; rerunning the scripts is only needed to reproduce them
on your own hardware.

---

## 1. NVIDIA Jetson Orin Nano

The Jetson is used in two roles: (i) as a GPU-equipped wireless device (PQC handshake costs,
AES-GCM energy, local computing model) and (ii) as an edge controller (decision latency of the
scheduler).

### 1.0 Environment
Tested with JetPack 6.2 (L4T R36.4.3), liboqs 0.16.0, and PyTorch 2.5 (NVIDIA Jetson wheel).
```bash
sudo apt install -y cmake gcc git python3-pip
pip3 install numpy scipy numba liboqs-python cryptography
# install the NVIDIA PyTorch wheel that matches your JetPack version, then torchvision
git clone https://github.com/mjchen-cs/PALM.git && cd PALM
```
The first `import oqs` builds liboqs automatically (about 5-10 minutes).

### 1.1 Fix the power mode and clocks (for reproducible results)
```bash
sudo nvpmodel -q        # show the current power mode
sudo nvpmodel -m 0      # select the power mode (the paper uses the 15 W mode)
sudo jetson_clocks      # lock the clocks at their maximum
```
Report the power mode, the JetPack version, and the liboqs version together with the results.

### 1.2 PQC latency and energy (Jetson column of the PQC table)
```bash
cd hardware/jetson
# (a) application-processor mode: one core at full frequency ("Jetson-class WD")
sudo taskset -c 2 python3 bench_pqc_energy.py --mode ap
# (b) optional low-frequency mode: limit one core to a low frequency
sudo sh -c 'echo 729600 > /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq'
sudo taskset -c 2 python3 bench_pqc_energy.py --mode lowfreq
```
The script samples the `VDD_IN` power rail through the on-board INA3221 monitor (sysfs hwmon) and writes
- `results/pqc_jetson_ap.json`: median latency, average power, and idle power of every operation
  (ML-KEM key generation/encapsulation/decapsulation, ML-DSA signing/verification, AES-256-GCM);
- `results/profile_jetson_ap.json`: the parameters used by the simulator (handshake time `t_hs` in ms,
  dynamic handshake energy `E_hs` in mJ, AES-GCM energy per kbit `eps_aead`).

The paper reports four independent runs (`pqc_jetson_ap_run1.json` to `_run3.json` and
`pqc_jetson_ap.json`).

> Note: the INA3221 samples about every 5 ms, which is longer than a single operation. The script
> therefore runs each operation repeatedly for 3 s and estimates the energy as average power x
> single-operation latency. It reports both the total energy and the dynamic energy above the idle
> level; the paper uses the dynamic energy.

### 1.3 Local computing model (GPU and CPU DVFS)
```bash
sudo python3 bench_local_compute.py
```
Writes `results/local_compute.json`:
- the input-data throughput `R_mbps` and the energy per kbit `e_per_kbit` of ResNet-18 inference
  on the GPU (the paper's $R^G$ and $\varrho$);
- the CPU power at several frequencies and the fitted coefficient $\kappa$ (W/GHz^3).

These values enter `palm/params.py` (`DEVICE_CLASSES["jetson"]`: `R_gpu`, `P_gpu0`, `zeta_gpu`,
`kappa`, with `P_gpu0/R_gpu + zeta_gpu = e_per_kbit`).

### 1.4 Decision latency of the scheduler (edge controller)
Run from the repository root:
```bash
sudo jetson_clocks
python3 hardware/jetson/bench_scheduler.py               # PALM-PS, PALM-G-PS, PALM, PALM-G, SCA, TDMA
python3 hardware/jetson/bench_scheduler.py --rt          # real-time configuration: PALM-PS-RT, PALM-RT
python3 hardware/jetson/bench_scheduler.py --baselines --N 10,20   # LyDiff, CEA, LyDDPG-Att, earlier greedy, PQC-agnostic
```
The three runs write `latency_profile_jetson_v2.json`, `latency_profile_jetson_rt.json`, and
`latency_profile_jetson_base.json` to `hardware/jetson/results/`: the median and 99th-percentile
decision time per frame for N = 10-50 WDs and the module power during scheduling. Other options:
`--schemes` (comma-separated list), `--N` (comma-separated list), `--out` (file name), and `--v1`
(configuration of the earlier version).

### 1.5 Rerunning the simulations with the measured values
```bash
cp hardware/jetson/results/latency_profile_jetson_*.json results/
PROFILE=hardware/jetson/results/profile_jetson_ap.json FRAMES=8000 SEEDS=1,2 \
    python3 experiments/run_online.py E6
FRAMES=6000 SEEDS=1,2 python3 experiments/run_online.py E5   # deadline-aware experiment
python3 experiments/plots.py
```

---

## 2. Origin Quantum superconducting processor (QAOA candidate generation)

The quantum processor is used only as an optional, offline generator of offloading candidates.
It is unrelated to the PQC security mechanism, which runs on classical processors.

### 2.0 Account and environment
1. Register on the Origin Quantum cloud platform (qcloud.originqc.com.cn) and obtain an **API key**
   from the console; make sure that the account has machine time on a real processor
   (the paper uses `WK_C180`, Wukong 180).
2. `pip install pyqpanda3` (tested with 0.5.0).
3. Provide the key through an environment variable (never write it into the code):
   ```bash
   export ORIGIN_QCLOUD_API_KEY=<your key>            # Linux/macOS
   $env:ORIGIN_QCLOUD_API_KEY = "<your key>"           # Windows PowerShell
   ```

### 2.1 Generate the instances (on any computer; requires `results/states.pkl` from the simulation)
```bash
python3 hardware/origin_quantum/export_qubo.py --n 6 --count 20
```
Each instance contains the Ising coefficients, QAOA parameters trained on a simulator (p = 1 and
p = 2), and the exact DPP value of all 2^n offloading vectors (for offline scoring; the hardware
only samples). We recommend n = 6 (15 RZZ gates per layer); n = 8 (28 RZZ gates per layer) is
possible as a supplement.

### 2.2 Local noisy simulation (to check the pipeline)
```bash
cd hardware/origin_quantum
python3 run_wukong.py --backend local --depol 0.005 --layers 1
```

### 2.3 Calibration data and hardware runs
```bash
python3 chip_info.py --backend WK_C180                # topology and calibration -> chip_info_WK_C180.json
python3 run_wukong.py --list                          # available backends
python3 run_wukong.py --backend WK_C180 --layers 1 --shots 1000 --K 8 --batch
python3 run_wukong.py --backend WK_C180 --layers 2 --shots 1000 --K 8 --batch
```
Use the backend name printed by `--list` if it differs. The script enables qubit mapping, circuit
optimization, and read-out error mitigation. Further options: `--max` (number of instances),
`--decompose` (CNOT-RZ-CNOT instead of RZZ), `--job` (fetch the results of a submitted job).

The results are written to `results_<backend>_p<layers>.json`. Its `summary` contains
- `p_opt`: probability of sampling the optimal offloading vector (compare with 1/2^n for uniform sampling);
- `best_of_K`: normalized DPP gain of the best of the K most frequent bitstrings, evaluated with the exact critic;
- `random_K`: the same metric for K random bitstrings (baseline);
- `wall_s`: wall-clock time including the cloud queue, which shows that the processor is suitable
  only for offline use, not for per-frame real-time decisions.

---

## 3. Files in this folder

| File | Content |
|---|---|
| `jetson/bench_pqc_energy.py` | PQC handshake and AES-GCM latency/energy |
| `jetson/bench_local_compute.py` | GPU (ResNet-18) and CPU DVFS computing model |
| `jetson/bench_scheduler.py` | per-frame decision latency of all schedulers |
| `jetson/results/` | measured results (PQC runs, local computing, decision latency, logs) |
| `origin_quantum/export_qubo.py` | QUBO/Ising instances from sampled system states |
| `origin_quantum/run_wukong.py` | QAOA on a local simulator or on the Origin Quantum cloud |
| `origin_quantum/chip_info.py` | topology and calibration of the processor |
| `origin_quantum/instances.json` | the 20 six-WD instances used in the paper |
| `origin_quantum/results_*.json` | hardware (`WK_C180`) and simulator (`local`) results |
| `origin_quantum/chip_info_WK_C180.json`, `WK_C180_calibration_2026-10-21.png` | calibration of 21 October 2026 |
| `photos/` | photos used in the testbed figure |
