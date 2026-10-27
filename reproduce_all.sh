#!/bin/sh
# Reproduces every result of the paper (about 5-6 h on 2 cores). Resumable: finished runs
# are cached in results/cache and are not recomputed.
set -e
cd "$(dirname "$0")"
mkdir -p results
# --- offline experiments and timing (run the timing steps alone, without other load) ---
python3 experiments/run_offline.py S E2 E9 E10        # sampled states, SIC-order gap, partial offloading, QAOA (simulation)
python3 experiments/run_offline.py E4 E4c E4d          # solver times, decision times (full and real-time configuration)
# --- online experiments ---
FRAMES=10000 SEEDS=1,2,3 PROCS=2 python3 experiments/run_online.py E1 E13
for e in E3 E6 E12 E11 E7 E7b E6b; do
  FRAMES=8000 SEEDS=1,2 PROCS=2 python3 experiments/run_online.py $e
done
FRAMES=6000 SEEDS=1,2 PROCS=2 python3 experiments/run_online.py E5     # uses the Jetson latencies in results/
FRAMES=8000 SEEDS=1 PROCS=2 python3 experiments/run_online.py E8
# --- figures, tables, and open-format data ---
python3 experiments/plots.py
python3 experiments/fig_testbed.py
python3 tools/export_data.py
python3 tools/export_offline.py
