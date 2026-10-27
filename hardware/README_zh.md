# 真机测试指南（Jetson Orin Nano + 本源量子超导芯片）

论文中所有标红（`\hw{...}`）的数字都要用这里脚本的实测结果替换。下面给出完整步骤。

---

## 一、NVIDIA Jetson Orin Nano

### 0. 环境（JetPack 5.1+/6.x）
```bash
sudo apt install -y cmake gcc git python3-pip
pip3 install numpy scipy numba liboqs-python cryptography
# PyTorch 请安装 NVIDIA 官方 Jetson wheel（与 JetPack 版本对应），再装 torchvision
git clone <本仓库>  &&  cd palm_code
```
`liboqs-python` 第一次 `import oqs` 会自动编译 liboqs（约 5–10 分钟）。

### 1. 固定功耗模式与频率（保证结果可复现）
```bash
sudo nvpmodel -q              # 查看当前模式
sudo nvpmodel -m 0            # 最大性能模式（Orin Nano 一般是 15W/25W，按型号）
sudo jetson_clocks            # 锁定最高频率
```
论文里写明你用的 nvpmodel 模式、JetPack 版本、liboqs 版本。

### 2. PQC 时延与能耗（对应论文 Table "PQC profile" 的 Jetson 列）
```bash
cd hardware/jetson
# (a) 应用处理器模式：单核满频，模拟“Jetson 类 WD”
sudo taskset -c 2 python3 bench_pqc_energy.py --mode ap
# (b) 低频模式：限制一个核到低频，模拟低端 WD（可选，作为补充数据）
sudo sh -c 'echo 729600 > /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq'
sudo taskset -c 2 python3 bench_pqc_energy.py --mode lowfreq
```
脚本通过 INA3221 (sysfs hwmon) 采样 `VDD_IN` 功率轨，输出：
- `results/pqc_jetson_ap.json`：每个算法（ML-KEM keygen/encaps/decaps，ML-DSA sign/verify，AES-256-GCM）的中位时延、平均功率、空闲功率；
- `results/profile_jetson_ap.json`：仿真器直接使用的参数（握手时间 `t_hs` ms、握手动态能耗 `E_hs` mJ、AES 每 kbit 能耗 `eps_aead`）。

> 注意：INA3221 采样周期约 5 ms，单次操作比采样周期短，所以脚本采用“连续重复执行 3 秒 × 平均功率 × 单次时延”的方式估计能耗，并报告“总能耗”和“扣除空闲后的动态能耗”两种，论文中用动态能耗。

### 3. 本地计算模型（GPU / CPU DVFS，回应审稿人 3-3）
```bash
sudo python3 bench_local_compute.py
```
输出 `results/local_compute.json`：
- GPU 跑 ResNet-18 推理时的输入数据吞吐 `R_mbps`、每 kbit 能耗 `e_per_kbit`（对应论文 $R^G$ 与 $\varrho$）；
- CPU 不同频点功耗拟合的 $\kappa$（W/GHz³）。

把这些数填进 `palm/params.py` 中 `DEVICE_CLASSES["jetson"]` 的 `R_gpu`、`P_gpu0`、`zeta_gpu`、`kappa`（`P_gpu0/R_gpu + zeta_gpu = e_per_kbit`）。

### 4. 调度器实时性（回应审稿人 1-3、AE“实时性证据”）
```bash
cd ../..          # 回到仓库根目录
sudo python3 hardware/jetson/bench_scheduler.py
```
输出 `hardware/jetson/results/latency_profile_jetson.json`：PALM / PALM-G / SCA / TDMA 在 N=10..50 时每帧决策时间的中位数、p99 和调度时功耗。

### 5. 用实测参数重跑仿真
```bash
cp hardware/jetson/results/latency_profile_jetson.json results/latency_profile.json
PROFILE=hardware/jetson/results/profile_jetson_ap.json FRAMES=8000 SEEDS=1,2 \
    python3 experiments/run_online.py E6 E5
python3 experiments/plots.py
```

---

## 二、本源量子超导芯片（QAOA 生成卸载候选）

### 0. 账号与环境
1. 在本源量子云平台（qcloud.originqc.com.cn）注册，进入控制台获取 **API Key**；确认账户有真机（如“悟空”）机时。
2. `pip install pyqpanda3`（本代码用 0.5.0 测试）。
3. `export ORIGIN_QCLOUD_API_KEY=你的key`

### 1. 生成实验实例（在任意电脑上，需先跑过仿真生成 `results/states.pkl`）
```bash
python3 hardware/origin_quantum/export_qubo.py --n 6 --count 20
```
每个实例包含 Ising 系数、已在模拟器上训练好的 QAOA 参数（p=1、p=2）以及所有 2^n 卸载组合的真实 DPP 值（用于离线打分，真机只负责采样）。
建议 n=6（真机上 RZZ 门数 15 个/层，噪声可控）；n=8 时 28 个 RZZ/层，可作为补充。

### 2. 本地含噪模拟（先跑通流程）
```bash
cd hardware/origin_quantum
python3 run_wukong.py --backend local --depol 0.005 --layers 1
```

### 3. 真机运行
```bash
python3 run_wukong.py --list                      # 查看可用后端名称
python3 run_wukong.py --backend origin_wukong --layers 1 --shots 1000 --K 8
python3 run_wukong.py --backend origin_wukong --layers 2 --shots 1000 --K 8
```
（如果后端名称不同，以 `--list` 输出为准。）脚本已打开 qubit mapping、电路优化与读出误差修正（amend）。
输出 `results_origin_wukong_p1.json`，SUMMARY 中三个指标填入论文 Table "QAOA"：
- `p_opt`：采样到真实最优卸载组合的概率（与均匀随机 1/2^n 对比）；
- `best_of_K`：最常出现的 K 个比特串经精确 Critic 评估后的归一化 DPP 增益；
- `random_K`：K 个随机比特串的同一指标（基线）。
另外记录 `wall_s`（排队+执行时间），用于论文中说明“目前量子硬件只能离线/辅助使用，不能逐帧实时决策”。

### 4. 论文中的表述建议
- 量子芯片只用于**优化加速（候选生成）**，与 PQC 安全机制无关；PQC 运行在经典处理器上。
- 不要宣称量子优势；如实报告：真机噪声下 p=1 的 best-of-K 与模拟器/随机基线的对比，以及云端时延。

---

## 三、需要你核对的事项（重要）
1. `palm/params.py` 中 `IOT_M4_MCYC`（Cortex-M4 上 ML-KEM/ML-DSA 的周期数）是参考 pqm4 公开基准量级给出的代表值，请到 pqm4 仓库 benchmarks 表核对后替换（或用你们的 STM32 板实测）。
2. Jetson 类 PQC 参数在实测前是占位值（x86 实测 × 4），实测后必须替换并重跑 E6/E5。
3. 论文中所有红色 `\hw{}` 文字在定稿前必须替换或删除。
