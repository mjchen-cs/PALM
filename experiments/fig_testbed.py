"""Testbed figure: (a) Jetson Orin Nano, (b) Origin Quantum superconducting system,
(c) WK_C180 chip topology with calibrated CZ fidelities, (d) measurement workflow."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import cm, colors
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "hardware", "photos")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figs", "testbed.pdf")
plt.rcParams.update({"font.family": "serif", "font.size": 7})

W, H = 7.16, 1.98
fig = plt.figure(figsize=(W, H))
h = 1.50; y0 = 0.40            # panel height and bottom (inches)
widths = [2.11, 0.97, 1.11, 2.40]; gaps = [0.07, 0.07, 0.42]
xs_ = [0.0]
for w, g in zip(widths[:-1], gaps):
    xs_.append(xs_[-1] + w + g)
def axes(k):
    return fig.add_axes([xs_[k] / W, y0 / H, widths[k] / W, h / H])
def tag(ax, t):
    ax.text(0.5, -0.03, t, transform=ax.transAxes, fontsize=7, ha="center", va="top")
def call(ax, xy, xytext, s, ha="left"):
    ax.annotate(s, xy=xy, xytext=xytext, fontsize=5.2, ha=ha, va="center", color="k",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.3", lw=0.4, alpha=0.9),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#d62728", mutation_scale=5))

# (a) Jetson
ax = axes(0)
im = Image.open(os.path.join(PH, "jetson_raw.jpg")).crop((0, 862, 640, 1300))
ax.imshow(im, aspect="auto"); ax.set_axis_off(); tag(ax, "(a) NVIDIA Jetson Orin Nano")
call(ax, (440, 140), (470, 30), "Orin Nano module", ha="center")
call(ax, (232, 62), (20, 28), "power in (VDD_IN)", ha="left")
call(ax, (205, 290), (15, 400), "Ethernet (SSH)", ha="left")
call(ax, (330, 330), (520, 405), "edge controller /\nGPU-equipped WD", ha="center")

# (b) quantum computer
ax = axes(1)
im = Image.open(os.path.join(PH, "wukong_raw.png")).crop((60, 160, 820, 1365))
ax.imshow(im, aspect="auto"); ax.set_axis_off(); tag(ax, "(b) Origin Quantum")
call(ax, (330, 500), (380, 90), "dilution\nrefrigerator", ha="center")
call(ax, (330, 1050), (380, 1150), "QPU stage", ha="center")

# (c) WK_C180 topology
ax = axes(2)
c = json.load(open(os.path.join(ROOT, "hardware", "origin_quantum", "chip_info_WK_C180.json")))
pos = {q: ((2 * (q % 9) + (1 if (q // 9) % 2 == 0 else 0)), -(q // 9)) for q in range(180)}
fid = {tuple(sorted(map(int, d["get_qubits"]))): d["get_fidelity"] for d in c["double_qubits_info"]}
cmap = cm.viridis; norm = colors.Normalize(0.95, 1.0)
for e in c["get_chip_topology"]:
    a_, b_ = int(e[0]), int(e[1]); f = fid.get(tuple(sorted((a_, b_))))
    ax.plot([pos[a_][0], pos[b_][0]], [pos[a_][1], pos[b_][1]], color=cmap(norm(f)) if f else "0.85",
            lw=1.0, solid_capstyle="round")
avail = set(int(q) for q in c["available_qubits"])
xs = np.array([pos[q][0] for q in range(180)]); ys = np.array([pos[q][1] for q in range(180)])
ok = np.array([q in avail for q in range(180)])
ax.scatter(xs[ok], ys[ok], s=2.5, c="#1f3b73", zorder=3, lw=0)
ax.scatter(xs[~ok], ys[~ok], s=2.5, facecolors="white", edgecolors="0.4", lw=0.3, zorder=3)
ax.set_xlim(-0.6, 17.6); ax.set_ylim(-19.8, 0.6); ax.set_axis_off()
ax.text(0.5, -0.03, "(c) WK_C180 chip", transform=ax.transAxes, fontsize=7, ha="center", va="top")
cax = fig.add_axes([(xs_[2] + widths[2] + 0.005) / W, (y0 + 0.22) / H, 0.045 / W, 1.0 / H])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_ticks([0.95, 0.97, 0.99]); cb.ax.tick_params(labelsize=4.5, length=1.2, pad=1); cb.set_label("CZ fidelity", fontsize=5, labelpad=1.5)
ax.text(0.5, -0.115, "166/180 qubits; median fidelity 1q 99.88%,\nCZ 97.93%, read-out 92.86%", transform=ax.transAxes,
        ha="center", va="top", fontsize=4.9)

# (d) workflow
ax = axes(3); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_axis_off()
tag(ax, "(d) Measurement workflow")
def box(x, y, w, hh, s, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.1", fc=fc, ec="0.35", lw=0.5))
    ax.text(x + w / 2, y + hh / 2, s, ha="center", va="center", fontsize=5.0, linespacing=1.1)
def arr(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.6, color="0.25"))
box(0.6, 7.0, 4.5, 2.8, "Jetson (a):\nPQC (liboqs), AES-GCM,\nResNet-18, scheduler", "#e8f1fb")
box(5.35, 7.0, 4.5, 2.8, "profiles:\n$\\tau^{\\rm hs}$, $E^{\\rm hs}$, $\\varepsilon_i$,\n$R^G$, $\\varrho$, $\\tau^{\\rm dec}$", "#e8f1fb")
box(2.75, 3.75, 4.5, 2.3, "PALM-PS\nsimulation (E1-E12)", "#fff4e0")
box(0.6, 0.2, 4.5, 2.8, "QUBO surrogate\n$\\rightarrow$ QAOA circuit\n(pyqpanda3)", "#eaf6ec")
box(5.35, 0.2, 4.5, 2.8, "WK_C180 (b, c):\n1000 shots $\\rightarrow$ 8 best\n$\\rightarrow$ exact critic", "#eaf6ec")
arr((4.75, 8.4), (5.25, 8.4)); arr((7.6, 6.9), (6.2, 6.15)); arr((3.8, 3.65), (2.6, 3.1))
arr((4.75, 1.6), (5.25, 1.6)); arr((7.6, 3.1), (6.2, 3.65))
fig.savefig(OUT, dpi=400)
print("saved", OUT)
