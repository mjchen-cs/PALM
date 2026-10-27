"""Generate all paper figures/tables from results/*.pkl|json  ->  figs/"""
import os, sys, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(F, exist_ok=True)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
       "#8c564b", "#17becf", "#7f7f7f", "#bcbd22", "#000000", "#9467bd"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "d", "p", "8"]
LS = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#dddddd", "grid.linewidth": 0.5, "figure.dpi": 200,
                     "savefig.bbox": "tight", "font.family": "serif"})
ORDER = ["PALM", "PALM-G", "LyDROO-SCA", "LyDROO-Greedy", "LyDROO-TDMA", "PQC-agnostic", "DDPG",
         "LyDiff", "CEA", "LyDDPG-Att", "KeepAlive", "PALM-G-PS", "PALM-PS"]
LABEL = {"PALM": "PALM", "PALM-G": "PALM-G (greedy critic)", "LyDROO-SCA": "NOMA-SCA (ch. order)",
         "LyDROO-Greedy": "NOMA-Greedy (ch. order) [earlier]", "LyDROO-TDMA": "TDMA",
         "PQC-agnostic": "PQC-agnostic", "DDPG": "DDPG (model-free)",
         "NOMA-ANNS": "NOMA-ANNS", "LyDiff": "LyDiff", "CEA": "CEA",
         "LyDDPG-Att": "LyDDPG-Att", "PALM-PS": "PALM-PS (proactive)", "PALM-G-PS": "PALM-G-PS",
         "KeepAlive": "Keep-alive"}
COL = {k: PAL[i] for i, k in enumerate(ORDER)}
SYM = {k: MK[i] for i, k in enumerate(ORDER)}
# real-time configuration (not part of the main comparison)
COL.update({"PALM-PS-RT": "#9467bd", "PALM-RT": "#a65628"}); SYM.update({"PALM-PS-RT": "8", "PALM-RT": "H"})
LABEL.update({"PALM-PS-RT": "PALM-PS-RT (real-time)", "PALM-RT": "PALM-RT"})
lsof = lambda k: LS[ORDER.index(k)] if k in ORDER else {"PALM-PS-RT": "-", "PALM-RT": "--"}.get(k, ":")


def label_panels(axs, loc="title"):
    """Mark subplots as (a), (b), (c) so that the text can refer to them."""
    axs = list(np.ravel(axs))
    for k, a in enumerate(axs):
        tag = f"({chr(97 + k)})"
        t = a.get_title()
        if t:   # panel already has a centred title: prefix it
            a.set_title(f"{tag} {t}", fontsize=8)
        else:
            a.set_title(tag, fontsize=8, loc="left", pad=3, fontweight="bold")


def load(name):
    p = os.path.join(R, name + ".pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def ma(x, w=200):
    c = np.cumsum(np.insert(x, 0, 0))
    out = (c[w:] - c[:-w]) / w
    return np.r_[np.full(w - 1, np.nan), out]


def agg(rows, key, by):
    """mean/std over seeds of summary[key] grouped by tuple(tag[b] for b in by)"""
    d = {}
    for tg, s, _ in rows:
        k = tuple(tg.get(b) for b in by)
        d.setdefault(k, []).append(s[key])
    return {k: (np.mean(v, 0), np.std(v, 0)) for k, v in d.items()}


def fig_E1():
    rows = load("E1")
    if rows is None:
        return
    fig, ax = plt.subplots(3, 1, figsize=(3.5, 3.6), sharex=True)
    for sch in ["PALM-PS", "PALM", "LyDROO-SCA", "LyDROO-Greedy", "LyDROO-TDMA", "LyDiff", "CEA", "DDPG"]:
        tr = [t for tg, s, t in rows if tg["scheme"] == sch]
        if not tr:
            continue
        for a, key, sc in zip(ax, ["Q", "E", "off"], [1e-3, 1.0, 100.0]):
            y = np.nanmean([ma(np.asarray(t[key], float)) for t in tr], 0) * sc
            a.plot(np.arange(len(y)), y, color=COL[sch], ls=LS[ORDER.index(sch)], label=LABEL[sch], lw=1.0)
    ax[0].set_ylabel("Queue (Mbit)"); ax[0].set_yscale("log")
    ax[1].set_ylabel("WD power (W)")
    ax[2].set_ylabel("Offloading (%)"); ax[2].set_xlabel("Time frame")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 0.99), frameon=False)
    label_panels(ax); fig.savefig(os.path.join(F, "conv.pdf")); plt.close(fig)
    # table
    lines = []
    TAB = ["PALM-PS", "PALM-G-PS", "PALM", "PALM-G", "LyDROO-SCA", "LyDROO-Greedy", "LyDROO-TDMA", "PQC-agnostic",
           "KeepAlive", "LyDiff", "CEA", "DDPG", "LyDDPG-Att"]
    for sch in TAB:
        ss = [s for tg, s, _ in rows if tg["scheme"] == sch]
        if not ss:
            continue
        m = lambda k, f=1.0: (np.mean([x[k] for x in ss]) * f, np.std([x[k] for x in ss]) * f)
        Q, E, Ees, off, dt = m("Q", 1e-3), m("E"), m("Ees"), m("off", 100), m("dt", 1e3)
        jb = np.mean([np.mean(x["perdev_Jbit"]) for x in ss]) * 1e9
        nm = "\\textbf{PALM-PS}" if sch == "PALM-PS" else LABEL[sch]
        qs = f"{Q[0]:.3f}$\\pm${Q[1]:.3f}" if Q[0] < 50 else f"{Q[0]:.1f} (div.)"
        lines.append(f"{nm} & {qs} & {E[0]:.3f} & {Ees[0]:.2f} & {jb:.1f} & {off[0]:.2f} & {dt[0]:.2f} \\\\")
    open(os.path.join(F, "tab_E1.tex"), "w").write("\n".join(lines))
    # per-device power and J/bit
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 2.3), sharex=True)
    sel = ["PALM-PS", "PALM", "LyDROO-Greedy", "LyDROO-TDMA", "DDPG"]
    N = len(rows[0][1]["perdev_P"])
    w = 0.8 / len(sel)
    gam = np.array([0.08] * 6 + [0.25] * 4)
    for j, sch in enumerate(sel):
        ss = [s for tg, s, _ in rows if tg["scheme"] == sch]
        if not ss:
            continue
        P = np.mean([x["perdev_P"] for x in ss], 0)
        Jb = np.mean([x["perdev_Jbit"] for x in ss], 0) * 1e9
        xs = np.arange(N) + (j - (len(sel) - 1) / 2) * w
        ax[0].bar(xs, P, w * 0.9, color=COL[sch], label=LABEL[sch])
        ax[1].bar(xs, Jb, w * 0.9, color=COL[sch])
    ax[0].step(np.arange(N + 1) - 0.5, np.r_[gam, gam[-1]], where="post", color="k", lw=0.8, ls="--", label="budget $\\gamma_i$")
    ax[0].set_ylabel("Avg. power (W)"); ax[1].set_ylabel("Energy per bit (nJ/bit)")
    ax[1].set_xticks(range(N)); ax[1].set_xticklabels([f"{i+1}\n{'IoT' if i < 6 else 'Jet.'}" for i in range(N)])
    ax[1].set_xlabel("WD index / class")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 0.99), frameon=False)
    label_panels(ax); fig.savefig(os.path.join(F, "perdev.pdf")); plt.close(fig)


def fig_E2():
    p = os.path.join(R, "E2.pkl")
    if not os.path.exists(p):
        return
    rows = pickle.load(open(p, "rb"))
    keys = [("exact", "Alg.1 (weight order, exact)"), ("search", "Alg.1+Alg.2 (budget 3 ms)"),
            ("search_full", "Alg.1+Alg.2 (no budget)"), ("beta_sca", "SCA, $\\beta$ order"),
            ("sca_c", "SCA, channel order"),
            ("greedy_c", "Greedy, channel order [earlier]")]
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    lines = []
    for i, (k, lab) in enumerate(keys):
        g = np.array([(x["global_best"] - x[k]) / abs(x["global_best"]) for x in rows])
        g = np.clip(g[np.isfinite(g)], 1e-7, None)
        xs = np.sort(g); ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs * 100, ys, where="post", color=PAL[i], ls=LS[i], label=lab)
        ub = np.array([(x["ub"] - x[k]) / abs(x["ub"]) for x in rows])
        lines.append(f"{lab} & {np.mean(g)*100:.2f} & {np.quantile(g,.95)*100:.2f} & {np.max(g)*100:.2f} & {np.mean(g<1e-6)*100:.2f} \\\\")
    ax.set_xscale("log"); ax.set_xlim(1e-4, 100)
    ax.set_xlabel("Gap to best over all $M!$ SIC orders (%)"); ax.set_ylabel("CDF")
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.savefig(os.path.join(F, "gap.pdf")); plt.close(fig)
    open(os.path.join(F, "tab_E2.tex"), "w").write("\n".join(lines))
    ubg = np.array([(x["ub"] - x["search"]) / abs(x["ub"]) for x in rows])
    open(os.path.join(F, "num_E2.json"), "w").write(json.dumps(dict(n=len(rows), ub_gap_mean=float(np.mean(ubg)),
                                                                 ub_gap_med=float(np.median(ubg)))))


def fig_E4():
    p = os.path.join(R, "E4.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    M = np.array(d["M"])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.85))
    for i, (k, lab) in enumerate([("exact", "Alg.1 (exact, weight order)"), ("greedy", "Greedy backward"),
                                  ("sca_numba", "SCA (Alg.1 inner solver)"), ("sca_cvx", "SCA (CVXPY+Clarabel)"),
                                  ("convex_cvx", "Convex $\\mathcal{P}3$ via CVXPY"), ("search", "Alg.2 refinement")]):
        if k not in d:
            continue
        v = np.array([x[0] for x in d[k]]) * 1e3
        ax[0].plot(M, v, marker=MK[i], ms=3, color=PAL[i], ls=LS[i], label=lab)
    ax[0].set_yscale("log"); ax[0].set_xlabel("Offloading WDs $M$"); ax[0].set_ylabel("Solver time (ms)")
    ax[0].legend(fontsize=6, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.42, -0.25), columnspacing=0.8)
    dec = d["decision_ms"]
    for i, sch in enumerate(["PALM-PS", "PALM-PS-RT", "PALM-G-PS", "PALM", "PALM-RT", "PALM-G", "LyDROO-SCA", "LyDROO-TDMA", "LyDiff", "CEA", "LyDDPG-Att"]):
        if sch not in dec:
            continue
        Ns = sorted(int(n) for n in dec[sch])
        ax[1].errorbar(Ns, [dec[sch][str(n)][0] for n in Ns], yerr=None, marker=SYM[sch], ms=3, color=COL[sch],
                       label=LABEL[sch], ls=lsof(sch))
        if sch in ("PALM-PS", "PALM-PS-RT", "PALM-G-PS"):
            ax[1].plot(Ns, [dec[sch][str(n)][1] for n in Ns], color=COL[sch], ls=":", lw=0.8)
    pj = os.path.join(R, "latency_profile_jetson_v2.json")
    if os.path.exists(pj):
        dj = json.load(open(pj))
        prt = os.path.join(R, "latency_profile_jetson_rt.json")
        if os.path.exists(prt):
            dj.update({k: v for k, v in json.load(open(prt)).items() if isinstance(v, dict)})
        for sch in ["PALM-PS", "PALM-PS-RT", "PALM-G-PS"]:
            Ns = sorted(int(n) for n in dj[sch] if n.isdigit())
            ax[1].plot(Ns, [dj[sch][str(n)] for n in Ns], marker=SYM[sch], ms=3, color=COL[sch], ls="none",
                       mfc="white", label=None)
        ax[1].plot([], [], marker="o", ms=3, color="k", ls="none", mfc="white", label="hollow: Jetson (meas.)")
    ax[1].axhline(50, color="k", lw=0.8, ls="--"); ax[1].text(10, 55, "frame $T$=50 ms", fontsize=6)
    ax[1].set_yscale("log"); ax[1].set_xlabel("Number of WDs $N$"); ax[1].set_ylabel("Decision time per frame (ms)")
    ax[1].legend(fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.6, -0.25), columnspacing=0.8)
    label_panels(ax); fig.savefig(os.path.join(F, "runtime.pdf")); plt.close(fig)
    # slopes
    sl = {k: float(np.polyfit(np.log(M), np.log([x[0] for x in d[k]]), 1)[0]) for k in ["exact", "sca_numba", "sca_cvx", "convex_cvx"]}
    open(os.path.join(F, "num_E4.json"), "w").write(json.dumps(sl))



def _backlog_main():
    """Average backlog (Mbit) in the main setting (E1, plus E13 for the real-time configuration)."""
    r = {}
    for nm in ("E1", "E13"):
        rows = load(nm) or []
        for tg, s, _ in rows:
            if tg.get("rekey", 20) == 20:
                r.setdefault(tg["scheme"], []).append(s["Q"] / 1e3)
    return {k: float(np.mean(v)) for k, v in r.items()}


def tradeoff(a, xeon, N="10"):
    """Decision latency on the Jetson (N=10) versus backlog in the main setting (E1/E13)."""
    from matplotlib.ticker import NullFormatter
    jet = {}
    for fn in ("latency_profile_jetson_v2.json", "latency_profile_jetson_rt.json", "latency_profile_jetson_base.json"):
        pj = os.path.join(R, fn)
        if os.path.exists(pj):
            jet.update({k: v for k, v in json.load(open(pj)).items() if isinstance(v, dict)})
    Q = _backlog_main()
    ours = ["PALM-G", "PALM-G-PS", "PALM-RT", "PALM-PS-RT", "PALM", "PALM-PS"]
    base = ["LyDROO-SCA", "LyDROO-TDMA", "LyDiff", "CEA", "LyDDPG-Att", "LyDROO-Greedy", "PQC-agnostic"]
    short = {"PALM-G": "PALM-G", "PALM-G-PS": "G-PS", "PALM-RT": "PALM-RT", "PALM-PS-RT": "PS-RT",
             "PALM": "PALM", "PALM-PS": "PALM-PS", "LyDROO-SCA": "SCA", "LyDROO-TDMA": "TDMA",
             "LyDiff": "LyDiff", "CEA": "CEA", "LyDDPG-Att": "LyDDPG-Att", "LyDROO-Greedy": "earlier version",
             "PQC-agnostic": "agnostic"}
    off = {"PALM-G": (3, 3), "PALM-G-PS": (-4, -9), "PALM-RT": (2, -8), "PALM-PS-RT": (-26, -7),
           "PALM": (-24, 2), "PALM-PS": (-30, -3), "LyDROO-SCA": (-6, 4), "LyDiff": (3, -3), "CEA": (2, 2),
           "LyDROO-Greedy": (4, -2), "PQC-agnostic": (3, 2)}
    pts, unstable = {}, []
    for sch in ours + base:
        if sch not in Q:
            continue
        est = sch not in jet or N not in jet[sch]
        if est and sch in xeon:
            lat = xeon[sch][N][0] * 3.0          # placeholder until measured on the Jetson
        elif est:
            continue
        else:
            lat = jet[sch][N]
        if Q[sch] > 0.3:                              # off scale
            unstable.append(short[sch] + (" (unstable)" if Q[sch] > 10 else f" ({Q[sch]:.2f})")); continue
        pts[sch] = (lat, Q[sch], est)
    front = []
    for sch in sorted([s for s in ours if s in pts], key=lambda s: pts[s][0]):
        if not front or pts[sch][1] < pts[front[-1]][1]:
            front.append(sch)
    a.plot([pts[s][0] for s in front], [pts[s][1] for s in front], color="k", lw=0.8, ls="-", zorder=1)
    for sch, (lat, q, est) in pts.items():
        mine = sch in ours
        c = COL.get(sch, "#444444")
        a.plot(lat, q, marker=SYM.get(sch, "o"), ms=4 if mine else 3.5, ls="none", color=c,
               mfc=("white" if est else c), zorder=3)
        if not mine and sch != "LyDROO-SCA":       # our schemes, SCA and TDMA use the legend markers
            a.annotate(short[sch], (lat, q), fontsize=5.5, xytext=off.get(sch, (2, 2)), textcoords="offset points",
                       color="#555555")
    a.set_xscale("log"); a.set_xlabel("Decision time on the Jetson, $N{=}10$ (ms)")
    a.set_ylabel("Avg. data queue (Mbit)")
    a.set_xticks([2, 5, 10, 20, 50]); a.set_xticklabels(["2", "5", "10", "20", "50"])
    a.xaxis.set_minor_formatter(NullFormatter())
    a.set_xlim(1.6, 60)
    open(os.path.join(F, "num_tradeoff.txt"), "w").write(json.dumps({"pts": pts, "front": front, "off": unstable}))
    a.set_ylim(0.16, 0.255)

def fig_E3():
    rows = load("E3")
    if rows is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.7))
    for sch in ["PALM-PS", "PALM-G-PS", "PALM", "PALM-G", "LyDROO-SCA", "LyDROO-Greedy", "LyDROO-TDMA", "KeepAlive", "LyDiff", "CEA", "LyDDPG-Att"]:
        rr = [(tg, s) for tg, s, _ in rows if tg["scheme"] == sch]
        if not rr:
            continue
        lams = sorted(set(tg["lam"] for tg, _ in rr))
        for a, key, sc in zip(ax, ["Q", "E", "wrate"], [1e-3, 1, 1]):
            m = [np.mean([s[key] for tg, s in rr if tg["lam"] == l]) * sc for l in lams]
            a.plot(lams, m, marker=SYM[sch], ms=3, color=COL[sch], ls=LS[ORDER.index(sch)], label=LABEL[sch])
    lams = np.array(sorted(set(tg["lam"] for tg, s, _ in rows)))
    ax[2].plot(lams, lams * 12.5, color="k", lw=0.8, ls="--", label="weighted arrival")
    ax[0].set_yscale("log"); ax[0].set_ylabel("Avg. data queue (Mbit)")
    ax[1].set_ylabel("Avg. WD power (W)"); ax[2].set_ylabel("Weighted rate (Mbps)")
    for a in ax:
        a.set_xlabel("Arrival rate $\\lambda_i$ (Mbps)")
    fig.subplots_adjust(wspace=0.42)
    h, l = ax[0].get_legend_handles_labels(); h2, l2 = ax[2].get_legend_handles_labels()
    fig.legend(h + h2[-1:], l + l2[-1:], ncol=5, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 0.97), frameon=False)
    label_panels(ax); fig.savefig(os.path.join(F, "load.pdf")); plt.close(fig)
    # stability boundary: largest lambda with queue slope < thr in all seeds
    lines = []
    for sch in ["PALM-PS", "PALM-G-PS", "PALM", "PALM-G", "LyDROO-SCA", "LyDROO-Greedy", "LyDROO-TDMA", "KeepAlive", "LyDiff", "CEA", "LyDDPG-Att"]:
        rr = [(tg, s) for tg, s, _ in rows if tg["scheme"] == sch]
        st = [l for l in sorted(set(tg["lam"] for tg, _ in rr))
              if all(s["Qslope"] < 0.5 and s["Q"] < 5000 for tg, s in rr if tg["lam"] == l)]
        lines.append(f"{sch}: max stable lambda = {max(st) if st else 'none'}")
    open(os.path.join(F, "num_E3.txt"), "w").write("\n".join(lines))


def fig_E6():
    rows = load("E6")
    if rows is None:
        return
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.8))
    rks = sorted(set(tg["rekey"] for tg, s, _ in rows))
    for i, lev in enumerate([1, 3, 5]):
        for sch, ls in [("PALM-PS", "-"), ("PALM", ":"), ("PQC-agnostic", "--")]:
            for a, key, sc in zip(ax, ["off", "Q", "viol"], [100, 1e-3, 1]):
                get = (lambda s: np.max(s["perdev_P"][:6] / 0.08)) if key == "viol" else (lambda s, k=key: s[k])
                m = [np.mean([get(s) for tg, s, _ in rows if tg["scheme"] == sch and tg["sec"] == lev and tg["rekey"] == r]) * sc for r in rks]
                a.plot(rks, m, color=PAL[i], ls=ls, marker=MK[i], ms=3,
                       label=f"{ {'PALM-PS': 'PALM-PS', 'PALM': 'PALM', 'PQC-agnostic': 'agnostic'}[sch]}, cat. {lev}")
    r11 = load("E11")
    if r11 is not None:
        sel = lambda sch, r: [x for tg, x, _ in r11 if tg.get("part") == "a" and tg["scheme"] == sch and tg["rekey"] == r]
        rk11 = sorted(set(tg["rekey"] for tg, x, _ in r11 if tg.get("part") == "a"))
        q0 = np.array([np.mean([x["Q"] for x in sel("PALM", r)]) for r in rk11])
        ax[3].axhline(1.0, color=PAL[1], lw=1.0, ls="-", label="PALM (on-demand)")
        for sch, lab, c, mk, ls in [("PALM-PS", "PALM-PS (proactive)", PAL[6], "s", "--"),
                                    ("KeepAlive", "Keep-alive", PAL[7], "^", "-.")]:
            q = np.array([np.mean([x["Q"] for x in sel(sch, r)]) for r in rk11])
            ax[3].plot(rk11, q / q0, color=c, marker=mk, ms=3, ls=ls, label=lab)
        ax[3].set_yscale("log")
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
        tk = [0.5, 1, 2, 4, 8, 16, 32]
        ax[3].yaxis.set_major_locator(FixedLocator(tk)); ax[3].yaxis.set_major_formatter(FixedFormatter([str(t) for t in tk]))
        ax[3].yaxis.set_minor_locator(NullLocator()); ax[3].set_ylim(0.45, 120)
        ax[3].set_ylabel("Backlog relative to PALM")
        ax[3].legend(fontsize=5.5, frameon=False, loc="upper right")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("$L^{\\mathrm{rk}}$ (frames)")
    ax[0].set_ylabel("Offloading ratio (%)"); ax[1].set_ylabel("Avg. data queue (Mbit)"); ax[1].set_yscale("log")
    ax[2].set_ylabel("Max IoT power / budget $\\gamma_i$"); ax[2].axhline(1.0, color="k", lw=0.8, ls=":")
    fig.subplots_adjust(wspace=0.5, left=0.06, right=0.99)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 0.97), frameon=False)
    label_panels(ax); fig.savefig(os.path.join(F, "pqc.pdf")); plt.close(fig)
    lines = []
    for lev in [1, 3, 5]:
        for r in rks:
            for sch in ["PALM", "PQC-agnostic"]:
                ss = [s for tg, s, _ in rows if tg["scheme"] == sch and tg["sec"] == lev and tg["rekey"] == r]
                pdv = np.mean([x["perdev_P"] for x in ss], 0)
                viol = np.max(pdv[:6] / 0.08)
                lines.append(f"{lev} {r} {sch} Q={np.mean([x['Q'] for x in ss])/1e3:.2f} off={np.mean([x['off'] for x in ss])*100:.1f} "
                             f"E={np.mean([x['E'] for x in ss]):.4f} hs={np.mean([x['hs'] for x in ss]):.3f} iotPmax/gamma={viol:.3f} "
                             f"iotoff={np.mean([np.mean(x['perdev_off'][:6]) for x in ss])*100:.1f} jetoff={np.mean([np.mean(x['perdev_off'][6:]) for x in ss])*100:.1f}")
    open(os.path.join(F, "num_E6.txt"), "w").write("\n".join(lines))


def fig_E6b():
    rows = load("E6b")
    if rows is None:
        return
    lines = []
    for key in ["pkt", "pst", "om"]:
        vals = sorted(set(tg[key] for tg, s, _ in rows if key in tg))
        for v in vals:
            ss = [s for tg, s, _ in rows if tg.get(key) == v]
            lines.append(f"{key}={v}: Q={np.mean([x['Q'] for x in ss])/1e3:.3f}Mb off={np.mean([x['off'] for x in ss])*100:.1f}% "
                         f"E={np.mean([x['E'] for x in ss]):.4f}W hs={np.mean([x['hs'] for x in ss]):.3f} "
                         f"iotoff={np.mean([np.mean(x['perdev_off'][:6]) for x in ss])*100:.1f} Jbit={np.mean([np.mean(x['perdev_Jbit']) for x in ss])*1e9:.1f}nJ")
    open(os.path.join(F, "num_E6b.txt"), "w").write("\n".join(lines))


def fig_E7():
    rows = load("E7")
    if rows is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.7))
    rb = load("E7b")
    if rb is not None:
        xs = sorted(set(tg["xi"] for tg, s, _ in rb if tg["xi"] >= 0))
        for i, aw in enumerate([0, 1]):
            m = [np.mean([s["Q"] for tg, s, _ in rb if tg["xi"] == x and tg["aware"] == aw]) / 1e3 for x in xs]
            ax[0].plot([max(x, 3e-4) for x in xs], m, marker=MK[i], ms=3, color=PAL[i], ls=LS[i],
                       label="PALM-PS (SIC-unaware)" if aw == 0 else "PALM-PS-R (SIC-aware)")
        tq = np.mean([s["Q"] for tg, s, _ in rb if tg["xi"] == -1]) / 1e3
        ax[0].axhline(tq, color=COL["LyDROO-TDMA"], ls=":", lw=1.0, label="TDMA")
        ax[0].set_xscale("log")
    else:
        xis = sorted(set(tg["xi"] for tg, s, _ in rows if "xi" in tg))
        for i, rm in enumerate(["adaptive", "fixed"]):
            m = [np.mean([s["Q"] for tg, s, _ in rows if tg.get("xi") == x and tg.get("rm") == rm]) / 1e3 for x in xis]
            ax[0].plot(xis, m, marker=MK[i], ms=3, color=PAL[i], ls=LS[i], label=f"rate {rm}")
    ax[0].set_xlabel("Residual SIC factor $\\xi$"); ax[0].set_ylabel("Avg. data queue (Mbit)"); ax[0].set_yscale("log")
    ax[0].legend(fontsize=6, frameon=False)
    sds = sorted(set(tg["csi"] for tg, s, _ in rows if "csi" in tg))
    for i, bo in enumerate([0.0, 1.0, 2.0]):
        m = [np.mean([s["Q"] for tg, s, _ in rows if tg.get("csi") == x and tg.get("bo") == bo] or [np.nan]) / 1e3 for x in sds]
        ax[1].plot(sds, m, marker=MK[i], ms=3, color=PAL[i], ls=LS[i],
                   label=["no back-off", "back-off $2\\sigma_e$", "robust rates"][i])
    ax[1].set_xlabel("CSI error std. $\\sigma_e$"); ax[1].set_ylabel("Avg. data queue (Mbit)"); ax[1].set_yscale("log")
    ax[1].legend(fontsize=6, frameon=False)
    mixes = [10, 6, 2]
    w = 0.2
    for j, sch in enumerate(["PALM-PS", "PALM", "LyDROO-Greedy", "LyDROO-TDMA"]):
        m = [np.mean([s["Q"] for tg, s, _ in rows if tg.get("mix") == mx and tg["scheme"] == sch]) / 1e3 for mx in mixes]
        ax[2].bar(np.arange(3) + (j - 1.5) * w, m, w * 0.9, color=COL[sch], label=LABEL[sch])
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["10 IoT", "6 IoT\n4 Jetson", "2 IoT\n8 Jetson"])
    ax[2].set_ylabel("Avg. data queue (Mbit)"); ax[2].set_yscale("log")
    ax[2].set_ylim(top=ax[2].get_ylim()[1] * 300); _h, _l = ax[2].get_legend_handles_labels(); ax[2].legend(_h, [x.replace(" (ch. order)", "") for x in _l], fontsize=6, frameon=False, loc="upper center", ncol=1)
    fig.subplots_adjust(wspace=0.42)
    label_panels(ax); fig.savefig(os.path.join(F, "robust.pdf")); plt.close(fig)
    lines = []
    for tg, s, _ in rows:
        pass
    rb = load("E7b") or []
    for tg, s_, _ in rb:
        pass
    groups = {}
    for tg, s_, _ in rb:
        groups.setdefault((tg["xi"], tg["aware"]), []).append(s_)
    for g, ss in sorted(groups.items()):
        lines.append(f"E7b xi,aware={g}: Q={np.mean([x['Q'] for x in ss])/1e3:.3f} E={np.mean([x['E'] for x in ss]):.4f} off={np.mean([x['off'] for x in ss])*100:.1f} slope={np.mean([x['Qslope'] for x in ss]):.2f}")
    keys = [("xi", "rm"), ("csi", "bo"), ("mix", "scheme"), ("esg",)]
    for kk in keys:
        groups = {}
        for tg, s, _ in rows:
            if kk[0] in tg:
                groups.setdefault(tuple(tg.get(k, tg.get("scheme")) for k in kk), []).append(s)
        for g, ss in sorted(groups.items(), key=lambda z: str(z[0])):
            lines.append(f"{kk}={g}: Q={np.mean([x['Q'] for x in ss])/1e3:.3f} E={np.mean([x['E'] for x in ss]):.4f} "
                         f"Ees={np.mean([x['Ees'] for x in ss]):.3f} off={np.mean([x['off'] for x in ss])*100:.1f} wrate={np.mean([x['wrate'] for x in ss]):.2f}")
    open(os.path.join(F, "num_E7.txt"), "w").write("\n".join(lines))


def fig_E8():
    rows = load("E8")
    if rows is None:
        return
    Vs = sorted(set(tg["V"] for tg, s, _ in rows)); nus = sorted(set(tg["nu"] for tg, s, _ in rows))
    Qm = np.zeros((len(Vs), len(nus))); Pm = np.zeros_like(Qm)
    for tg, s, _ in rows:
        i, j = Vs.index(tg["V"]), nus.index(tg["nu"])
        Qm[i, j] = s["Q"] / 1e3
        Pm[i, j] = np.max(s["perdev_P"] / np.array([0.08] * 6 + [0.25] * 4))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.42))
    fig.subplots_adjust(wspace=0.35)
    for a, M, t, cm in [(ax[0], np.log10(Qm), "Avg. data queue (Mbit; log color scale)", "Blues"),
                        (ax[1], Pm, "max$_i$ avg. power / $\\gamma_i$", "Oranges")]:
        im = a.imshow(M, cmap=cm, aspect="auto", origin="lower")
        a.set_xticks(range(len(nus))); a.set_xticklabels(nus); a.set_yticks(range(len(Vs))); a.set_yticklabels(Vs)
        a.set_xlabel("$\\nu$"); a.set_ylabel("$V$"); a.set_title(t); a.grid(False)
        for i in range(len(Vs)):
            for j in range(len(nus)):
                frac = (M[i, j] - M.min()) / max(M.max() - M.min(), 1e-12)
                a.text(j, i, f"{(10**M[i,j] if a is ax[0] else M[i,j]):.2f}", ha="center", va="center", fontsize=6,
                       color="white" if frac > 0.6 else "k")
        plt.colorbar(im, ax=a, fraction=0.046)
    label_panels(ax); fig.savefig(os.path.join(F, "vnu.pdf")); plt.close(fig)


def fig_E5():
    rows = load("E5")
    if rows is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.45), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    lines = []
    labs = ["PALM-PS", "PALM-PS-RT", "PALM-G-PS", "PALM", "PALM-RT", "PALM-G", "LyDROO-SCA", "LyDROO-TDMA", "PALM-PS-ideal"]
    for j, lab in enumerate(labs):
        rr = [(tg, s) for tg, s, _ in rows if tg.get("label", tg["scheme"]) == lab]
        if not rr:
            continue
        Ts = sorted(set(tg["T"] for tg, _ in rr))
        q = [np.mean([s["Q"] for tg, s in rr if tg["T"] == T]) / 1e3 for T in Ts]
        e = [np.mean([s["E"] for tg, s in rr if tg["T"] == T]) for T in Ts]
        c = COL.get(lab, "#444444"); mk = SYM.get(lab, "x"); ls = lsof(lab)
        nm = LABEL.get(lab, "PALM-PS, zero latency (ideal)")
        ax[0].plot(Ts, q, marker=mk, ms=3, color=c, ls=ls, label=nm)
        ax[1].plot(Ts, e, marker=mk, ms=3, color=c, ls=ls, label=nm)
        for T in Ts:
            ss = [s for tg, s in rr if tg["T"] == T]; tau = [tg["tau"] for tg, s in rr if tg["T"] == T][0]
            lines.append(f"{lab} T={T} tau={tau:.2f} Q={np.mean([x['Q'] for x in ss])/1e3:.3f} slope={np.mean([x['Qslope'] for x in ss]):.2f} off={np.mean([x['off'] for x in ss])*100:.1f} wrate={np.mean([x['wrate'] for x in ss]):.2f} E={np.mean([x['E'] for x in ss]):.4f}")
    ax[0].set_yscale("log"); ax[0].set_xscale("log"); ax[0].set_xlabel("Frame length $T$ (ms)"); ax[0].set_ylabel("Avg. data queue (Mbit)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("Frame length $T$ (ms)"); ax[1].set_ylabel("Avg. WD power (W)")
    from matplotlib.ticker import NullFormatter
    for a in ax[:2]:
        a.set_xticks([50, 100, 200]); a.set_xticklabels(["50", "100", "200"]); a.xaxis.set_minor_formatter(NullFormatter())
    fig.subplots_adjust(wspace=0.42)
    tradeoff(ax[2], json.load(open(os.path.join(R, "E4.json")))["decision_ms"])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 0.97), frameon=False)
    label_panels(ax); fig.savefig(os.path.join(F, "deadline.pdf")); plt.close(fig)
    open(os.path.join(F, "num_E5.txt"), "w").write("\n".join(lines))


def fig_E10():
    p = os.path.join(R, "E10.pkl")
    if not os.path.exists(p):
        return
    rows = pickle.load(open(p, "rb"))
    nr = lambda k: np.array([(x[k] - x["f0"]) / (x["opt"] - x["f0"]) if x["opt"] > x["f0"] else 1.0 for x in rows])
    keys = [("random", "Random (K=8)"), ("qubo_argmax_true", "QUBO argmax"), ("qaoa_p1", "QAOA p=1"),
            ("qaoa_p2", "QAOA p=2"), ("qaoa_p3", "QAOA p=3"), ("noisy_0.001", "p=2, depol. 0.1%"),
            ("noisy_0.005", "p=2, depol. 0.5%"), ("noisy_0.01", "p=2, depol. 1%")]
    fig, ax = plt.subplots(figsize=(3.5, 1.35))
    m = [nr(k).mean() for k, _ in keys]; h = [np.mean(nr(k) > 1 - 1e-9) for k, _ in keys]
    ax.bar(range(len(keys)), m, 0.6, color=PAL[0], label="normalized DPP gain")
    ax.bar(np.arange(len(keys)) + 0.0, h, 0.25, color=PAL[1], label="hit rate of optimum")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l in keys], rotation=35, ha="right")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(os.path.join(F, "qaoa.pdf")); plt.close(fig)
    out = {k: dict(gain=float(nr(k).mean()), hit=float(np.mean(nr(k) > 1 - 1e-9))) for k, _ in keys}
    out["popt"] = {pl: float(np.mean([x[f"popt_p{pl}"] for x in rows])) for pl in [1, 2, 3]}
    out["n"] = len(rows)
    open(os.path.join(F, "num_E10.json"), "w").write(json.dumps(out, indent=1))


def fig_E11():
    rows = load("E11")
    if rows is None:
        return
    S = [("PALM", "PALM (on-demand)", PAL[0], "o", "-"), ("PALM-PS", "PALM-PS (proactive)", PAL[1], "s", "--"),
         ("KeepAlive", "Keep-alive", PAL[2], "^", "-.")]
    sel = lambda part, sch, **kw: [s for tg, s, _ in rows if tg.get("part") == part and tg["scheme"] == sch
                                   and all(tg.get(k) == v for k, v in kw.items())]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0))
    rks = sorted(set(tg["rekey"] for tg, s, _ in rows if tg.get("part") == "a"))
    q0 = np.array([np.mean([x["Q"] for x in sel("a", "PALM", rekey=r)]) for r in rks])
    for sch, lab, c, mk, ls in S:
        q = np.array([np.mean([x["Q"] for x in sel("a", sch, rekey=r)]) for r in rks])
        ax[0].plot(rks, q / q0, color=c, marker=mk, ms=3, ls=ls, label=lab)
        io = [np.mean([np.mean(x["perdev_off"][:6]) for x in sel("a", sch, rekey=r)]) * 100 for r in rks]
        ax[1].plot(rks, io, color=c, marker=mk, ms=3, ls=ls, label=lab)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("Session lifetime $L^{\\mathrm{rk}}$ (frames)")
    ax[0].set_ylabel("Backlog relative to PALM")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    tk = [0.5, 1, 2, 4, 8]
    ax[0].yaxis.set_major_locator(FixedLocator(tk)); ax[0].yaxis.set_major_formatter(FixedFormatter([str(t) for t in tk]))
    ax[0].yaxis.set_minor_locator(NullLocator()); ax[0].set_ylim(0.45, 9)
    ax[1].set_xscale("log"); ax[1].set_xlabel("Session lifetime $L^{\\mathrm{rk}}$ (frames)")
    ax[1].set_ylabel("IoT offloading ratio (%)")
    lams = sorted(set(tg["lam"] for tg, s, _ in rows if tg.get("part") == "b"))
    for sch, lab, c, mk, ls in S:
        q = [np.mean([x["Q"] for x in sel("b", sch, lam=l)]) / 1e3 for l in lams]
        ax[2].plot(lams, q, color=c, marker=mk, ms=3, ls=ls, label=lab)
    ax[2].set_yscale("log"); ax[2].set_xlabel("Arrival rate $\\lambda_i$ (Mbps), $L^{\\mathrm{rk}}=5$")
    ax[2].set_ylabel("Avg. data queue (Mbit)")
    fig.subplots_adjust(wspace=0.42)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 0.97), frameon=False)
    label_panels(ax); fig.savefig(os.path.join(F, "proactive.pdf")); plt.close(fig)
    lines = []
    for tg0 in sorted({tuple(sorted((k, v) for k, v in tg.items() if k not in ("seed", "scheme"))) for tg, s, _ in rows}, key=str):
        tg0 = dict(tg0)
        for sch in ["PALM", "PALM-PS", "KeepAlive"]:
            ss = [s for tg, s, _ in rows if tg["scheme"] == sch and all(tg.get(k) == v for k, v in tg0.items())]
            if not ss:
                continue
            m = lambda k: np.mean([x[k] for x in ss])
            pdv = np.mean([x["perdev_P"] for x in ss], 0)
            lines.append(f"{tg0} {sch}: Q={m('Q')/1e3:.4f} slope={m('Qslope'):.3f} E={m('E'):.5f} off={m('off')*100:.2f} "
                         f"iotoff={np.mean([np.mean(x['perdev_off'][:6]) for x in ss])*100:.2f} hs={m('hs'):.3f} "
                         f"hpro={np.mean([x.get('hpro', 0) for x in ss]):.3f} maxIoT={np.max(pdv[:6]/0.08):.4f} "
                         f"dt={m('dt')*1e3:.2f} Ees={m('Ees'):.3f}")
    open(os.path.join(F, "num_E11.txt"), "w").write("\n".join(lines))


def tab_E12():
    rows = load("E12")
    if rows is None:
        return
    S = ["PALM-PS", "PALM-G-PS", "PALM", "PALM-G", "LyDROO-SCA", "LyDROO-Greedy", "LyDROO-TDMA",
         "PQC-agnostic", "KeepAlive", "LyDiff", "CEA", "DDPG", "LyDDPG-Att"]
    NAME = dict(LABEL, **{"PALM-PS": "\\textbf{PALM-PS}", "LyDROO-SCA": "NOMA-SCA",
                          "LyDROO-Greedy": "NOMA-Greedy [earlier]", "DDPG": "DDPG", "PALM-G": "PALM-G"})
    lines = []
    for sch in S:
        ss = [x for tg, x, _ in rows if tg["scheme"] == sch]
        if not ss:
            continue
        q = np.mean([x["Q"] for x in ss]) / 1e3; sl = np.mean([x["Qslope"] for x in ss])
        cells = [NAME.get(sch, sch), "div." if (sl > 0.5 or q > 50) else f"{q:.3f}",
                 f"{np.mean([x['E'] for x in ss]):.4f}",
                 f"{np.mean([np.mean(x['perdev_off'][:6]) for x in ss])*100:.2f}",
                 f"{np.mean([x['hs'] for x in ss]):.2f}"]
        lines.append(" & ".join(cells) + " \\\\")
    open(os.path.join(F, "tab_E12.tex"), "w").write("\n".join(lines))


if __name__ == "__main__":
    for f in [fig_E1, fig_E2, fig_E3, fig_E4, fig_E5, fig_E6, fig_E6b, fig_E7, fig_E8, fig_E10, fig_E11, tab_E12]:
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            import traceback; traceback.print_exc()
