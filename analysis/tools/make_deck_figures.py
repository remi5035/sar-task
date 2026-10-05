"""English figures for the presentation deck.

1. views_1000245_en.png  : what the drone sees on mountain seed 1000245 (depth + RGB at three moments)
2. fix_mountain_1000245_en.png : champion vs agent_final on the same seed (top view + approach over time)

Inputs (local, not versioned): fixes/views/*.npz from capture_views.py, and the traj/ folders of
data/champion_full and data/v6_final from collect_rollouts.py. Output: results/figures/.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "diagnostic"))
import sar_diag as sd  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
VIEWS = ROOT / "fixes" / "views"
SUCC_C, FAIL_C = "#1baf7a", "#e34948"
SEED, GROUP = 1000245, "type3_mountain"


def views_figure():
    tags = [("start", "take-off"), ("victim", "victim largest in view"), ("above", "at 2 m horizontally")]
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.0), gridspec_kw=dict(hspace=0.16, wspace=0.10))
    for j, (tag, lab) in enumerate(tags):
        s = np.load(VIEWS / f"{SEED}_{tag}.npz")
        im = ax[0, j].imshow(s["depth"] * 29.5 + 0.5, cmap="magma_r", vmin=0.5, vmax=30)
        ax[1, j].imshow(np.clip(s["rgb"], 0, 1))
        if int(s["vis"]) == 1:
            x, y = (float(s["u"]) + 1) * 128, (1 - float(s["v"])) * 128
            for a in ax[:, j]:
                a.add_patch(plt.Circle((x, y), max(8, float(s["px"]) * 0.7), fill=False, color=SUCC_C, lw=2))
            zx, zy = int(np.clip(x, 24, 232)), int(np.clip(y, 24, 232))
            ins = ax[1, j].inset_axes([0.02, 0.02, 0.36, 0.36])
            ins.imshow(np.clip(s["rgb"], 0, 1)[zy - 24:zy + 24, zx - 24:zx + 24], interpolation="nearest")
            ins.set_xticks([]); ins.set_yticks([]); ins.set_title("zoom ×5", fontsize=9, color=SUCC_C)
            for sp in ins.spines.values():
                sp.set_edgecolor(SUCC_C); sp.set_linewidth(2)
        ax[0, j].set_title(f"{lab}, t = {float(s['t']):.1f} s\n{float(s['hdist']):.1f} m away, "
                           f"{float(s['h_above']):+.1f} m above", fontsize=11)
        for a in ax[:, j]:
            a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel("depth (0.5–30 m)", fontsize=11)
    ax[1, 0].set_ylabel("RGB (on request)", fontsize=11)
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.45, anchor=(0, 0.95), label="depth: distance (m)")
    fig.suptitle(f"Mountain seed {SEED}: what the drone sees (green circle = victim, ground truth)", fontsize=13)
    fig.savefig(OUT / f"views_{SEED}_en.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def fix_figure():
    data = ROOT / "data"
    runs = []
    for name, run, c in [("champion", data / "champion_full", FAIL_C), ("agent_final", data / "v6_final", SUCC_C)]:
        r = sd.load_episodes(run).query("seed == @SEED").iloc[0]
        runs.append((name, sd.load_traj(run, GROUP, SEED), r, c))
    ra, ta = runs[0][2], runs[0][1]
    fig = plt.figure(figsize=(17, 5.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.9])
    a = fig.add_subplot(gs[:, 0])
    a.pcolormesh(ta["hm_x"], ta["hm_y"], ta["hm_z"], cmap="Greys", shading="auto", alpha=0.7)
    for name, tr, r, c in runs:
        a.plot(tr["x"], tr["y"], color=c, lw=1.8, label=f"{name}: score {r['score']:.2f}")
        a.plot(tr["x"][-1], tr["y"][-1], "o", color=c, ms=6)
    a.plot(ra["start_x"], ra["start_y"], "s", color="#2b2b29", ms=8, label="start")
    a.plot(ra["victim_x"], ra["victim_y"], "*", color="#eda100", ms=18, mec="#2b2b29", label="victim")
    xs = np.r_[runs[0][1]["x"], runs[1][1]["x"]]; ys = np.r_[runs[0][1]["y"], runs[1][1]["y"]]
    a.set_xlim(xs.min() - 3, xs.max() + 3); a.set_ylim(ys.min() - 3, ys.max() + 3)
    a.set_aspect("equal"); a.legend(fontsize=10, loc="best")
    a.set_title("Top view (m)", fontsize=12)
    ev = lambda r, k: next((e["t"] for e in r["events"] if e["kind"] == k), None)
    t_unl = ev(ra, "rgb_unlatch")
    spec = [("hdist", "horizontal\ndistance (m)", (0, 2), (0, 8)),
            ("h_above", "height above\nvictim (m)", (2, 4), (0, 10)),
            ("vz_cmd", "commanded vertical\nspeed (m/s)", None, None)]
    for k, (col, lab, band, ylim) in enumerate(spec):
        ax = fig.add_subplot(gs[k, 1])
        for name, tr, r, c in runs:
            y = tr["a_dz"] * tr["a_speed"] * 3 if col == "vz_cmd" else tr[col]
            ax.plot(tr["t"], y, color=c, lw=1.6, label=name)
            if r["t_confirm"] == r["t_confirm"] and r["t_confirm"] is not None:
                ax.axvline(r["t_confirm"], color=c, ls=":", lw=1.2)
        if band:
            ax.axhspan(*band, color=SUCC_C, alpha=0.12, label="confirmation window")
        for _, _, r, c in runs:
            if ev(r, "rgb_latch"):
                ax.axvline(ev(r, "rgb_latch"), color=c, lw=1.2, ls="--")
        if t_unl:
            ax.axvline(t_unl, color=FAIL_C, lw=1.6, ls="-.")
        ax.set_xlim(10, 40)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel(lab, fontsize=10)
        if k == 0:
            ax.legend(fontsize=9, loc="upper right", ncol=3)
            ax.set_title("dashed: RGB lock · dash-dot: champion gives up the target · dotted: confirmation", fontsize=11)
        if k < 2:
            ax.set_xticklabels([])
    ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / f"fix_mountain_{SEED}_en.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    for name, _, r, _ in runs:
        print(name, r["outcome"], round(r["score"], 3), "confirm", r["t_confirm"], "latch", ev(r, "rgb_latch"))


if __name__ == "__main__":
    views_figure()
    fix_figure()
    print("written to", OUT)
