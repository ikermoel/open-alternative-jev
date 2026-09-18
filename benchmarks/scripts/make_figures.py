"""Figures for the README from the v2 runs. No model dependencies.

    python benchmarks/scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# Palette: emphasis form (one hue for the packed modes, gray for baselines); categorical slots for series.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # categorical slots 1-4, fixed order
GRAY, INK, INK2, GRID = "#9a9891", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "ytick.left": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
})

race = json.loads((ROOT / "results/v2_race_32679039/summary.json").read_text())
mmlu = json.loads((ROOT / "results/v2_mmlu_32679038/summary.json").read_text())
race_an = json.loads((ROOT / "results/v2_race_32679039/analysis.json").read_text())
mmlu_an = json.loads((ROOT / "results/v2_mmlu_32679038/analysis.json").read_text())


def bars(ax, labels, values, colors, fmt, ylim=None, ci=None):
    x = range(len(labels))
    b = ax.bar(x, values, color=colors, width=0.62)
    tops = list(values)
    if ci is not None:
        for i, (lo, hi) in enumerate(ci):
            if lo is not None:
                ax.plot([i, i], [lo, hi], color=INK2, linewidth=1.2, solid_capstyle="butt")
                ax.plot([i - 0.08, i + 0.08], [hi, hi], color=INK2, linewidth=1.2)
                ax.plot([i - 0.08, i + 0.08], [lo, lo], color=INK2, linewidth=1.2)
                tops[i] = max(tops[i], hi)
    span = (ylim[1] - ylim[0]) if ylim else max(values)
    for rect, v, top in zip(b, values, tops):
        ax.text(rect.get_x() + rect.get_width() / 2, top + 0.015 * span, fmt(v), ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks(list(x), labels, fontsize=8.5)
    ax.set_ylim(*ylim) if ylim else ax.set_ylim(0, max(tops) * 1.15)
    ax.tick_params(axis="x", length=0)


def panel_set(summary, modes, names, title, fname, acc_ylim):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    colors = [BLUE if m.startswith("C") else GRAY for m in modes]
    acc = [100 * summary[m]["accuracy"] for m in modes]
    ci = []
    for m in modes:
        if "vs_A" in summary[m]:
            lo, hi = summary[m]["vs_A"]["cluster_bootstrap_95_ci"]
            ci.append((100 * (summary["A"]["accuracy"] + lo), 100 * (summary["A"]["accuracy"] + hi)))
        else:
            ci.append((None, None))
    bars(axes[0], names, acc, colors, lambda v: f"{v:.1f}", acc_ylim, ci)
    axes[0].set_title("Accuracy, % (95 % CI vs A)")
    bars(axes[1], names, [summary[m]["questions_per_second"] for m in modes], colors, lambda v: f"{v:.2f}")
    axes[1].set_title("Questions per second")
    bars(axes[2], names, [summary[m]["padded_tokens"] / 1000 for m in modes], colors, lambda v: f"{v:.0f}k")
    axes[2].set_title("Tokens processed")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, -0.04, "Blue: packed (this library). Gray: baselines. Qwen3.6-27B, bitsandbytes 8-bit, one H200 MIG 2g.35gb slice, "
             "Hugging Face Transformers. Throughput is GPU forward time only.", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)


panel_set(race, ["A", "B4", "C4"], ["A\none at a time", "B\nbatch of 4", "C\npacked,\nstate once"],
          "RACE-H, 250 passages x 4 questions (n = 1000): shared state", "race.png", (85, 97))
panel_set(mmlu, ["A", "B3", "C3", "C6", "C12"], ["A\none at\na time", "B\nbatch\nof 3", "C\n3 packed", "C\n6 packed", "C\n12 packed"],
          "MMLU, 1200 questions, no shared state", "mmlu.png", (76, 90))

# Interference: answers that change vs A, by position in the packed sequence, against the numerical noise floor.
fig, ax = plt.subplots(figsize=(7.2, 3.6))
noise = 100 * mmlu["A_pad"]["vs_A"]["answer_disagreement"]
ax.axhline(noise, color=GRAY, linestyle="--", linewidth=1.5)
ax.text(11.9, noise + 0.3, f"noise floor: same prefix, padded shape only ({noise:.1f} %)", ha="right", fontsize=8.5, color=INK2)
series = [("C3", "MMLU, 3 packed", BLUE, mmlu_an), ("C6", "MMLU, 6 packed", ORANGE, mmlu_an), ("C12", "MMLU, 12 packed", AQUA, mmlu_an),
          ("C4", "RACE-H, 4 packed", YELLOW, race_an)]
for mode, label, color, an in series:
    pts = sorted((int(p), 100 * v["flips_vs_A"] / v["n"]) for p, v in an["flips_by_position"][mode].items())
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor="white", markeredgewidth=1.5, label=label)
    if mode in ("C6", "C12"):  # direct labels only where the line end is free; the legend covers the rest
        ax.text(xs[-1] + 0.15, ys[-1], label, fontsize=8.5, color=color, va="center")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_xlabel("Position of the question in the packed sequence")
ax.set_ylabel("Answers changed vs one-at-a-time, %")
ax.set_xlim(-0.3, 14.5)
ax.set_ylim(0, 14)
ax.set_xticks(range(0, 12))
ax.set_title("Interference between packed questions (accuracy unchanged, individual answers move)")
fig.tight_layout()
fig.savefig(OUT / "interference.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# Calibration: ECE raw vs after cross-fitted temperature scaling.
fig, ax = plt.subplots(figsize=(7.2, 3.2))
rows = [("MMLU A", mmlu_an), ("MMLU C12", mmlu_an), ("RACE-H A", race_an), ("RACE-H C4", race_an)]
keys = ["A", "C12", "A", "C4"]
raw = [100 * an["temperature_scaling_2fold"][k]["ece_raw"] for (_, an), k in zip(rows, keys)]
scaled = [100 * an["temperature_scaling_2fold"][k]["ece_scaled"] for (_, an), k in zip(rows, keys)]
temps = [an["temperature_scaling_2fold"][k]["temperatures"] for (_, an), k in zip(rows, keys)]
x = range(len(rows))
w = 0.36
b1 = ax.bar([i - w / 2 for i in x], raw, width=w - 0.03, color=GRAY, label="raw")
b2 = ax.bar([i + w / 2 for i in x], scaled, width=w - 0.03, color=BLUE, label="temperature scaled (cross-fitted)")
for rect, v in list(zip(b1, raw)) + list(zip(b2, scaled)):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(list(x), [f"{name}\nT = {min(t):.2f}-{max(t):.2f}" for (name, _), t in zip(rows, temps)])
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Expected calibration error, %")
ax.set_title("Temperature scaling, fitted on held-out labels (T > 1 = model was over-confident)")
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "calibration.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("FIGURES_OK", sorted(p.name for p in OUT.iterdir()))
