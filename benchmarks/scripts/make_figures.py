"""Figures for the README from the v2 runs. No model dependencies.

    python benchmarks/scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
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
    fig.text(0.01, -0.04, "Blue: Open Alternative to Jev (packed). Gray: baselines. Qwen3.6-27B, bitsandbytes 8-bit, one H200 MIG 2g.35gb slice, "
             "Hugging Face Transformers. Throughput is GPU forward time only.", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)


panel_set(race, ["A", "B4", "C4"], ["A\none at a time", "B\nbatch of 4", "Open Alternative\nto Jev\n(state once)"],
          "RACE-H, 250 passages x 4 questions (n = 1000): shared state", "race.png", (85, 97))
panel_set(mmlu, ["A", "B3", "C3", "C6", "C12"], ["A\none at\na time", "B\nbatch\nof 3", "Open Alt.\nto Jev\n3 packed", "Open Alt.\nto Jev\n6 packed", "Open Alt.\nto Jev\n12 packed"],
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

# typed-decisions: accuracy and calibration of every model we ran, against the published Jev row and the ceiling.
import glob
td = {}
for d in sorted(glob.glob(str(ROOT / "results/td_*"))):
    f = Path(d) / "summary.json"
    if f.exists():
        td[Path(d).name] = json.loads(f.read_text())
def pick(prefix):
    for k, v in td.items():
        if k.startswith(prefix):
            return v
    return None
entries = [  # (label, summary key prefix, color)
    ("Qwen3\n0.6B", "td_qwen3-0_6b_packed_3", GRAY), ("Qwen3\n1.7B", "td_qwen3-1_7b_packed_3", GRAY),
    ("Qwen3.5\n2B", "td_qwen3_5-2b_packed_3", GRAY), ("Qwen3.5\n4B", "td_qwen4b_packed_3", GRAY),
    ("Qwen3.6-27B\nzero-shot\n(this library)", "td_qwen27b_packed_3", BLUE),
    ("Laya\nbase", "td_laya_base", ORANGE), ("Laya\nfine-tuned\non this bench", "td_laya_ft_3", ORANGE),
]
labels, acc, ece, colors = [], [], [], []
for label, key, color in entries:
    s = pick(key)
    if s:
        labels.append(label); acc.append(100 * s["overall"]["acc"]); ece.append(100 * s["overall"]["ece"]); colors.append(color)
labels.append("Jev 1.13.0\n(via API, by the\nbenchmark authors)"); acc.append(72.7); ece.append(14.4); colors.append(INK2)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
x = range(len(labels))
for ax, vals, title, fmt in [(axes[0], acc, "Accuracy vs teacher gold, %", "{:.1f}"), (axes[1], ece, "Expected calibration error, % (lower is better)", "{:.1f}")]:
    b = ax.bar(x, vals, color=colors, width=0.66)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + max(vals) * 0.01, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(list(x), labels, fontsize=7.8)
    ax.tick_params(axis="x", length=0)
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.22)
axes[0].axhline(73.5, color=INK2, linestyle="--", linewidth=1.2)
axes[0].text(3.5, 88, "dashed: teacher self-agreement ceiling, 73.5 %.\nAbove it a model is learning the teacher's quirks.", fontsize=7.5, color=INK2, ha="center", va="top")
fig.suptitle("LocalLLaMA/typed-decisions: 400 cases x 5 typed questions, one shared state each (n = 2,000)", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
fig.text(0.01, -0.05, "Blue: Open Alternative to Jev on a stock model, no training. Orange: Laya, run here with the same scorer (matches its published 0.766). "
         "Gray: smaller stock models. Jev row: measured by the benchmark authors through TypeSafe's API on 2026-09-18 (dataset card).", fontsize=7.5, color=INK2)
fig.tight_layout()
fig.savefig(OUT / "typed_decisions.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("FIGURES_OK", sorted(p.name for p in OUT.iterdir()))

# Quality vs latency scatter plots: one per benchmark, one point per model. Log x axis.
def scatter(points, title, fname, xlabel, note, ylim, hline=None):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.set_xscale("log")
    if hline:
        ax.axhline(hline[0], color=INK2, linestyle="--", linewidth=1.1)
        ax.text(0.01, hline[0] + 0.7, hline[1], transform=ax.get_yaxis_transform(), ha="left", fontsize=7.5, color=INK2)
    for label, ms, acc, color, dx, dy in points:
        ax.scatter([ms], [acc], s=70, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(label, (ms, acc), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=color, ha="left" if dx >= 0 else "right", va="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy, %")
    ax.set_ylim(*ylim)
    xs = [pt[1] for pt in points]
    ax.set_xlim(min(xs) / 1.6, max(xs) * 2.4)
    ticks = [tk for tk in [5, 10, 20, 50, 100, 200, 500, 1000, 2000] if min(xs) / 1.6 <= tk <= max(xs) * 2.4]
    ax.set_xticks(ticks, [str(tk) for tk in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_title(title)
    fig.text(0.01, -0.02, note, fontsize=7.5, color=INK2, wrap=True)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)

td_points = []
for label, key, color, dx, dy in [("Qwen3-0.6B", "td_qwen3-0_6b_packed_3", GRAY, 8, 0), ("Qwen3-1.7B", "td_qwen3-1_7b_packed_3", GRAY, 8, 0),
                                  ("Qwen3.5-2B", "td_qwen3_5-2b_packed_3", GRAY, 8, 0), ("Qwen3.5-4B", "td_qwen4b_packed_3", GRAY, 8, 0),
                                  ("Qwen3.6-27B zero-shot (this library)", "td_qwen27b_packed_3", BLUE, -10, 12),
                                  ("Laya base", "td_laya_base", ORANGE, 8, 0), ("Laya fine-tuned on this benchmark", "td_laya_ft_3", ORANGE, 8, 0)]:
    s = pick(key)
    if s:
        td_points.append((label, s["ms_per_case_p50"], 100 * s["overall"]["acc"], color, dx, dy))
td_points.append(("Jev 1.13.0 (API, measured by benchmark authors)", 710, 72.7, INK2, 8, -10))
scatter(td_points, "typed-decisions: accuracy vs latency per case (5 decisions)", "typed_decisions_scatter.png",
        "ms per case, p50, log scale", "Qwen and Laya measured here on one H200 MIG slice (Qwen3.6-27B in 8-bit with fallback kernels; Laya is a 421M encoder). "
        "Jev: accuracy and p50 latency measured by the benchmark authors through TypeSafe's API (2026-09-18), so its latency includes the network round-trip.", (20, 85), hline=(73.5, "teacher self-agreement ceiling 73.5 %"))

race_points = []
race27 = race["C4"]
race_points.append(("Qwen3.6-27B 8-bit (this library)", 1000 / race27["questions_per_second"], 100 * race27["accuracy"], BLUE, -8, 8))
for label, prefix, color in [("Qwen3-0.6B", "race_qwen3-0_6b_", GRAY), ("Qwen3-1.7B", "race_qwen3-1_7b_", GRAY),
                             ("Qwen3.5-2B", "race_qwen3_5-2b_", GRAY), ("Qwen3.5-4B", "race_qwen3_5-4b_", GRAY)]:
    for d in sorted(glob.glob(str(ROOT / f"results/{prefix}*"))):
        f = Path(d) / "summary.json"
        if f.exists():
            s = json.loads(f.read_text())["C4"]
            race_points.append((label, 1000 / s["questions_per_second"], 100 * s["accuracy"], color, 8, 0))
            break
for label, prefix in [("Laya base", "race_laya_base_"), ("Laya fine-tuned (typed-decisions ckpt)", "race_laya_ft_")]:
    for d in sorted(glob.glob(str(ROOT / f"results/{prefix}*"))):
        f = Path(d) / "summary.json"
        if f.exists():
            s = json.loads(f.read_text())
            race_points.append((label, s["ms_per_passage_p50"] / 4, 100 * s["accuracy"], ORANGE, 8, 0))
            break
if len(race_points) > 1:
    scatter(race_points, "RACE-H: accuracy vs latency per question (packed, 4 per passage)", "race_scatter.png",
            "ms per question, log scale", "All measured here on one H200 MIG 2g.35gb slice. Qwen: GPU forward time per question in packed mode (benchmark_v2, C4). "
            "Laya: wall-clock per passage / 4, including tokenization; passages beyond its context are truncated by the model.", (20, 100))
print("SCATTER_OK", len(td_points), len(race_points))
