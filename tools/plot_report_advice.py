"""Publication-sized English figures from the previously audited advice tables.

Run with /tmp/llm_advice_viz_venv/bin/python tools/plot_report_advice.py.
No model calls, no simulations, and no changes to source reports.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALIAS = Path(str(ROOT).lower())
if ALIAS.is_dir() and ALIAS.samefile(ROOT):
    ROOT = ALIAS
SOURCE = ROOT / "reports/advice_deepseek_gpt_qwen_20261029"
OUT = ROOT / "reports/Preliminary_1029/figures"
MODELS = ("DeepSeek", "GPT", "Qwen")
COLORS = {"DeepSeek": "#D27A2A", "GPT": "#287AB5", "Qwen": "#009E91"}
INK = "#17394F"
MUTED = "#5D7280"
REPEATS = {"DeepSeek": 1, "GPT": 2, "Qwen": 2}
CMAP = LinearSegmentedColormap.from_list("advice_frequency", ["#F3F6F8", "#A1D7CB", "#17958C", INK])


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 8.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
        "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False,
    })


def read_data():
    frames = {m: pd.read_csv(SOURCE / "data" / f"{m}_clean_answers.csv") for m in MODELS}
    for m, d in frames.items():
        assert len(d) == 512 * REPEATS[m]
        assert d.grid_key.nunique() == 512
        assert d.groupby("grid_key").size().eq(REPEATS[m]).all()
        assert d.recommendation.isin(["C", "D"]).all()
        assert (d.is_c == d.recommendation.eq("C").astype(int)).all()
    return frames


def transition_data(frames):
    rows = []
    for m, d in frames.items():
        for current in ("C", "D"):
            sub = d.loc[d.center.eq(current)]
            for advice in ("C", "D"):
                count = int(sub.recommendation.eq(advice).sum())
                rows.append(dict(model=m, current_action=current, advice=advice,
                                 count=count, replies=len(sub), configurations=sub.grid_key.nunique(),
                                 replies_per_configuration=REPEATS[m], fraction=count / len(sub)))
    return pd.DataFrame(rows)


def plot_transitions(data):
    fig = plt.figure(figsize=(7.1, 2.8))
    gs = fig.add_gridspec(1, 3, left=.082, right=.893, bottom=.25, top=.83, wspace=.30)
    for j, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, j])
        sub = data.loc[data.model.eq(m)].set_index(["current_action", "advice"])
        vals = np.array([[sub.loc[(current, advice), "fraction"] for advice in ("C", "D")] for current in ("C", "D")])
        im = ax.imshow(vals, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
        for i, current in enumerate(("C", "D")):
            for k, advice in enumerate(("C", "D")):
                row = sub.loc[(current, advice)]
                color = "white" if row.fraction >= .66 else INK
                ax.text(k, i - .10, f"{row.fraction:.2%}", ha="center", va="center", weight="bold", fontsize=12, color=color)
                ax.text(k, i + .20, f"{int(row['count'])}/{int(row.replies)}", ha="center", va="center", fontsize=8.2, color=color)
        ax.set_xticks([0, 1], ["C", "D"])
        ax.set_yticks([0, 1], ["C", "D"])
        ax.set_xlabel("Recommended action", labelpad=3)
        if j == 0:
            ax.set_ylabel("Current action", labelpad=5)
        ax.set_title(f"{m} (n = {REPEATS[m]})", color=COLORS[m], pad=9)
        ax.set_xticks([.5], minor=True)
        ax.set_yticks([.5], minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.922, .25, .012, .58])
    cb = fig.colorbar(im, cax=cax, ticks=[0, .5, 1], format=PercentFormatter(1, decimals=0))
    cb.ax.tick_params(labelsize=7.5, length=2, pad=3)
    cb.outline.set_visible(False)
    fig.text(.082, .045, "n = replies per configuration. Rows sum to 100%; counts show replies / row total.", fontsize=8, color=MUTED)
    return fig


def neighbor_data(frames):
    rows = []
    for m, d in frames.items():
        for (current, k), sub in d.groupby(["center", "cardinal_c"]):
            rows.append(dict(model=m, current_action=current, cooperating_direct_neighbors=int(k),
                             C_replies=int(sub.is_c.sum()), replies=len(sub),
                             configurations=sub.grid_key.nunique(), replies_per_configuration=REPEATS[m],
                             fraction_C=float(sub.is_c.mean())))
    return pd.DataFrame(rows)


def plot_neighbors(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3), sharey=True)
    fig.subplots_adjust(left=.108, right=.985, bottom=.245, top=.765, wspace=.20)
    for i, (ax, current) in enumerate(zip(axes, ("C", "D"))):
        for m, marker in zip(MODELS, ("o", "^", "s")):
            sub = data.loc[data.model.eq(m) & data.current_action.eq(current)].sort_values("cooperating_direct_neighbors")
            ax.plot(sub.cooperating_direct_neighbors, sub.fraction_C, color=COLORS[m],
                    marker=marker, markersize=4.8, markeredgewidth=.8, markeredgecolor="white",
                    linewidth=1.7, label=f"{m} (n = {REPEATS[m]})")
        title = "(a) Currently cooperating (C)" if current == "C" else "(b) Currently defecting (D)"
        ax.set(xlim=(-.12, 4.12), ylim=(-.04, 1.05), xticks=range(5), yticks=[0, .25, .5, .75, 1],
               xlabel="Cooperating direct neighbors", title=title)
        ax.set_title(title, pad=8)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E3EAEE", linewidth=.7)
        ax.spines["left"].set_color("#CAD6DE")
        ax.spines["bottom"].set_color("#CAD6DE")
        ax.tick_params(length=3, color="#B1C3CC")
    axes[0].set_ylabel("Advice recommending C", labelpad=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .995), ncol=3,
               frameon=False, fontsize=8.5, handlelength=2.2, columnspacing=1.9)
    fig.text(.09, .035, "Configurations per panel at 0–4 cooperating neighbors: 16, 64, 96, 64, 16.", fontsize=8, color=MUTED)
    return fig


def example_data(frames):
    rows = []
    for m, d in frames.items():
        sub = d.loc[d.grid_key.eq("CCCCDCCCC")].sort_values("repetition")
        assert len(sub) == REPEATS[m]
        for _, row in sub.iterrows():
            rows.append(dict(model=m, grid_key=row.grid_key, repetition=int(row.repetition), recommendation=row.recommendation))
    return pd.DataFrame(rows)


def plot_example(data):
    fig = plt.figure(figsize=(7.1, 2.1))
    ax = fig.add_axes([.06, .10, .27, .82])
    ax.set(xlim=(-.1, 3.1), ylim=(-.1, 3.1), aspect="equal")
    ax.axis("off")
    for row in range(3):
        for col in range(3):
            center = row == col == 1
            ax.add_patch(Rectangle((col, 2 - row), 1, 1, facecolor="#F9EBDE" if center else "#D9EFEB",
                                   edgecolor="#D27A2A" if center else "white", linewidth=2.3 if center else 2))
            ax.text(col + .5, 2.5 - row, "D" if center else "C", ha="center", va="center", fontsize=18, color=INK, weight="bold" if center else "normal")
    ax.add_patch(Rectangle((1, 1), 1, 1, facecolor="none", edgecolor="#D27A2A", linewidth=2.3))
    fig.text(.38, .89, "Same configuration, different recorded advice", fontsize=10.5, weight="bold")
    for y, m in zip([.68, .47, .26], MODELS):
        answers = " / ".join(data.loc[data.model.eq(m), "recommendation"])
        fig.text(.38, y, m, fontsize=11, weight="bold", color=COLORS[m])
        fig.text(.64, y, answers, fontsize=12, weight="bold")
        fig.text(.79, y, f"n = {REPEATS[m]}", fontsize=9, color=MUTED)
    fig.text(.38, .07, "Center: current D. All eight surrounding positions: C.", fontsize=8, color=MUTED)
    return fig


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    frames = read_data()
    transitions = transition_data(frames)
    neighbors = neighbor_data(frames)
    example = example_data(frames)
    # The derived quantities must agree exactly with the published audit tables.
    old = pd.read_csv(SOURCE / "tables/总体与本人条件统计.csv").set_index("model")
    for m in MODELS:
        for current, column in [("C", "C_given_current_C"), ("D", "C_given_current_D")]:
            val = transitions.loc[transitions.model.eq(m) & transitions.current_action.eq(current) & transitions.advice.eq("C"), "fraction"].iloc[0]
            assert np.isclose(val, old.loc[m, column])
    old_neighbors = pd.read_csv(SOURCE / "tables/四直接邻居条件统计.csv")
    comparison = neighbors.merge(old_neighbors, left_on=["model", "current_action", "cooperating_direct_neighbors"], right_on=["model", "center", "cardinal_c"], validate="one_to_one", suffixes=("", "_old"))
    assert len(comparison) == 30
    assert np.allclose(comparison.fraction_C, comparison.C_fraction)
    assert comparison.C_replies.eq(comparison.C).all()
    assert comparison.replies.eq(comparison.replies_old).all()
    transitions.to_csv(OUT / "advice_transitions_data.csv", index=False)
    neighbors.to_csv(OUT / "advice_neighbors_data.csv", index=False)
    example.to_csv(OUT / "advice_example_data.csv", index=False)
    save(plot_transitions(transitions), "advice_transitions")
    save(plot_neighbors(neighbors), "advice_neighbors")
    save(plot_example(example), "advice_example")
    notes = {
        "advice_transitions": "Conditional distribution of recorded advice, stratified by the focal player's current action. Each model covers all 512 binary 3 × 3 configurations: 256 with a currently cooperative focal player and 256 with a currently defecting focal player. DeepSeek has one reply per configuration; GPT and Qwen have two. Each row sums to 100%; annotations give percentages and reply counts divided by the corresponding row total. These empirical response frequencies are not transition rates observed in simulation.",
        "advice_neighbors": "Recorded cooperation recommendations by the number of cooperative direct neighbors, stratified by the focal player's current action. Each panel includes 16, 64, 96, 64, and 16 configurations at 0, 1, 2, 3, and 4 cooperative direct neighbors, respectively. Curves aggregate all corner arrangements. Replies per configuration differ across models (DeepSeek: 1; GPT/Qwen: 2); the points are empirical frequencies without uncertainty intervals, and connecting lines serve only as visual guides.",
        "advice_example": "A current defector surrounded by eight cooperative positions (CCC/CDC/CCC). Recorded advice is D for DeepSeek, C/D for GPT, and C/C for Qwen. This exact-configuration example is distinct from the neighbor-count panel, which pools all corner arrangements.",
        "limits": "The three batches used identical prompts for corresponding configurations, but generation settings were not fully controlled. Prompts specified 1 < b < 2, not a numerical temptation value. Frequency summaries weight configurations equally, not by simulation visitation frequency. One DeepSeek reply cannot establish deterministic model behavior. No DeepSeek SPD simulation result is included.",
        "source": str(SOURCE.relative_to(ROOT)),
        "reproduce": "/tmp/llm_advice_viz_venv/bin/python tools/plot_report_advice.py",
        "checks": "512 unique configurations per model; 512/1024/1024 replies; derived transition and neighbor counts checked against existing audited summaries."
    }
    (OUT / "advice_figure_notes.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved three English advice figures and plotted data in {OUT}")


if __name__ == "__main__":
    main()
