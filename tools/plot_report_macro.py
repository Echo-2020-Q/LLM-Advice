"""English report figures from existing audited SPD caches; no simulation/API calls.

Run with: python tools/plot_report_macro.py
The input files are read only. Output contains figures, plotted values, and
checksums documenting which previously audited artifacts were used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WRITE_ROOT = Path(str(ROOT).lower())
if not (WRITE_ROOT.is_dir() and WRITE_ROOT.samefile(ROOT)):
    WRITE_ROOT = ROOT
REPORT = WRITE_ROOT / "reports/Preliminary_1029"
INPUTS = {
    "GPT": ROOT / "reports/gpt_20261027_repeats2/data_audit_gpt",
    "Qwen": ROOT / "reports/qwen_20261027_repeats2",
}
KEYS = ["b", "lambda_prob", "llm_access_fraction", "seed"]
METRIC = "tail_mean_cooperation_rate"
COLORS = {"GPT": "#287AB5", "Qwen": "#009E91", "Fermi": "#87929E"}
INK = "#173447"
MUTED = "#526574"
SEEDS = [42, 43]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_audited():
    data, provenance = {}, {}
    for model, folder in INPUTS.items():
        metadata = json.loads((folder / "data/cache_metadata.json").read_text())
        checked = {}
        for name, expected in metadata["artifact_sha256"].items():
            path = folder / name
            assert sha(path) == expected, f"Previously audited artifact changed: {path}"
            checked[str(path.relative_to(ROOT))] = expected
        df = pd.read_csv(folder / "data/validated_runs.csv")
        assert len(df) == 616 and not df.duplicated(KEYS).any()
        assert df.groupby(KEYS[:-1]).size().eq(2).all()
        assert set(df.seed) == set(SEEDS)
        keys = [tuple(k) for k in metadata["ordered_keys"]]
        with np.load(folder / "data/trajectory_cache.npz", allow_pickle=False) as z:
            trajectories = dict(zip(keys, z["cooperation"].copy()))
        assert len(trajectories) == 616
        for row in df.itertuples(index=False):
            k = (row.b, row.lambda_prob, row.llm_access_fraction, row.seed)
            series = trajectories[k]
            assert len(series) == 10001
            assert np.isclose(series[9001:10001].mean(), getattr(row, METRIC), atol=1e-12)
        provenance[model] = {
            "basis": "Previously audited cache; all four cached artifacts rechecked against stored SHA256. All 616 final-window means independently recomputed from cached full trajectories.",
            "metadata_file": str((folder / "data/cache_metadata.json").relative_to(ROOT)),
            "metadata_sha256": sha(folder / "data/cache_metadata.json"),
            "verified_artifacts": checked,
            "source_batch": metadata["batch"],
            "raw_sources_in_prior_audit": len(metadata["source_sha256"]),
        }
        data[model] = (df, trajectories)
    left = data["GPT"][0].set_index(KEYS).sort_index()
    right = data["Qwen"][0].set_index(KEYS).sort_index()
    assert left.index.equals(right.index)
    assert np.array_equal(left.control_tail_mean_cooperation_rate, right.control_tail_mean_cooperation_rate)
    joined = left[[METRIC]].rename(columns={METRIC: "gpt_tail_cooperation"}).join(
        right[[METRIC]].rename(columns={METRIC: "qwen_tail_cooperation"})).reset_index()
    joined["qwen_minus_gpt_pp"] = 100 * (joined.qwen_tail_cooperation - joined.gpt_tail_cooperation)
    means = joined.groupby(KEYS[:-1], as_index=False).agg(
        gpt_tail_cooperation=("gpt_tail_cooperation", "mean"),
        qwen_tail_cooperation=("qwen_tail_cooperation", "mean"),
        qwen_minus_gpt_pp=("qwen_minus_gpt_pp", "mean"),
        n_seeds=("seed", "size"))
    active = means[(means.lambda_prob > 0) & (means.llm_access_fraction > 0)]
    counts = {"qwen_higher": int((active.qwen_minus_gpt_pp > 0).sum()),
              "equal": int((active.qwen_minus_gpt_pp == 0).sum()),
              "gpt_higher": int((active.qwen_minus_gpt_pp < 0).sum())}
    assert counts == {"qwen_higher": 130, "equal": 80, "gpt_higher": 0}
    provenance["active_parameter_sets"] = counts
    provenance["metric"] = "Mean cooperation over steps 9001–10000, then arithmetic mean across seeds 42 and 43. Difference is Qwen minus GPT in percentage points."
    return data, joined, means, provenance


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 8.3, "axes.titlesize": 10,
        "xtick.labelsize": 7.7, "ytick.labelsize": 7.7,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#C1CCD3", "axes.linewidth": .7,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5EBEF", linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.7)


def overview(means):
    fig = plt.figure(figsize=(7.1, 4.0))
    gs = fig.add_gridspec(1, 2, left=.085, right=.985, top=.88,
                          bottom=.285, wspace=.16)
    cmap = LinearSegmentedColormap.from_list("qwen_gpt_report", ["#F6FAFB", "#9FD9D2", "#009E91", "#075951"])
    bs = sorted(means.b.unique())
    ps = [round(x / 10, 1) for x in range(1, 11)]
    for j, lam in enumerate([.5, 1.]):
        ax = fig.add_subplot(gs[0, j])
        a = means[means.lambda_prob.eq(lam)].pivot(index="b", columns="llm_access_fraction", values="qwen_minus_gpt_pp").loc[bs, ps].to_numpy()
        im = ax.imshow(a, cmap=cmap, norm=Normalize(0, 100), aspect="auto")
        for iy in range(a.shape[0]):
            for ix in range(a.shape[1]):
                v = a[iy, ix]
                label = "0" if v == 0 else ("<0.1" if v < .1 else f"{v:.1f}")
                ax.text(ix, iy, label, ha="center", va="center", fontsize=6.6,
                        color="white" if v >= 60 else INK)
        ax.set_xticks(range(10), [f"{p:g}" for p in ps])
        ax.set_yticks(range(7), [f"{b:g}" for b in bs])
        ax.set_xlabel("Advice coverage, $P$")
        if j == 0:
            ax.set_ylabel("Temptation, $b$")
        ax.set_title(f"({'ab'[j]})  Adoption probability $\\lambda={lam:g}$", loc="left", fontweight="bold", pad=10)
        ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.6)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.28, .14, .52, .037])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    cb.outline.set_visible(False)
    cb.ax.set_xlabel("Qwen minus GPT: cooperation difference (percentage points)", labelpad=4, fontsize=8)
    return fig


def detail(data):
    fig = plt.figure(figsize=(7.1, 4.0))
    gs = fig.add_gridspec(1, 2, left=.085, right=.985, top=.80,
                          bottom=.24, wspace=.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    ax = axes[0]
    for model in ["GPT", "Qwen"]:
        df = data[model][0]
        sub = df[df.b.eq(1.02) & df.lambda_prob.eq(1.)]
        a = sub.pivot(index="llm_access_fraction", columns="seed", values=METRIC)[SEEDS]
        p, y = a.index.to_numpy(), a.to_numpy() * 100
        ax.fill_between(p, y.min(1), y.max(1), color=COLORS[model], alpha=.14, lw=0)
        ax.plot(p, y.mean(1), color=COLORS[model], lw=1.7, zorder=3)
        for i, marker in enumerate(["o", "^"]):
            ax.plot(p, y[:, i], linestyle="none", color=COLORS[model], marker=marker,
                    markersize=3.5, markeredgewidth=.6, markerfacecolor="white", zorder=4)
    ax.annotate("−88.1 pp\n($P$: 0.6 → 0.7)", xy=(.7, 1.66584), xytext=(.82, 52),
                color="#A75B24", fontsize=7.5, ha="center",
                arrowprops={"arrowstyle": "-", "color": "#D27A2A", "lw": .8,
                            "connectionstyle": "angle,angleA=0,angleB=90,rad=3"})
    ax.set(xlim=(-.025, 1.025), ylim=(-3, 106), xticks=[0, .2, .4, .6, .8, 1],
           yticks=[0, 25, 50, 75, 100], xlabel="Advice coverage, $P$",
           ylabel="Final-window cooperation")
    ax.set_title("(a) Coverage response", loc="left", fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    clean(ax)

    ax = axes[1]
    # Keep all early steps, then a fixed regular display subsample. Statistics use all steps.
    take = np.unique(np.r_[np.arange(201), np.arange(210, 10001, 10)])
    trajectory_rows = []
    for model, source, lam in [("GPT", "GPT", 1.), ("Qwen", "Qwen", 1.), ("Fermi", "GPT", 0.)]:
        y_full = np.stack([data[source][1][1.02, lam, .7, seed] for seed in SEEDS]) * 100
        if model == "Fermi":
            for seed in SEEDS:
                assert np.array_equal(data["GPT"][1][1.02, 0., .7, seed], data["Qwen"][1][1.02, 0., .7, seed])
        y = y_full[:, take]
        ax.fill_between(take, y.min(0), y.max(0), color=COLORS[model], alpha=.14, lw=0)
        ax.plot(take, y.mean(0), color=COLORS[model], lw=1.4,
                linestyle="--" if model == "Fermi" else "-")
        for i, seed in enumerate(SEEDS):
            trajectory_rows.append(pd.DataFrame({"model": model, "b": 1.02,
                "lambda_prob": lam, "llm_access_fraction": .7, "seed": seed,
                "step": np.arange(10001), "cooperation_percent": y_full[i],
                "shown_in_figure": np.isin(np.arange(10001), take)}))
    ax.set_xscale("symlog", linthresh=10)
    ax.set(xlim=(0, 10000), ylim=(-3, 106), xticks=[0, 10, 100, 1000, 10000],
           yticks=[0, 25, 50, 75, 100], xlabel="Simulation step", ylabel="Cooperation")
    ax.set_title("(b) Dynamics at $P=0.7$", loc="left", fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    # Keep the terminal tick fully inside the 7.1-inch figure at report size.
    ax.get_xticklabels()[-1].set_horizontalalignment("right")
    clean(ax)
    fig.text(.085, .94, "$b=1.02$; $\\lambda=1$ for advice; Fermi uses $\\lambda=0$", fontsize=9, color=INK)
    fig.legend(handles=[Line2D([], [], color=COLORS[n], label=n, lw=1.9,
                       linestyle="--" if n == "Fermi" else "-") for n in COLORS],
               loc="upper right", bbox_to_anchor=(.99, .985), frameon=False,
               ncol=3, fontsize=8, handlelength=1.6, columnspacing=1.2)
    fig.text(.085, .12, "Lines: two-seed mean. Shading: observed seed range (not a confidence interval).", fontsize=7.4, color=MUTED)
    fig.text(.085, .065, "Panel (a): ○ seed 42; △ seed 43. Panel (b): time axis is linear to step 10, then logarithmic.", fontsize=7.15, color=MUTED)
    return fig, pd.concat(trajectory_rows, ignore_index=True)


def main():
    style()
    figures, tables = REPORT / "figures", REPORT / "data"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    data, joined, means, provenance = load_audited()
    selected = means[means.lambda_prob.isin([.5, 1.]) & means.llm_access_fraction.gt(0)]
    selected.to_csv(tables / "macro_overview_values.csv", index=False)
    joined.to_csv(tables / "macro_all_matched_runs.csv", index=False)
    coverage = pd.concat([df[df.b.eq(1.02) & df.lambda_prob.eq(1.)][KEYS + [METRIC, "final_cooperation_rate"]].assign(model=model)
                          for model, (df, _) in data.items()], ignore_index=True)
    coverage.to_csv(tables / "macro_detail_coverage.csv", index=False)
    f2, trajectories = detail(data)
    trajectories.to_csv(tables / "macro_detail_trajectories.csv", index=False)
    for name, fig in [("macro_overview", overview(means)), ("macro_detail", f2)]:
        for suffix in ["png", "pdf", "svg"]:
            fig.savefig(figures / f"{name}.{suffix}", dpi=300)
        plt.close(fig)
    (tables / "macro_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (tables / "macro_evidence.txt").write_text(
        "MACROSCOPIC RESULTS: PLOTTED METRICS AND INTERPRETATION\n\n"
        "1. Figure macro_overview shows Qwen minus GPT cooperation, in percentage points, "
        "for lambda=0.5 and 1.0. For each run, cooperation is averaged over steps 9001–10000; "
        "the two seed values are then averaged. Across all 210 active-advice settings "
        "(including lambda=0.1, which is not pictured), Qwen has a higher mean in 130 "
        "settings, the means tie in 80, and GPT is higher in none. This count describes "
        "the scanned parameter grid, not a statistical test or model-population ranking.\n\n"
        "2. Figure macro_detail uses the post-hoc diagnostic slice b=1.02, lambda=1. "
        "At P=0.6, GPT final-window cooperation is 89.71838% (seeds: 93.28976%, 86.1470%). "
        "At P=0.7 it is 1.66584% (2.10428%, 1.22740%); Qwen is 99.16444% "
        "(99.12888%, 99.2000%). A sharp decrease occurs in both GPT seeds between "
        "these two sampled coverage values. This does not identify a precise transition threshold.\n\n"
        "3. Coverage plots show both seed observations, their mean, and their min–max "
        "range; trajectories show means and seed ranges. Ranges are not confidence "
        "intervals. Trajectories use every step through 200 and every tenth step afterward "
        "for display; all 10,000 steps are retained in the accompanying CSV. Fermi "
        "trajectories use lambda=0 from the GPT cache and were checked equal to Qwen.\n\n"
        "4. Only two simulation seeds (42,43) and one fixed empirical advice table per "
        "model are available. Two replies per configuration are not independent advice-table "
        "replicates. A final-window mean does not establish an equilibrium. Seeds and "
        "controls match, but exact initial cellwise equality was not independently "
        "recoverable from saved initial grids. These figures include no DeepSeek simulation.\n\n"
        "5. Provenance: input artifact SHA256 values were rechecked against prior audit "
        "metadata, and all 616 tail means per model were recomputed from the cached full "
        "trajectories. Source runs and prior reports were not modified. See "
        "macro_provenance.json for paths and hashes.\n", encoding="utf-8")
    print(f"Created macro_overview and macro_detail in {figures}")
    print(json.dumps(provenance["active_parameter_sets"]))


if __name__ == "__main__":
    main()
