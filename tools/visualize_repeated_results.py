"""Audit and visualize repeated scans, preserving the earlier one-seed report.

python tools/visualize_repeated_results.py --batch-dir main/SPD/results/scan_20261022_172829_883738
This layout expects the existing seven b / four lambda settings and fixed K/P.
All means and sample SDs use seeds as replicates. Paired deltas are formed first.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path, PureWindowsPath

from visualize_recent_results import (
    ROOT, INK, MUTED, COLORS, GROUP_COLORS, GROUP_NAMES, close, require, sha,
    setup_font, clean, draw_heat,
)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

DEFAULT_BATCH = ROOT / "main/SPD/results/scan_20261022_172829_883738"
DEFAULT_REPORT = ROOT / "reports/group_meeting_20261022_repeats10"
OLD_BATCH = ROOT / "main/SPD/results/scan_20261020_114505_599047"
BS = [1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.2]
LAMS = [0., .1, .5, 1.]


def load_and_audit(batch):
    config = json.loads((batch / "scan_config.json").read_text(encoding="utf-8"))
    df = pd.read_csv(batch / "summary.csv").copy()
    seeds = sorted(config["seed_values"])
    n = config["repeats"]
    require(n >= 2 and len(seeds) == len(set(seeds)) == n, "需要至少两次、种子不重复的运行。")
    require(config["status"] == "completed" and (df.status == "completed").all(), "扫描未全部完成。")
    require(len(df) == config["total_runs"] == config["completed_runs"] == 28 * n, "运行数与设计不符。")
    require(not df.duplicated(["b", "lambda_prob", "seed"]).any(), "参数/种子存在重复行。")
    require(set(zip(df.b, df.lambda_prob, df.seed)) == set(product(BS, LAMS, seeds)), "参数或种子覆盖不完整。")
    for name, value in {"k": .1, "llm_access_fraction": .2, "size": 50, "steps": 1000,
                        "boundary": "periodic", "update_mode": "synchronous", "repeats": n,
                        "tail_start_round": 901, "tail_end_round": 1000,
                        "early_start_round": 1, "early_end_round": 100}.items():
        require((df[name] == value).all(), f"本报告布局要求 {name}={value}。")
    require(df.paired_fermi_control.all(), "需要配对Fermi对照。")
    sources = [batch / "scan_config.json", batch / "summary.csv"]
    histories, grids = {}, {}
    delta_cols = [c for c in df if c.startswith("delta_")]
    for index, row in df.iterrows():
        key = (row.b, row.lambda_prob, int(row.seed))
        require(seeds[int(row.repeat_index) - 1] == row.seed, f"{key}重复序号与种子不符")
        run = (batch / row.output_dir).resolve()
        require(run.is_relative_to(batch), "结果路径必须位于扫描目录内。")
        hp, gp = run / "history.csv", run / "final_grid.csv"
        h = pd.read_csv(hp)
        grid = pd.read_csv(gp, header=None).to_numpy()
        sources.extend([hp, gp])
        close(h["round"].to_numpy(), np.arange(1001), f"{key}完整轮次")
        require(grid.shape == (50, 50) and set(grid.ravel()) <= {"C", "D"}, f"{key}网格错误")
        close((grid == "C").mean(), row.final_cooperation_rate, f"{key}网格合作率")
        for prefix in ("", "control_"):
            coop = h[prefix + "cooperation_rate"]
            require(np.isfinite(coop).all() and coop.between(0, 1).all(), f"{key}合作率无效")
            close(coop.iloc[0], row.initial_cooperation_rate, f"{key}同初态合作率")
            close(coop.iloc[-1], row[prefix + "final_cooperation_rate"], f"{key}末态合作率")
            close(coop.iloc[901:].mean(), row[prefix + "tail_mean_cooperation_rate"], f"{key}末段合作率")
            close(coop.iloc[1:].mean(), row[prefix + "time_mean_cooperation_rate"], f"{key}全程合作率")
            close(float(coop.iloc[-1] == 0), float(row[prefix + "extinct_at_end"]), f"{key}终点灭绝标记")
            for group in GROUP_NAMES:
                v = h[prefix + group + "_mean_payoff"]
                require(np.isfinite(v).all(), f"{key}收益含空值")
                for metric, expected in (("cumulative_mean_payoff", v.iloc[1:].sum()),
                                         ("time_mean_payoff", v.iloc[1:].mean()),
                                         ("early_mean_payoff", v.iloc[1:101].mean()),
                                         ("tail_mean_payoff", v.iloc[901:].mean())):
                    close(expected, row[prefix + group + "_" + metric], f"{key}{prefix}{group}_{metric}")
                close(h[prefix + group + "_cumulative_mean_payoff"].iloc[1:], v.iloc[1:].cumsum(), f"{key}逐轮累计")
            close(h[prefix + "population_mean_payoff"],
                  .2 * h[prefix + "eligible_mean_payoff"] + .8 * h[prefix + "ineligible_mean_payoff"],
                  f"{key}人数加权")
        for col in delta_cols:
            base = col.removeprefix("delta_")
            close(row[col], row[base] - row["control_" + base], f"{key}{col}")
        for col in (c for c in h if c.startswith("delta_")):
            base = col.removeprefix("delta_")
            close(h[col], h[base] - h["control_" + base], f"{key}逐轮{col}")
        if row.lambda_prob == 0:
            close(row[delta_cols].to_numpy(dtype=float), 0., f"{key}零干预差值")
        histories[key] = h
        grids[key] = grid
        if (index + 1) % 70 == 0:
            print(f"Audited {index + 1}/{len(df)} runs", flush=True)
    for b, seed in product(BS, seeds):
        base = histories[b, 0., seed]
        for lam in LAMS[1:]:
            h = histories[b, lam, seed]
            for col in (c for c in h if c.startswith("control_")):
                close(h[col], base[col.removeprefix("control_")], f"{b}/{seed}相同纯Fermi基线")
    src = config["advisor_sources"][0]
    fingerprint = {name: meta["sha256"] for name, meta in src["source_files"].items()}
    require(all({name: m["sha256"] for name, m in s["source_files"].items()} == fingerprint
                for s in config["advisor_sources"]), "扫描内建议来源不同，不能直接合并。")
    advice_dir = ROOT / "logs/GPT_Multistepsa" / PureWindowsPath(config["advisor_batch"]).name
    for name, expected in fingerprint.items():
        source = advice_dir / name
        require(sha(source) == expected, f"建议来源变动：{name}")
        sources.append(source)
    old_comparison = None
    if (OLD_BATCH / "summary.csv").exists():
        old_config = json.loads((OLD_BATCH / "scan_config.json").read_text(encoding="utf-8"))
        for field in ("fixed_parameters", "steps", "tail_average_steps", "early_average_steps", "paired_fermi_control"):
            require(config[field] == old_config[field], f"相对旧扫描除重复外还改变了{field}")
        for field in ("b", "k", "lambda_prob", "llm_access_fraction"):
            require(config["scan_values"][field] == old_config["scan_values"][field], f"相对旧扫描改变了{field}")
        old_src = old_config["advisor_sources"][0]
        require(fingerprint == {name: meta["sha256"] for name, meta in old_src["source_files"].items()}, "相对旧扫描建议表变动")
        old = pd.read_csv(OLD_BATCH / "summary.csv")
        merged = df.merge(old, on=["b", "lambda_prob", "seed"], suffixes=("_new", "_old"))
        require(len(merged) == 28, "旧seed42未完整纳入新扫描。")
        numeric = [c for c in old.select_dtypes(include="number") if c not in
                   ("b", "lambda_prob", "seed", "run_id", "parameter_set_id", "repeat_index", "repeats")]
        for col in numeric:
            close(merged[col + "_new"], merged[col + "_old"], "旧seed42复现：" + col)
        old_comparison = "same settings and advisor; all numeric outcomes of the old 28 seed42 runs reproduced"
        sources.extend([OLD_BATCH / "scan_config.json", OLD_BATCH / "summary.csv"])
    audit = {"batch": str(batch.relative_to(ROOT)), "runs": len(df), "parameter_sets": 28,
             "repeats_per_parameter": n, "seeds": seeds, "validation": "passed",
             "old_report_comparison": old_comparison,
             "statistical_unit": "seed; first compute within-seed treatment minus paired control",
             "error_bars": "sample SD across seeds (ddof=1); not SEM or confidence intervals",
             "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources}}
    return config, df, histories, grids, audit


def aggregate(df):
    metrics = ["tail_mean_cooperation_rate", "final_cooperation_rate", "time_mean_cooperation_rate"]
    metrics += [c for c in df if c.startswith("delta_")]
    records = []
    for (b, lam), group in df.groupby(["b", "lambda_prob"]):
        for metric in metrics:
            v = group[metric].to_numpy(dtype=float)
            require(np.isfinite(v).all(), f"{b}/{lam}/{metric}含空值")
            records.append({"b": b, "lambda_prob": lam, "metric": metric, "n": len(v),
                            "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                            "min": float(v.min()), "max": float(v.max()),
                            "positive_count": int((v > 1e-12).sum())})
    return pd.DataFrame.from_records(records)


def page(title, subtitle, number, n, batch_date, footnote):
    fig = plt.figure(figsize=(16, 9))
    fig.text(.055, .958, f"LLM ADVICE  /  REPEATED SIMULATIONS  /  n = {n}", fontsize=10, color=MUTED)
    fig.text(.055, .901, title, fontsize=26, weight="bold")
    fig.text(.055, .853, subtitle, fontsize=13, color=MUTED)
    fig.add_artist(plt.Line2D([.055, .945], [.825, .825], transform=fig.transFigure, color="#DCE5EB", lw=1))
    fig.text(.055, .102, footnote, fontsize=10.5, color=MUTED, va="top", linespacing=1.55)
    fig.text(.055, .025, f"来源：{batch_date} 扫描  ·  每组 {n} 个种子  ·  同一离线建议表", fontsize=10, color=MUTED)
    fig.text(.945, .025, number, fontsize=11, ha="right", color=MUTED)
    return fig


def matrix(stats, metric, field="mean", lams=LAMS):
    s = stats[stats.metric == metric]
    return s.pivot(index="b", columns="lambda_prob", values=field).loc[BS, lams].to_numpy()


def heat_sd(ax, means, sds, xlabels, cmap, norm, title=None, unit=""):
    im = draw_heat(ax, means, xlabels, [f"{b:g}" for b in BS], cmap, norm, lambda v: "", title=title)
    for (i, j), value in np.ndenumerate(means):
        color = im.cmap(im.norm(value))
        light = .2126 * color[0] + .7152 * color[1] + .0722 * color[2]
        ink = "white" if light < .55 else INK
        value_text = f"{value:.2f}{unit}" if unit else f"{value:+.1f}"
        sd_text = f"± {sds[i,j]:.2f}" if unit else f"± {sds[i,j]:.1f}"
        ax.text(j, i - .14, value_text, fontsize=15, ha="center", va="center", color=ink, weight="bold")
        ax.text(j, i + .20, sd_text, fontsize=10.5, ha="center", va="center", color=ink)
    return im


def make_figures(df, stats, histories, grids, n, seeds, date):
    seq = LinearSegmentedColormap.from_list("coop_repeat", ["#F2F6F8", "#71C8BD", "#067D79", "#153951"])
    div = LinearSegmentedColormap.from_list("gain_repeat", ["#B64A42", "#FFFDFC", "#108B86"])
    def new(title, sub, idx, note):
        return page(title, sub, idx, n, date, note)
    figures = []
    f = new("重复运行后的合作率：均值与波动一起看", "第901–1000步合作率：先求每次运行的末段均值，再在种子之间汇总", "01",
            "格内上行为均值（%），下行为 ± 样本SD（百分点）；SD表示种子间离散程度，不是置信区间。\nL=50，K=0.1，P=20%，同步周期边界；全部参数组合均包含相同的一组种子。")
    ax = f.add_axes([.11, .21, .52, .57])
    means = matrix(stats, "tail_mean_cooperation_rate") * 100
    sds = matrix(stats, "tail_mean_cooperation_rate", "sd") * 100
    im = heat_sd(ax, means, sds, ["λ=0\n纯 Fermi", "λ=0.1", "λ=0.5", "λ=1"], seq, Normalize(0, 100), unit="%")
    ax.set_ylabel("背叛诱惑收益 b", labelpad=17)
    cax = f.add_axes([.655, .21, .015, .57]); f.colorbar(im, cax=cax, format=PercentFormatter(100))
    f.text(.735, .71, "重复后更新结论", fontsize=19, weight="bold")
    f.text(.735, .61, f"b=1.02：λ=0.5 均值\n{means[1,2]:.2f}%，λ=1 为 {means[1,3]:.2f}%。", fontsize=15, linespacing=1.8)
    f.text(.735, .44, "均值高低有波动，\n不单凭排序判定最优。", fontsize=15, linespacing=1.8)
    f.text(.735, .27, "本次 b≥1.04 的所有\n运行末100步均为全 D。", fontsize=15, linespacing=1.8)
    figures.append(("01_cooperation", f))

    f = new("完整演化：平均趋势与运行范围", f"粗线为{n}次运行均值；阴影为同一轮的最小–最大范围（不是置信区间）", "02",
            "灰色竖带为第901–1000步；不平滑。同一 b、seed 的重复Fermi对照只计一次。\n种子是重复单位；时间步及格点不增加独立样本量。")
    gs = f.add_gridspec(2, 4, left=.065, right=.96, bottom=.20, top=.77, wspace=.25, hspace=.42)
    for i, b in enumerate(BS):
        ax = f.add_subplot(gs[i // 4, i % 4]); ax.axvspan(901, 1000, color="#E8EDF2", zorder=0)
        for lam in LAMS:
            arr = np.stack([histories[b, lam, seed].cooperation_rate.to_numpy() for seed in seeds])
            x = np.arange(arr.shape[1]); color = COLORS[lam]
            ax.fill_between(x, arr.min(axis=0), arr.max(axis=0), color=color, alpha=.12, linewidth=0)
            ax.plot(x, arr.mean(axis=0), color=color, lw=1.9, ls="--" if lam == 0 else "-")
        ax.set(xlim=(0, 1000), ylim=(-.01, .65), title=f"b={b:g}", xticks=[0, 500, 1000])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.tick_params(labelsize=10)
        if i % 4 == 0: ax.set_ylabel("合作率")
        if i >= 4: ax.set_xlabel("时间步")
        clean(ax)
    ax = f.add_subplot(gs[1, 3]); ax.axis("off")
    handles = [plt.Line2D([], [], color=COLORS[l], lw=3, ls="--" if l == 0 else "-",
                         label="λ=0 · Fermi" if l == 0 else f"λ={l:g}") for l in LAMS]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=15)
    ax.text(.02, .02, "阴影宽：运行差异大\n阴影窄：运行较一致", fontsize=12, color=MUTED, transform=ax.transAxes)
    figures.append(("02_dynamics", f))

    f = new("累计收益：先求配对差，再汇总种子", "每个种子：建议实验 − 同初态纯 Fermi；累计第1–1000步的轮末收益", "03",
            "格内为人均累计收益差的均值 ± 样本SD；三个面板按均值共用色标，SD不是置信区间。\n按固定资格分组；均值满足20%/80%人数加权，SD单独计算；差值包含邻域传播作用。")
    gs = f.add_gridspec(1, 3, left=.08, right=.91, bottom=.22, top=.73, wspace=.25)
    lim = max(10., np.ceil(max(np.abs(matrix(stats, f"delta_{g}_cumulative_mean_payoff")).max()
                              for g in GROUP_NAMES) / 25) * 25)
    for i, group in enumerate(GROUP_NAMES):
        metric = f"delta_{group}_cumulative_mean_payoff"; ax = f.add_subplot(gs[0, i])
        im = heat_sd(ax, matrix(stats, metric, lams=LAMS[1:]), matrix(stats, metric, "sd", LAMS[1:]),
                     ["λ=0.1", "λ=0.5", "λ=1"], div, TwoSlopeNorm(0, -lim, lim), title=GROUP_NAMES[group])
        if i == 0: ax.set_ylabel("背叛诱惑收益 b")
    cax = f.add_axes([.935, .22, .014, .51]); f.colorbar(im, cax=cax)
    figures.append(("03_welfare", f))

    f = new("合作改善是否在不同种子中一致？", "末100步合作率的配对差（百分点）；每个圆点是一个种子，菱形为均值，误差条为 ±1 SD", "04",
            "先在相同seed内减去配对Fermi，再计算均值和SD；不将不同λ共享的对照视作额外重复。\n“提高次数”统计配对差>0的运行数；SD不是置信区间，不作显著性标记。")
    gs = f.add_gridspec(1, 3, left=.07, right=.95, bottom=.23, top=.73, wspace=.25)
    for i, b in enumerate(BS[:3]):
        ax = f.add_subplot(gs[0, i]); ax.axhline(0, color=INK, lw=.9)
        for j, lam in enumerate(LAMS[1:]):
            sub = df[(df.b == b) & (df.lambda_prob == lam)].sort_values("seed")
            v = sub.delta_tail_mean_cooperation_rate.to_numpy() * 100
            ax.scatter(j + np.linspace(-.14, .14, n), v, s=35, color=COLORS[lam], alpha=.55, zorder=3)
            ax.errorbar(j, v.mean(), yerr=v.std(ddof=1), fmt="D", ms=8, color=INK, capsize=6, lw=1.7, zorder=4)
            ax.text(j, .94, f"{(v > 1e-10).sum()}/{n} 次提高", ha="center", fontsize=12, transform=ax.get_xaxis_transform())
            ax.text(j, -.11, f"{v.mean():+.2f}\n± {v.std(ddof=1):.2f}", ha="center", va="top", fontsize=11,
                    transform=ax.get_xaxis_transform(), clip_on=False, linespacing=1.3)
        ax.set(xticks=range(3), xticklabels=["λ=0.1", "λ=0.5", "λ=1"], title=f"b={b:g}", ylim=(-17, 31), xlim=(-.5, 2.5))
        if i == 0: ax.set_ylabel("Δ 末段合作率（百分点）")
        clean(ax)
    figures.append(("04_paired_effects", f))

    f = new("复核旧案例：单次的收益分化没有稳定重现", "原报告的两个参数组合；圆点为每个seed，空心圆突出seed42，菱形及误差条为均值 ± SD", "05",
            "Δ 为相同资格位置相对配对Fermi的人均累计收益差；纵轴在两个案例中保持一致。\nseed42已包含在新批次中，不重复增加样本；“资格增、全体减”按每个seed的两个差值联合判断。")
    gs = f.add_gridspec(1, 2, left=.09, right=.94, bottom=.24, top=.72, wspace=.25)
    groups = ["eligible", "ineligible", "population"]
    cases = [(1.04, 1.), (1.05, .5)]
    all_values = np.concatenate([df[(df.b == b) & (df.lambda_prob == lam)][
        [f"delta_{g}_cumulative_mean_payoff" for g in groups]].to_numpy().ravel() for b, lam in cases])
    low, high = all_values.min(), all_values.max(); pad = (high - low) * .17
    for i, (b, lam) in enumerate(cases):
        sub = df[(df.b == b) & (df.lambda_prob == lam)].sort_values("seed")
        ax = f.add_subplot(gs[0, i]); ax.axhline(0, color=INK, lw=1)
        for j, g in enumerate(groups):
            v = sub[f"delta_{g}_cumulative_mean_payoff"].to_numpy()
            x = j + np.linspace(-.15, .15, n)
            ax.scatter(x, v, color=GROUP_COLORS[g], s=35, alpha=.55)
            mask = sub.seed.to_numpy() == 42
            ax.scatter(x[mask], v[mask], facecolors="none", edgecolors=INK, s=110, lw=1.5, zorder=4)
            ax.errorbar(j, v.mean(), yerr=v.std(ddof=1), fmt="D", ms=8, color=INK, capsize=6, zorder=5)
            ax.text(j, -.16, f"{v.mean():+.2f} ± {v.std(ddof=1):.2f}", transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=12, clip_on=False)
        count = ((sub.delta_eligible_cumulative_mean_payoff > 0) & (sub.delta_population_cumulative_mean_payoff < 0)).sum()
        ax.set(xticks=range(3), xticklabels=[GROUP_NAMES[g] for g in groups],
               title=f"b={b:g}，λ={lam:g}  |  资格增、全体减：{count}/{n}", ylim=(low - pad, high + pad), xlim=(-.5, 2.5))
        if i == 0: ax.set_ylabel("Δ 人均累计收益（第1–1000步）")
        clean(ax)
    figures.append(("05_tradeoff_recheck", f))

    f = new("到第1000步，多少次运行仍有合作？", "逐运行统计最终合作率>0；展示存活次数，不把有限时间观测解释为长期存活概率", "06",
            "末100步均值>0与第1000步仍有合作是不同指标；某些运行在末段内归零。\n表中次数来自同一建议表下的种子重复，未计入建议表重新生成的不确定性。")
    ax = f.add_axes([.11, .21, .52, .57])
    counts = df.assign(alive=(df.final_cooperation_rate > 0).astype(int)).groupby(["b", "lambda_prob"]).alive.sum().unstack().loc[BS, LAMS].to_numpy()
    im = draw_heat(ax, counts, ["λ=0\n纯 Fermi", "λ=0.1", "λ=0.5", "λ=1"], [f"{b:g}" for b in BS],
                   seq, Normalize(0, n), lambda v: f"{v}/{n}", fontsize=20)
    ax.set_ylabel("背叛诱惑收益 b", labelpad=17)
    cax = f.add_axes([.655, .21, .015, .57]); f.colorbar(im, cax=cax)
    f.text(.735, .67, "临界附近存在差异", fontsize=19, weight="bold")
    f.text(.735, .54, f"b=1.03、λ=1：\n{counts[2,3]}/{n} 次在终点仍有合作。", fontsize=16, linespacing=1.8)
    f.text(.735, .36, "相同 b 下其余 λ\n在终点均已全 D。", fontsize=16, linespacing=1.8)
    figures.append(("06_survival", f))

    f = new("空间快照：固定seed42，便于与旧报告对照", "第1000步真实网格；只展示一个预先固定的种子，不代表10次运行的平均空间状态", "07",
            "青绿 = 合作 C；浅灰 = 背叛 D。标签为该次运行第1000步合作率，不是多次平均。\n展示b=1.01/1.02/1.03；空间示例只作辅助，主要结论以全部种子汇总为准。")
    gs = f.add_gridspec(3, 4, left=.15, right=.88, bottom=.17, top=.77, wspace=.12, hspace=.27)
    cmap = ListedColormap(["#E8EDF2", "#008F87"])
    require(42 in seeds, "空间对照图需要seed42。")
    for i, b in enumerate(BS[:3]):
        for j, lam in enumerate(LAMS):
            ax = f.add_subplot(gs[i, j]); grid = grids[b, lam, 42]
            ax.imshow(grid == "C", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if i == 0: ax.set_title("λ=0 · Fermi" if lam == 0 else f"λ={lam:g}", fontsize=15, pad=12)
            if j == 0: ax.set_ylabel(f"b={b:g}", fontsize=15, rotation=0, labelpad=43, va="center")
            ax.text(1.06, .5, f"{(grid == 'C').mean():.1%}", transform=ax.transAxes, va="center", fontsize=12, color=MUTED)
    figures.append(("07_spatial", f))
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--font")
    args = parser.parse_args()
    font = setup_font(args.font)
    batch = args.batch_dir.resolve(); output = args.report_dir.resolve()
    require(output != ROOT / "reports/group_meeting_20261022", "请选择新报告目录，保留旧单次结果。")
    config, df, histories, grids, audit = load_and_audit(batch)
    stats = aggregate(df)
    tables = output / "tables"; tables.mkdir(parents=True, exist_ok=True)
    figdir = output / "figures"; figdir.mkdir(exist_ok=True)
    stats.to_csv(tables / "跨种子汇总_均值_SD.csv", index=False, encoding="utf-8-sig")
    df.to_csv(tables / "逐次运行汇总.csv", index=False, encoding="utf-8-sig")
    for field, label in [("mean", "均值"), ("sd", "SD")]:
        pd.DataFrame(matrix(stats, "tail_mean_cooperation_rate", field) * 100, index=BS, columns=LAMS).rename_axis("b").to_csv(
            tables / f"末100步合作率_{label}_百分比.csv", encoding="utf-8-sig")
    alive = df.assign(alive=(df.final_cooperation_rate > 0).astype(int)).groupby(["b", "lambda_prob"]).alive.agg(["sum", "count"])
    alive.to_csv(tables / "终点合作存活次数.csv", encoding="utf-8-sig")
    metric_cols = ["b", "lambda_prob", "seed", "delta_tail_mean_cooperation_rate"] + [
        f"delta_{g}_cumulative_mean_payoff" for g in GROUP_NAMES]
    df[metric_cols].to_csv(tables / "逐种子配对差.csv", index=False, encoding="utf-8-sig")
    audit["font_family"] = font
    audit["versions"] = {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    (output / "数据核验.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stamp = batch.name.split("_")[1]; date = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
    figures = make_figures(df, stats, histories, grids, config["repeats"], audit["seeds"], date)
    with PdfPages(output / "all_figures.pdf") as pdf:
        for name, fig in figures:
            fig.savefig(figdir / (name + ".png"), dpi=300)
            fig.savefig(figdir / (name + ".svg"))
            fig.savefig(figdir / (name + ".pdf"))
            pdf.savefig(fig); plt.close(fig)
            print(f"Exported {name}", flush=True)
    print(json.dumps({"report_dir": str(output), "figures": len(figures), "runs_validated": len(df)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
