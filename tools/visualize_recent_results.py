"""Reproduce the audited October 20 scan figures without running experiments.

python tools/visualize_recent_results.py --batch-dir main/SPD/results/scan_20261020_114505_599047
Dependencies: matplotlib, numpy, pandas. Optional --font points to a CJK font.
This report supports one K/P/seed and the seven b/four lambda values in this scan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath

os.environ.setdefault("MPLCONFIGDIR", "/tmp/llm_advice_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "main/SPD/results/scan_20261020_114505_599047"
INK = "#18334A"
MUTED = "#607184"
COLORS = {0.0: "#566777", 0.1: "#287AB5", 0.5: "#009E91", 1.0: "#D67B2A"}
GROUP_COLORS = {"population": "#566777", "eligible": "#009E91", "ineligible": "#D67B2A"}
GROUP_NAMES = {"population": "全体", "eligible": "有建议资格者", "ineligible": "无建议资格者"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, label):
    require(np.allclose(a, b, rtol=1e-8, atol=1e-9, equal_nan=True), f"数据核验失败：{label}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_font(explicit):
    candidates = [explicit] if explicit else []
    candidates += ["/mnt/c/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyh.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    for value in candidates:
        if value and Path(value).is_file():
            font_manager.fontManager.addfont(value)
            for bold in (Path(value).with_name("msyhbd.ttc"),):
                if bold.is_file():
                    font_manager.fontManager.addfont(str(bold))
            family = font_manager.FontProperties(fname=value).get_name()
            break
    else:
        family = next((f.name for f in font_manager.fontManager.ttflist
                       if any(s in f.name for s in ("CJK", "Microsoft YaHei", "SimHei"))), None)
        require(family, "未找到中文字体，请使用 --font 指定中文 TTF/TTC 文件。")
    plt.rcParams.update({"font.family": family, "font.size": 13, "axes.unicode_minus": False,
                        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#CBD5DF",
                        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
                        "axes.spines.top": False, "axes.spines.right": False,
                        "figure.facecolor": "white", "axes.facecolor": "white",
                        "pdf.fonttype": 42, "svg.fonttype": "path", "savefig.facecolor": "white"})
    return family


def load_and_audit(batch):
    config = json.loads((batch / "scan_config.json").read_text(encoding="utf-8"))
    df = pd.read_csv(batch / "summary.csv")
    require(config["status"] == "completed" and (df.status == "completed").all(), "批次未全部完成。")
    require(len(df) == config["total_runs"] == config["completed_runs"] == 28, "预期本次完整28组扫描。")
    require(df.b.nunique() == 7 and set(df.b) == {1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.2}, "b 扫描范围不同。")
    require(set(df.lambda_prob) == {0., .1, .5, 1.}, "lambda 扫描范围不同。")
    for field in ("seed", "k", "llm_access_fraction", "size", "steps", "update_mode", "boundary"):
        require(df[field].nunique() == 1, f"本报告要求单一 {field}，以免合并不同条件。")
    require(not df.duplicated(["b", "lambda_prob"]).any(), "存在重复参数行。")
    require(df.seed.iloc[0] == 42 and df.steps.iloc[0] == 1000 and df["size"].iloc[0] == 50,
            "本报告针对 seed42、50×50、1000步的批次。")
    require(df.k.iloc[0] == .1 and df.llm_access_fraction.iloc[0] == .2
            and df.update_mode.iloc[0] == "synchronous" and df.boundary.iloc[0] == "periodic",
            "本报告要求 K=.1、P=.2、同步周期边界。")
    require(df.paired_fermi_control.all() and (df.repeats == 1).all(), "需要单次配对对照。")
    require((df.early_start_round == 1).all() and (df.early_end_round == 100).all()
            and (df.tail_start_round == 901).all() and (df.tail_end_round == 1000).all(), "平均窗口不同。")
    sources = [batch / "scan_config.json", batch / "summary.csv"]
    history, grids = {}, {}
    for _, row in df.iterrows():
        key = (row.b, row.lambda_prob)
        run = batch / row.output_dir
        hp, gp = run / "history.csv", run / "final_grid.csv"
        h = pd.read_csv(hp)
        grid = pd.read_csv(gp, header=None).to_numpy()
        sources.extend([hp, gp])
        close(h["round"].to_numpy(), np.arange(1001), f"{key} 完整轮次")
        require(grid.shape == (50, 50) and set(grid.ravel()) <= {"C", "D"}, f"{key} 网格无效")
        close((grid == "C").mean(), row.final_cooperation_rate, f"{key} 最终网格合作率")
        for prefix in ("", "control_"):
            coop = h[prefix + "cooperation_rate"]
            close(coop.iloc[-1], row[prefix + "final_cooperation_rate"], f"{key} 最终合作")
            close(coop.iloc[901:1001].mean(), row[prefix + "tail_mean_cooperation_rate"], f"{key} 末段合作")
            close(coop.iloc[1:].mean(), row[prefix + "time_mean_cooperation_rate"], f"{key} 全程合作")
            for group in GROUP_NAMES:
                values = h[prefix + group + "_mean_payoff"]
                for metric, expected in (("cumulative_mean_payoff", values.iloc[1:].sum()),
                                         ("time_mean_payoff", values.iloc[1:].mean()),
                                         ("early_mean_payoff", values.iloc[1:101].mean()),
                                         ("tail_mean_payoff", values.iloc[901:1001].mean())):
                    close(expected, row[prefix + group + "_" + metric], f"{key} {prefix}{group}_{metric}")
                close(h[prefix + group + "_cumulative_mean_payoff"].iloc[1:],
                      values.iloc[1:].cumsum(), f"{key} 逐轮累计")
            close(h[prefix + "population_mean_payoff"],
                  .2 * h[prefix + "eligible_mean_payoff"] + .8 * h[prefix + "ineligible_mean_payoff"],
                  f"{key} 人数加权")
        for col in (c for c in df if c.startswith("delta_")):
            base = col.removeprefix("delta_")
            close(row[col], row[base] - row["control_" + base], f"{key} {col}")
        for col in (c for c in h if c.startswith("delta_")):
            base = col.removeprefix("delta_")
            close(h[col], h[base] - h["control_" + base], f"{key} 逐轮{col}")
        history[key], grids[key] = h, grid
    for b in sorted(df.b.unique()):
        baseline = history[b, 0.]
        for lam in (.1, .5, 1.):
            close(history[b, lam].control_cooperation_rate, baseline.cooperation_rate, f"b={b}重复基线")
    src = config["advisor_sources"][0]
    batch_name = PureWindowsPath(config["advisor_batch"]).name
    advice_dir = ROOT / "logs/GPT_Multistepsa" / batch_name
    for name, meta in src["source_files"].items():
        source = advice_dir / name
        require(sha(source) == meta["sha256"], f"策略来源文件变动：{name}")
        sources.append(source)
    advice = pd.read_csv(advice_dir / "results.csv", encoding="utf-8-sig")
    require(len(advice) == 1024 and (advice.status == "success").all(), "策略表不是1024条成功记录。")
    require(advice.grid_key.nunique() == 512 and (advice.groupby("grid_key").size() == 2).all(), "局面覆盖不完整。")
    require(set(advice.recommendation) <= {"C", "D"}, "策略表存在无效建议。")
    advice["center"] = advice.grid_key.str[4]
    advice["cardinal_c"] = advice.grid_key.map(lambda s: sum(s[i] == "C" for i in (1, 3, 5, 7)))
    advice["is_c"] = (advice.recommendation == "C").astype(int)
    advisor_stats = advice.groupby(["center", "cardinal_c"]).is_c.agg(["sum", "count", "mean"]).reset_index()
    audit = {"scan": str(batch.relative_to(ROOT)) if batch.is_relative_to(ROOT) else str(batch),
             "runs": 28, "history_rows_per_run": 1001, "independent_repeats_per_parameter": 1,
             "seed": 42, "validation": "passed: summary/history/grid, payoffs, paired deltas, advisor hashes",
             "checks": ["28组逐轮合作率与summary一致", "轮末收益累计/前期/末期均值一致",
                        "所有history和summary配对差值一致", "最终网格合作率一致", "策略表来源哈希一致"],
             "source_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in sources}}
    return config, df, history, grids, advice, advisor_stats, audit


def page(title, subtitle, number, footnote=None):
    fig = plt.figure(figsize=(16, 9))
    fig.text(.055, .958, "LLM ADVICE  /  SPATIAL PRISONER'S DILEMMA", fontsize=10, color=MUTED)
    fig.text(.055, .901, title, fontsize=27, weight="bold")
    fig.text(.055, .853, subtitle, fontsize=13, color=MUTED)
    fig.add_artist(plt.Line2D([.055, .945], [.825, .825], transform=fig.transFigure, color="#DCE5EB", lw=1))
    if footnote:
        fig.text(.055, .102, footnote, fontsize=11, color=MUTED, va="top", linespacing=1.65)
    fig.text(.055, .025, "来源：2026-10-20 扫描  ·  单次运行 seed=42  ·  不表示统计显著性", fontsize=10, color=MUTED)
    fig.text(.945, .025, number, fontsize=11, ha="right", color=MUTED)
    return fig


def clean(ax):
    ax.grid(axis="y", color="#E5EBF0", lw=.7)
    ax.set_axisbelow(True)


def draw_heat(ax, values, xlabels, ylabels, cmap, norm, fmt, title=None, fontsize=14):
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(length=0, pad=10)
    ax.set_xticks(np.arange(-.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for (i, j), value in np.ndenumerate(values):
        rgba = im.cmap(im.norm(value))
        light = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
        ax.text(j, i, fmt(value), ha="center", va="center", fontsize=fontsize,
                color="white" if light < .55 else INK)
    if title:
        ax.set_title(title, fontsize=17, pad=17)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def make_figures(df, histories, grids, advice, stats):
    bs = sorted(df.b.unique())
    lams = sorted(df.lambda_prob.unique())
    seq = LinearSegmentedColormap.from_list("coop", ["#F2F6F8", "#71C8BD", "#067D79", "#153951"])
    div = LinearSegmentedColormap.from_list("gain", ["#B64A42", "#FFFDFC", "#108B86"])
    foot = "L=50，K=0.1，P=20%，同步更新，周期边界；λ 是有资格者每次更新采用建议的概率。"
    figures = []

    f = page("合作水平随 b 和建议采纳概率变化", "末100步平均合作率（第901–1000步）；每个格子对应一组参数的一次运行", "01", foot)
    ax = f.add_axes([.11, .19, .52, .59])
    values = df.pivot(index="b", columns="lambda_prob", values="tail_mean_cooperation_rate").loc[bs, lams].to_numpy()
    im = draw_heat(ax, values, ["λ = 0\n纯 Fermi", "λ = 0.1", "λ = 0.5", "λ = 1"],
                   [f"{b:g}" for b in bs], seq, matplotlib.colors.Normalize(0, 1), lambda v: f"{v:.2%}", fontsize=17)
    ax.set_ylabel("背叛诱惑收益 b", labelpad=17)
    cax = f.add_axes([.655, .19, .015, .59]); f.colorbar(im, cax=cax, format=PercentFormatter(1))
    f.text(.735, .70, "非单调响应", fontsize=21, weight="bold")
    f.text(.735, .61, "b=1.01：λ=0.5 的末段\n合作率高于 λ=1。", fontsize=16, linespacing=1.8)
    f.text(.735, .46, "b=1.03：仅 λ=1\n在末段仍有合作。", fontsize=16, linespacing=1.8)
    f.text(.735, .30, "b≥1.04：本次扫描\n所有组末段均为全 D。", fontsize=16, linespacing=1.8)
    figures.append(("01_cooperation", f))

    f = page("演化轨迹展示了末段均值之外的差异", "全部7个 b；原始逐轮合作率，不平滑；所有子图使用相同纵轴", "02",
             "灰色区域为第901–1000步；同一 b 的配对 Fermi 轨迹一致，因此仅展示一条基线。")
    gs = f.add_gridspec(2, 4, left=.065, right=.96, bottom=.18, top=.77, wspace=.23, hspace=.44)
    for idx, b in enumerate(bs):
        ax = f.add_subplot(gs[idx // 4, idx % 4])
        ax.axvspan(901, 1000, color="#E8EDF2", zorder=0)
        for lam in lams:
            h = histories[b, lam]
            ax.plot(h["round"], h.cooperation_rate, color=COLORS[lam], lw=1.35,
                    alpha=.9, linestyle="--" if lam == 0 else "-", label=f"λ={lam:g}")
        ax.set(xlim=(0, 1000), ylim=(-.015, .70), title=f"b = {b:g}", xticks=[0, 500, 1000])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(labelsize=11)
        if idx % 4 == 0: ax.set_ylabel("合作率")
        if idx >= 4: ax.set_xlabel("时间步")
        clean(ax)
    ax = f.add_subplot(gs[1, 3]); ax.axis("off")
    handles = [plt.Line2D([], [], color=COLORS[l], lw=3, ls="--" if l == 0 else "-",
                         label="λ=0 · 纯 Fermi" if l == 0 else f"λ={l:g}") for l in lams]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=15)
    ax.text(.02, .02, "单一 seed 的轨迹\n不将时间步视作独立重复。", fontsize=12, color=MUTED, transform=ax.transAxes)
    figures.append(("02_dynamics", f))

    f = page("建议带来的累计收益变化取决于参数与人群", "人均累计收益配对差 = 建议实验 − 同初态纯 Fermi；累计第1–1000步轮末收益", "03",
             "三个面板共用色标；正数为相对对照获益。全体 = 20% × 有资格组 + 80% × 无资格组。\n组别按固定资格位置划分；差值包含邻域传播作用，不是一次咨询的直接效应。")
    gs = f.add_gridspec(1, 3, left=.08, right=.91, bottom=.22, top=.73, wspace=.25)
    all_values = df[[f"delta_{g}_cumulative_mean_payoff" for g in GROUP_NAMES]].to_numpy()
    lim = np.ceil(np.max(np.abs(all_values)) / 50) * 50
    for idx, group in enumerate(GROUP_NAMES):
        ax = f.add_subplot(gs[0, idx])
        arr = df.pivot(index="b", columns="lambda_prob", values=f"delta_{group}_cumulative_mean_payoff").loc[bs, [.1, .5, 1.]].to_numpy()
        im = draw_heat(ax, arr, ["λ=0.1", "λ=0.5", "λ=1"], [f"{b:g}" for b in bs], div,
                       TwoSlopeNorm(0, -lim, lim), lambda v: f"{v:+.1f}", title=GROUP_NAMES[group], fontsize=15)
        if idx == 0: ax.set_ylabel("背叛诱惑收益 b")
    cax = f.add_axes([.935, .22, .014, .51]); f.colorbar(im, cax=cax)
    figures.append(("03_welfare", f))

    f = page("少数组合出现“资格组略增益、全体受损”", "两个观察到的案例；差值幅度较小，只作为后续重复实验的线索", "04",
             "Δ 为同位置分组相对纯 Fermi 的人均累计收益差；不比较组内总额。\n仅 seed=42：不能据此认定存在稳定的收益转移；两案例末100步合作率均为0。")
    gs = f.add_gridspec(1, 2, left=.09, right=.94, bottom=.23, top=.73, wspace=.30)
    for idx, (b, lam) in enumerate([(1.04, 1.), (1.05, .5)]):
        row = df[(df.b == b) & (df.lambda_prob == lam)].iloc[0]
        ax = f.add_subplot(gs[0, idx])
        groups = ["eligible", "ineligible", "population"]
        values = [row[f"delta_{g}_cumulative_mean_payoff"] for g in groups]
        bars = ax.bar(range(3), values, width=.56, color=[GROUP_COLORS[g] for g in groups])
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(range(3), [GROUP_NAMES[g] for g in groups], fontsize=14)
        ax.set(ylim=(-11, 3), title=f"b = {b:g}，λ = {lam:g}")
        if idx == 0: ax.set_ylabel("Δ 人均累计收益（第1–1000步）")
        clean(ax)
        for rect, v in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, v + (.28 if v > 0 else -.28),
                    f"{v:+.3f}", ha="center", va="bottom" if v > 0 else "top", fontsize=19, weight="bold")
    figures.append(("04_group_tradeoff", f))

    f = page("最终空间分布：合作团簇在何处保留", "第1000步的真实策略网格；展示 b=1.01、1.02、1.03，其余 b 的末态均为全 D", "05",
             "青绿 = 合作 C；浅灰 = 背叛 D。标签为第1000步合作率，与末100步平均值不同。")
    gs = f.add_gridspec(3, 4, left=.15, right=.88, bottom=.15, top=.77, wspace=.12, hspace=.27)
    cmap = ListedColormap(["#E8EDF2", "#008F87"])
    for i, b in enumerate([1.01, 1.02, 1.03]):
        for j, lam in enumerate(lams):
            ax = f.add_subplot(gs[i, j]); grid = grids[b, lam]
            ax.imshow(grid == "C", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if i == 0: ax.set_title("λ=0 · Fermi" if lam == 0 else f"λ={lam:g}", fontsize=15, pad=12)
            if j == 0: ax.set_ylabel(f"b={b:g}", fontsize=15, rotation=0, labelpad=43, va="center")
            ax.text(1.06, .5, f"{(grid == 'C').mean():.1%}", transform=ax.transAxes,
                    va="center", fontsize=12, color=MUTED)
    figures.append(("05_spatial", f))

    f = page("策略表更常建议保持合作，较少建议从 D 转为 C", "离线建议表：512个完整3×3局面 × 每局面2次回复；共1024条成功记录", "06",
             "等权枚举局面统计，非仿真中实际遇到的局面频率；按四邻居数汇总仅用于展示，仿真仍匹配完整3×3。\n每个3×3局面仅2次回复；该图描述条件行为，不单独证明宏观结果的原因。")
    ax = f.add_axes([.09, .23, .55, .53])
    for state, color, marker in [("C", "#009E91", "o"), ("D", "#D67B2A", "s")]:
        sub = stats[stats.center == state].sort_values("cardinal_c")
        ax.plot(sub.cardinal_c, sub["mean"], marker=marker, color=color, lw=3, ms=9,
                label=f"本人当前为 {state}")
        for _, row in sub.iterrows():
            offset = 12 if state == "C" else -20
            ax.annotate(f"{row['mean']:.1%}", (row.cardinal_c, row["mean"]), xytext=(0, offset),
                        textcoords="offset points", ha="center", color=color, fontsize=12)
    ax.set(xlim=(-.15, 4.15), ylim=(-.06, 1.12), xticks=range(5),
           xlabel="四个直接邻居中的合作人数", ylabel="建议合作 C 的比例")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); clean(ax)
    ax.legend(frameon=False, loc="upper left")
    f.text(.72, .69, "总体合作建议比例", fontsize=16)
    f.text(.72, .61, f"{advice.is_c.mean():.2%}", fontsize=34, weight="bold", color="#008F87")
    f.text(.72, .51, "各邻居数下，每种本人状态\n的回复数分别为：\n32 / 128 / 192 / 128 / 32", fontsize=13, linespacing=1.7, color=MUTED)
    f.text(.72, .31, "所有 b 共用同一建议表；\n通过本地有放回抽样复用。", fontsize=14, linespacing=1.7)
    figures.append(("06_advisor", f))
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/group_meeting_20261022")
    parser.add_argument("--font")
    args = parser.parse_args()
    font = setup_font(args.font)
    batch = args.batch_dir.resolve()
    config, df, histories, grids, advice, stats, audit = load_and_audit(batch)
    output = args.report_dir.resolve(); figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"; tables.mkdir(exist_ok=True)
    df.pivot(index="b", columns="lambda_prob", values="tail_mean_cooperation_rate").mul(100).to_csv(
        tables / "末100步合作率_百分比.csv", encoding="utf-8-sig")
    columns = ["b", "lambda_prob", "seed", "tail_mean_cooperation_rate", "delta_tail_mean_cooperation_rate",
               "first_extinction_round", "final_cooperation_rate"]
    columns += [f"delta_{g}_cumulative_mean_payoff" for g in GROUP_NAMES]
    df[columns].to_csv(tables / "关键指标.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(tables / "策略表分组统计.csv", index=False, encoding="utf-8-sig")
    audit["font_family"] = font
    audit["plot_versions"] = {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    (output / "数据核验.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    figures = make_figures(df, histories, grids, advice, stats)
    with PdfPages(output / "all_figures.pdf") as pdf:
        for name, fig in figures:
            fig.savefig(figures_dir / (name + ".png"), dpi=300)
            fig.savefig(figures_dir / (name + ".svg"))
            fig.savefig(figures_dir / (name + ".pdf"))
            pdf.savefig(fig)
            plt.close(fig)
    print(json.dumps({"report_dir": str(output), "figures": len(figures), "validation": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
