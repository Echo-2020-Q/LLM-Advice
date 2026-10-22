#!/usr/bin/env python3
"""Build the 10-repeat group-meeting presentation and Chinese speaker script.

Usage:
    python tools/build_repeated_meeting_deck.py --report-dir reports/... \
        --batch-dir main/SPD/results/scan_20261022_172829_883738

Read-only source experiments; requires python-pptx and seven report figures.
The introductory and concluding slides contain editable text. Every slide has
speaker notes. This narrative is specific to the 28-setting, 10-seed scan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

from build_group_meeting_deck import (
    INK, MUTED, NAVY, PALE, TEAL, WHITE,
    add_notes, figure_slide, footer, header, new_slide, rectangle, text,
)


REPORT_DATE = "2026-10-22"
FIGURE_NAMES = (
    "01_cooperation.png",
    "02_dynamics.png",
    "04_paired_effects.png",
    "03_welfare.png",
    "05_tradeoff_recheck.png",
    "06_survival.png",
    "07_spatial.png",
)


def read_inputs(batch_dir: Path):
    config = json.loads((batch_dir / "scan_config.json").read_text(encoding="utf-8-sig"))
    with (batch_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "status", "b", "lambda_prob", "seed", "repeat_index", "repeats",
        "tail_mean_cooperation_rate", "delta_tail_mean_cooperation_rate",
        "control_tail_mean_cooperation_rate", "final_cooperation_rate",
        "delta_population_cumulative_mean_payoff",
        "delta_eligible_cumulative_mean_payoff",
        "delta_ineligible_cumulative_mean_payoff",
        "tail_start_round", "tail_end_round",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("summary.csv lacks fields required for the repeat report.")
    if (len(rows) != 280 or config.get("completed_runs") != 280
            or config.get("status") != "completed"
            or any(row["status"] != "completed" for row in rows)):
        raise ValueError("The presentation requires the completed 280-run scan.")
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["b"]), float(row["lambda_prob"])].append(row)
    expected = {(b, probability) for b in (1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.2)
                for probability in (0.0, 0.1, 0.5, 1.0)}
    if set(grouped) != expected:
        raise ValueError("This presentation expects the original 28 parameter settings.")
    for key, group in grouped.items():
        if (len(group) != 10 or {int(row["seed"]) for row in group} != set(range(42, 52))
                or {int(row["repeat_index"]) for row in group} != set(range(1, 11))
                or {int(row["repeats"]) for row in group} != {10}):
            raise ValueError(f"Incomplete or duplicate repeated runs at {key}.")
        group.sort(key=lambda row: int(row["seed"]))
    if {(int(row["tail_start_round"]), int(row["tail_end_round"]))
            for row in rows} != {(901, 1000)}:
        raise ValueError("The presentation expects the tail window 901–1000.")
    for row in rows:
        expected_delta = (float(row["tail_mean_cooperation_rate"])
                          - float(row["control_tail_mean_cooperation_rate"]))
        if not math.isclose(float(row["delta_tail_mean_cooperation_rate"]),
                            expected_delta, abs_tol=1e-12):
            raise ValueError("Recorded tail-cooperation paired differences do not match.")
    if any(float(row["tail_mean_cooperation_rate"]) != 0
           for row in rows if float(row["b"]) >= 1.04):
        raise ValueError("The b≥1.04 observation changed; update the report narrative.")
    return config, grouped


def values(grouped, b, probability, field, scale=1.0):
    return [scale * float(row[field]) for row in grouped[b, probability]]


def mean_sd(grouped, b, probability, field, scale=1.0, signed=False):
    data = values(grouped, b, probability, field, scale)
    average, deviation = statistics.mean(data), statistics.stdev(data)
    return f"{average:+.2f} ± {deviation:.2f}" if signed else f"{average:.2f} ± {deviation:.2f}"


def cooperation(grouped, b, probability):
    return mean_sd(grouped, b, probability, "tail_mean_cooperation_rate", 100) + "%"


def paired_cooperation(grouped, b, probability):
    return mean_sd(grouped, b, probability, "delta_tail_mean_cooperation_rate", 100, True)


def payoff(grouped, b, probability, group):
    return mean_sd(grouped, b, probability, f"delta_{group}_cumulative_mean_payoff", signed=True)


def positive_count(grouped, b, probability, field):
    return sum(value > 0 for value in values(grouped, b, probability, field))


def build_deck(report_dir: Path, batch_dir: Path):
    config, grouped = read_inputs(batch_dir)
    figure_dir = report_dir / "figures"
    for name in FIGURE_NAMES:
        if not (figure_dir / name).is_file():
            raise FileNotFoundError(f"Missing report figure: {figure_dir / name}")

    fixed, scan = config["fixed_parameters"], config["scan_values"]
    size = int(fixed["size"])
    access = float(scan["llm_access_fraction"][0])
    steps = int(config["steps"])
    b_values = "、".join(f"{value:g}" for value in scan["b"])
    lambda_values = " / ".join(f"{value:g}" for value in scan["lambda_prob"])
    source_label = f"{REPORT_DATE}  ·  {batch_dir.name}  ·  28 组 × 10 次配对重复"
    field = "delta_tail_mean_cooperation_rate"
    positive_half = positive_count(grouped, 1.02, 0.5, field)
    positive_full = positive_count(grouped, 1.02, 1, field)
    survivors = positive_count(grouped, 1.03, 1, "final_cooperation_rate")
    if survivors != 4:
        raise ValueError("The b=1.03, λ=1 survivor count changed; update the narrative.")
    tradeoff_counts = {
        key: sum(float(row["delta_eligible_cumulative_mean_payoff"]) > 0
                 and float(row["delta_population_cumulative_mean_payoff"]) < 0
                 for row in grouped[key])
        for key in ((1.04, 1.0), (1.05, 0.5))
    }

    notes = [
        (
            "这次把每组参数从一次运行扩展为十次重复，其他实验设置保持一致。"
            f"共 28 组参数、280 次实验运行，每次都有对应的纯 Fermi 对照。"
            f"仍是 {size}×{size} 周期格点、四近邻相互作用、同步更新，演化 {steps} 步；"
            f"初始合作概率为 {float(fixed['initial_cooperation']):.0%}，"
            f"固定 {access:.0%} 的位置有咨询资格，Fermi 噪声 k={float(scan['k'][0]):g}。"
            f"扫描 b={b_values}，λ={lambda_values}；λ 是资格位置采用建议的概率，λ=0 是纯 Fermi。"
            "每组使用 seed=42 到 51，旧的 seed=42 只算在这十个样本中，不再额外加一次。"
            "合作率先在单次运行内取第 901–1000 步平均；干预效果先在同 seed 内计算实验减配对对照，"
            "随后在十次重复间汇总。均值旁的 SD 是样本标准差，描述种子间离散程度，不是置信区间。"
            "格点与时间步不充当独立重复。配对初态和资格位置一致，但干预后随机数消费可能不同。"
        ),
        (
            "合作率热图现在同时展示十次重复的均值与样本标准差。"
            f"b=1.02 时，纯 Fermi 的末段合作率为 {cooperation(grouped, 1.02, 0)}，"
            f"λ=0.5 为 {cooperation(grouped, 1.02, 0.5)}，"
            f"λ=1 为 {cooperation(grouped, 1.02, 1)}。"
            "因此旧的单种子排序不能直接当作总体结论；重复均值下，λ=0.5 在此 b 的结果高于 λ=1，"
            "但这里只作描述性比较，不据此认定最优概率。"
            f"b=1.03、λ=1 时末段平均为 {cooperation(grouped, 1.03, 1)}，且种子间差异较大。"
            "本批 b≥1.04 的所有 160 次实验，末 100 步合作率均为零。"
            "末段均值是本次有限时间观测，不等同于已经达到长期稳态。"
        ),
        (
            "实线是每个时间步上十次运行的平均合作率，阴影是同一时间步的最小值到最大值。"
            "阴影表达实际观测到的跨种子范围，不是标准差带，也不是置信区间。"
            "从完整轨迹可以同时看合作的衰减速度、暂时维持及种子之间的分化。"
            "均值曲线本身不是一条实际运行轨迹；平均曲线出现平滑的小合作率，也可能只由少数运行贡献。"
            "即使终点或末段合作率相同，前期合作持续时间仍会改变全程累计收益。"
        ),
        (
            "这一页直接看同 seed 配对的末段合作率差，单位是百分点。"
            "每个点对应一次实验减去它的纯 Fermi 对照；大点及误差线表示十个配对差的均值加减样本标准差。"
            f"b=1.02、λ=0.5 的差为 {paired_cooperation(grouped, 1.02, 0.5)} 个百分点，"
            f"{positive_half}/10 次高于对照；λ=1 为 {paired_cooperation(grouped, 1.02, 1)} 个百分点，"
            f"{positive_full}/10 次高于对照。"
            "这里既能看到平均改善，也能看到有些种子未改善，因此不将平均值解释为每次都有效。"
            "标准差不是显著性检验，正向次数也是描述性计数。"
            "不同 λ 共用相同的 seed 集合和对应对照，不能把各参数组或全部点直接混成独立样本。"
        ),
        (
            "收益指标先在每个 seed 内计算实验减去配对 Fermi 对照，再汇总十次差值的均值与样本标准差。"
            "累计第 1–1000 步每个位置的轮末收益，再按组内人数取人均，未计入初始 round 0。"
            "全体、资格组和无资格组由初始化时的固定位置划分，不按实际咨询次数分组。"
            "正值表示该组相同位置的人均累计收益高于对照，负值表示低于对照。"
            "配对差包含邻居传播和整条演化路径改变的作用，不能解释为一次咨询的直接收益。"
            "即使高 b 下末段合作为零，前期轨迹仍可能导致累计收益有差异。"
        ),
        (
            "这一页专门复核上次 seed=42 的两个资格组略获益、全体损失的案例，避免沿用单次运行的结论。"
            f"在 b=1.04、λ=1，十次重复的资格组差为 {payoff(grouped, 1.04, 1, 'eligible')}，"
            f"无资格组为 {payoff(grouped, 1.04, 1, 'ineligible')}，"
            f"全体为 {payoff(grouped, 1.04, 1, 'population')}。"
            f"在 b=1.05、λ=0.5，资格组为 {payoff(grouped, 1.05, 0.5, 'eligible')}，"
            f"无资格组为 {payoff(grouped, 1.05, 0.5, 'ineligible')}，"
            f"全体为 {payoff(grouped, 1.05, 0.5, 'population')}。"
            "这两个参数设置下，十次均值中的三类群体均为同方向，且离散程度较大。"
            f"原先的资格组获益而全体损失，在两组中分别出现 {tradeoff_counts[1.04, 1.0]}/10 和"
            f" {tradeoff_counts[1.05, 0.5]}/10 次，都是 seed=42。"
            "因此可以说单次案例确实存在，但当前重复结果不足以把它概括为稳定的群体收益分化规律。"
        ),
        (
            "这次最主要的更新有三点。"
            f"第一，b=1.02、λ=0.5 相对配对对照平均增加 {paired_cooperation(grouped, 1.02, 0.5)} 个百分点，"
            f"{positive_half}/10 次为正；应汇报效应和离散，而不只引用 seed=42。"
            f"第二，b=1.03、λ=1 的终点仍有合作为 {survivors}/10 次，"
            "本批 b≥1.04 的所有重复在末 100 步均无合作，说明参数区间和有限时间内的存续需要分开讨论。"
            "第三，上次两个收益分化例子在十次均值中都不再呈现组间相反方向，不能直接推广。"
            "这轮重复仍共用同一张建议表，评估的是固定建议表下的随机种子差异，"
            "并未覆盖重新生成建议表的随机性。后续可独立重复建议表，并对 b≈1.03 的过渡区域延长运行。"
            "本报告不做显著性判定，也不把末段均值或终点存续称为长期稳态结论。"
        ),
        (
            "附录把第 1000 步仍存在至少一个合作者的运行次数列出来，分母都是十次。"
            "b=1.01 和 1.02 的全部 λ 都是 10/10；"
            f"b=1.03 只有 λ=1 为 {survivors}/10，其他 λ 均为 0/10；b≥1.04 均为 0/10。"
            "终点存续和末 100 步平均合作率是不同指标；运行可能在末 100 步期间尚有合作，但在终点已全 D。"
            "这是有限时间实验中的计数，不能直接当作已知的长期存活概率。"
        ),
        (
            "空间附录沿用 seed=42 的实际格点作为示例，展示第 1000 步的空间状态。"
            "这些图不是十次运行的平均格点，也不代表典型种子；seed=42 已经包含在前面的十次统计中。"
            "格点图适合观察团簇形态或全 D 状态，但单个快照不能证明形成机制或长期稳定性。"
            "结论以前面的十次统计、完整动态和配对收益为依据。"
        ),
    ]

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(16), Inches(9)
    prs.core_properties.title = "LLM 建议与空间合作：每组 10 次重复"
    prs.core_properties.subject = "28 组参数 × 10 次重复的配对统计与旧案例复核"
    prs.core_properties.author = "LLM_Advice"
    prs.core_properties.keywords = "空间合作,LLM,Fermi,10次重复,配对差,样本标准差"

    slide = new_slide(prs)
    header(slide, f"组会汇报  /  10 次重复更新  /  {REPORT_DATE}",
           "LLM 建议与空间合作：每组 10 次重复")
    text(slide, 0.94, 2.10, 14.1, 0.65,
         "同时看平均效果、种子间离散，以及上次单次运行结论能否保持。", 24, MUTED)
    rectangle(slide, 0.9, 3.00, 8.75, 3.92, PALE)
    text(slide, 1.25, 3.25, 8.0, 0.5, "实验设置保持一致", 24, NAVY, True)
    text(slide, 1.25, 3.95, 8.0, 2.8,
         f"{size} × {size} 格点｜周期边界｜四近邻｜同步更新\n"
         f"演化 {steps:,} 步；初始合作概率 {float(fixed['initial_cooperation']):.0%}；k = {float(scan['k'][0]):g}\n"
         f"固定 {access:.0%} 位置有咨询资格（{round(size * size * access)} / {size * size}）\n"
         f"λ = {lambda_values}；λ = 0 为纯 Fermi\n"
         f"b = {b_values}", 22)
    rectangle(slide, 10.00, 3.00, 5.10, 3.92, NAVY)
    text(slide, 10.40, 3.28, 4.25, 0.7, "28 组 × 10 次", 32, WHITE, True)
    text(slide, 10.40, 4.22, 4.25, 2.4,
         "280 次实验 + 对应 Fermi 对照\nseed = 42–51\n末段均值：第 901–1000 步\n先算 seed 配对差，再跨 seed 汇总", 20, WHITE)
    text(slide, 0.95, 7.28, 14.1, 0.7,
         "均值 ± SD：SD 表示种子间离散；共用一张建议表；不作显著性或稳态判断。", 20, TEAL, True)
    footer(slide, 1, source_label)
    add_notes(slide, notes[0])
    for index, name in enumerate(FIGURE_NAMES[:5], start=1):
        figure_slide(prs, figure_dir / name, notes[index])

    slide = new_slide(prs)
    header(slide, "10 次重复后的更新", "平均效果更清楚，单次案例需要修正")
    conclusions = (
        ("01", "b = 1.02 的合作平均改善，但有种子差异",
         f"λ = 0.5：配对差 {paired_cooperation(grouped, 1.02, 0.5)} 个百分点；{positive_half}/10 次高于对照。"),
        ("02", "b = 1.03 存在少量有限时间存续",
         f"λ = 1 终点仍有合作：{survivors}/10；本批 b ≥ 1.04 的全部重复，末 100 步合作均为 0。"),
        ("03", "上次的两例收益分化没有体现在重复均值中",
         "两例的三类群体平均收益差均同方向；资格组获益、全体损失各仅出现 1/10 次。"),
    )
    for index, (number, heading, body) in enumerate(conclusions):
        y = 2.22 + index * 1.46
        rectangle(slide, 0.9, y, 14.2, 1.21, PALE)
        text(slide, 1.18, y + 0.22, 0.8, 0.65, number, 29, TEAL, True)
        text(slide, 2.05, y + 0.15, 12.5, 0.42, heading, 23, NAVY, True)
        text(slide, 2.05, y + 0.69, 12.5, 0.4, body, 18, INK)
    text(slide, 0.98, 6.97, 14.0, 0.55,
         "下一步：独立生成多张建议表；对 b ≈ 1.03 过渡区域延长观测。", 23, TEAL, True)
    text(slide, 0.98, 7.65, 14.0, 0.42,
         "解释边界：固定建议表下的 10 次 seed 重复；SD 不是置信区间，配对差包含邻居传播。", 18, MUTED)
    footer(slide, 7, source_label)
    add_notes(slide, notes[6])
    figure_slide(prs, figure_dir / FIGURE_NAMES[5], notes[7])
    figure_slide(prs, figure_dir / FIGURE_NAMES[6], notes[8])

    report_dir.mkdir(parents=True, exist_ok=True)
    destination = report_dir / "group_meeting_repeats10.pptx"
    prs.save(destination)
    titles = (
        "每组 10 次重复：实验设置", "末 100 步合作率：均值与 SD", "合作演化：均值与观测范围",
        "同 seed 配对合作差", "相对配对 Fermi 的累计收益差", "上次两个收益案例的重复复核",
        "重复实验后的三个更新", "附录：终点仍有合作的次数", "附录：seed=42 空间状态示例",
    )
    script = report_dir / "组会讲稿.txt"
    blocks = [
        f"组会讲稿｜{REPORT_DATE}｜每组 10 次重复\n数据来源：{batch_dir.name}\n"
        "建议：前 7 页主讲，后 2 页按需补充。图中 ± 均为样本标准差；动态阴影为 min–max。"
    ]
    blocks.extend(f"第 {index} 页｜{title}\n{note}"
                  for index, (title, note) in enumerate(zip(titles, notes), start=1))
    script.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return destination, script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True,
                        help="Scan containing summary.csv and scan_config.json")
    args = parser.parse_args()
    for path in build_deck(args.report_dir.resolve(), args.batch_dir.resolve()):
        print(path)


if __name__ == "__main__":
    main()
