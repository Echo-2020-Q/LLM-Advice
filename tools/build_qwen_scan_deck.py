#!/usr/bin/env python3
"""Build the Qwen scan presentation and Chinese speaker notes.

Usage:
    python tools/build_qwen_scan_deck.py --report-dir reports/qwen_20261027_repeats2 \
        --batch-dir main/SPD/results/scan_20261027_122530_2次Qwen微扫描

This narrative targets the completed 308-setting, two-seed, 10,000-step scan.
Source experiments are read only. Ten 16:9 figures are required under figures/.
The opening and conclusion remain editable; all twelve slides contain notes.
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


REPORT_DATE = "2026-10-27"
B_VALUES = (1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.2)
LAMBDA_VALUES = (0.0, 0.1, 0.5, 1.0)
P_VALUES = tuple(i / 10 for i in range(11))
MAIN_FIGURES = (
    "01_cooperation.png", "02_coverage.png", "04_welfare.png",
    "06_equal_intensity.png", "07_dynamics.png", "10_horizon.png",
)
APPENDIX_FIGURES = (
    "03_repeat_spread.png", "05_group_welfare.png",
    "08_advisor.png", "09_spatial.png",
)
COOP = "tail_mean_cooperation_rate"
PAYOFF = "delta_population_time_mean_payoff"


def read_inputs(batch_dir: Path):
    config = json.loads((batch_dir / "scan_config.json").read_text(encoding="utf-8-sig"))
    with (batch_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "status", "b", "lambda_prob", "llm_access_fraction", "seed", "steps",
        "repeat_index", "repeats", COOP, PAYOFF,
        "control_tail_mean_cooperation_rate", "delta_tail_mean_cooperation_rate",
        "population_time_mean_payoff", "control_population_time_mean_payoff",
        "tail_start_round", "tail_end_round", "early_start_round", "early_end_round",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("summary.csv is missing fields required for this report.")
    if (len(rows) != 616 or config.get("completed_runs") != 616
            or config.get("status") != "completed"
            or any(row["status"] != "completed" for row in rows)):
        raise ValueError("The deck requires the completed 616-run scan.")
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["b"]), float(row["lambda_prob"]),
                float(row["llm_access_fraction"])].append(row)
        if (int(row["steps"]) != 10000
                or (int(row["tail_start_round"]), int(row["tail_end_round"])) != (9001, 10000)
                or (int(row["early_start_round"]), int(row["early_end_round"])) != (1, 200)):
            raise ValueError("This deck expects 10,000 steps, tail 9001–10000, early 1–200.")
        for experimental, control, delta in (
            (COOP, "control_tail_mean_cooperation_rate", "delta_tail_mean_cooperation_rate"),
            ("population_time_mean_payoff", "control_population_time_mean_payoff", PAYOFF),
        ):
            if not math.isclose(float(row[experimental]) - float(row[control]),
                                float(row[delta]), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Invalid paired difference in run {row.get('run_id')}: {delta}")
    expected = {(b, lam, p) for b in B_VALUES for lam in LAMBDA_VALUES for p in P_VALUES}
    if set(grouped) != expected:
        raise ValueError("This deck requires the expected 7 × 4 × 11 parameter grid.")
    for key, group in grouped.items():
        if (len(group) != 2 or {int(row["seed"]) for row in group} != {42, 43}
                or {int(row["repeat_index"]) for row in group} != {1, 2}
                or {int(row["repeats"]) for row in group} != {2}):
            raise ValueError(f"Missing or duplicate seed runs at {key}.")
        group.sort(key=lambda row: int(row["seed"]))
    fixed = config["fixed_parameters"]
    if (int(config["steps"]) != 10000 or int(config["repeats"]) != 2
            or int(fixed["size"]) != 50 or fixed["boundary"] != "periodic"
            or fixed["update_mode"] != "synchronous"
            or config["scan_values"]["k"] != [0.1]):
        raise ValueError("Experiment settings differ from the presentation narrative.")
    sources = config["advisor_sources"]
    models = {source["model"].replace("\\", "/").rsplit("/", 1)[-1] for source in sources}
    if models != {"Qwen3.8-27B-f16.gguf"}:
        raise ValueError("This presentation expects the recorded Qwen3.8 model.")
    if any(source["n"] != 2 or source["case_count"] != 512 or source["b"] is not None
           for source in sources):
        raise ValueError("The recorded advice-table settings differ from the narrative.")
    # These statements are deliberately local to this scan, with checks that
    # prevent the deck from silently reusing them for a different experiment.
    for b in B_VALUES:
        for lam, p in ((0.5, 0.9), (0.5, 1.0), (1.0, 1.0)):
            if any(float(row[COOP]) != 1 for row in grouped[b, lam, p]):
                raise ValueError("The all-C observation changed; revise the narrative.")
        if any(float(row[COOP]) >= 1 for row in grouped[b, 1.0, 0.9]):
            raise ValueError("The λ=1, P=0.9 observation changed; revise the narrative.")
    if any(float(row[COOP]) != 0 for row in rows
           if float(row["b"]) >= 1.04 and float(row["lambda_prob"]) == 0.1):
        raise ValueError("The low-frequency, high-b observation changed.")
    return config, grouped


def values(grouped, key, field=COOP, scale=1.0):
    return [scale * float(row[field]) for row in grouped[key]]


def average(grouped, key, field=COOP, scale=1.0):
    return statistics.mean(values(grouped, key, field, scale))


def cooperation(grouped, key):
    return f"{average(grouped, key, COOP, 100):.2f}%"


def payoff(grouped, key):
    return f"{average(grouped, key, PAYOFF):+.3f}"


def make_notes(config, grouped):
    worst_key = max(grouped, key=lambda key: max(values(grouped, key)) - min(values(grouped, key)))
    worst_values = values(grouped, worst_key, COOP, 100)
    spread = max(worst_values) - min(worst_values)
    equal_keys = ((1.02, 1.0, 0.1), (1.02, 0.5, 0.2), (1.02, 0.1, 1.0))
    equal_c = "、".join(cooperation(grouped, key) for key in equal_keys)
    equal_y = "、".join(payoff(grouped, key) for key in equal_keys)
    notes = [
        (
            "本次汇报 Qwen3.8 建议表对应的参数扫描。配置文件记录的模型文件名是 "
            "Qwen3.8-27B-f16.gguf。7 个 b、4 个 λ、11 个 P 组成 308 组参数，"
            "每组分别用 seed=42 和 43 运行，共完成 616 次实验及其配对纯 Fermi 对照。"
            "P 表示初始化时固定获得建议资格的位置比例，λ 表示资格位置更新时采用建议的概率。"
            "格点为 50×50，周期边界、四近邻相互作用、同步更新，初始合作概率 0.5，Fermi 的 K=0.1。"
            "每次演化 10000 步；末段合作率取第 9001–10000 步平均；前期收益窗口为第 1–200 步。"
            "本批只包含两个仿真种子，图中的均值和范围是这两个观测的描述，"
            "不是置信区间，不能说明结果已经稳定或具有显著性。"
            "所有仿真使用同一张固定建议表，每局面两条 LLM 回答与每参数两次仿真是两种不同的重复。"
            "对照与实验同初态、同资格位置；干预后随机数的消费不保证逐事件相同。"
        ),
        (
            "四张热图分别固定 λ，横轴为资格覆盖率 P，纵轴为 b，颜色是两次运行的末 1000 步合作率均值。"
            "λ=0 和 P=0 都不使用建议，是理解其余区域的基线。"
            "在本批所有 7 个 b 中，P=0.9、λ=0.5 的两个种子末段合作率均为 100%；"
            "P=1、λ=0.5 或 1 时也都是 100%。"
            "低频率 λ=0.1 则不同：b≥1.04 的所有 P、两个种子末段合作率都为零。"
            "因此合作的维持同时依赖覆盖范围、使用频率和 b，不能仅凭更广覆盖作单因素推广。"
            "100% 的说法限定于这些已完成的运行与最后 1000 步，不直接推断无限时间极限。"
        ),
        (
            "这页按 b 展开覆盖率曲线，保留两个种子的观测范围。"
            "重点是 λ=0.5 与 λ=1 的排序并不固定。"
            f"例如 b=1.2、P=0.9 时，λ=0.5 的末段均值为 {cooperation(grouped, (1.2, 0.5, 0.9))}，"
            f"λ=1 为 {cooperation(grouped, (1.2, 1.0, 0.9))}。"
            "同一覆盖率下每次都采用建议，并没有比一半概率采用建议得到更高的末段合作。"
            "这给出需要复核的非单调关系，并不代表 λ=0.5 是所有参数条件的最优值。"
            "连接离散扫描点的线条仅辅助阅读；未扫描区域和精确阈值不能从连线直接确定。"
        ),
        (
            "收益图展示三个非零 λ 下的全体人均每步收益配对差。"
            "每个种子先计算实验减去对应纯 Fermi 对照，再对两个差值取平均。"
            "收益累计第 1–10000 步、不含初始第 0 步，除以人数和 10000 得到每人每步的量纲。"
            "这样不会因为增加总步数而机械地放大累计收益数值。"
            f"例如 b=1.2、P=0.9，λ=0.5 的配对收益差为 {payoff(grouped, (1.2, 0.5, 0.9))}，"
            f"λ=1 为 {payoff(grouped, (1.2, 1.0, 0.9))}。"
            "末段合作与全程收益不是同一指标：较早形成合作的运行，也可能积累更高的全程平均收益。"
            "配对差包含邻居传播及整个路径改变，不能解释成单次使用建议的直接收益。"
        ),
        (
            "这页比较 P×λ 都等于 0.1 的三种配置：P=0.1、λ=1，P=0.2、λ=0.5，"
            "以及 P=1、λ=0.1，分别对应少数位置频繁使用建议和更多位置低频使用建议。"
            "P×λ 是每个个体更新获得建议的平均概率尺度，不代表每步实际使用次数完全相同，"
            "也没有消除固定资格位置带来的空间差异。"
            f"b=1.02 时，按上面顺序，末段合作均值为 {equal_c}；"
            f"全体每人每步收益配对差为 {equal_y}。"
            "在 b=1.01、1.02、1.03 的这三种配置中，广覆盖低频使用建议的末段合作均值更高。"
            "b≥1.04 的这三种配置末段合作都为零，因此不能把这项排序推广到全部 b，"
            "也不能据此宣称覆盖率始终比频率更重要。"
        ),
        (
            "动态页选取 b=1.02、1.1、1.2，与 P=0.2、0.5、0.9 组成九个观察窗口。"
            "各曲线展示相同参数下两个种子的平均轨迹，范围仅对应这两个种子的实际观测。"
            "读图时同时观察合作的下降、恢复和形成高合作状态的时间；均值曲线本身不是一次真实运行。"
            "10000 步与末 1000 步窗口比短时观察更完整，但并不自动证明已经达到长期稳态。"
            "如果目标是判断下一轮是否仍需延长，应检查具体运行尾部是否继续漂移，而不是只看终点或均值。"
        ),
        (
            "这一页直接回答延长到 10000 步的价值。本批有 4 次实验在第 5000 步仍有合作，"
            "到第 10000 步时已为全 D，短时观察会遗漏这种后期灭绝。"
            "第一组是 b=1.05、λ=0.5、P=0.4：seed=42 和 43 在第 4001–5000 步的平均合作率"
            "分别为 9.7866% 和 10.80708%，之后分别从第 6315 和 9057 步起持续全 D 至终点。"
            "seed=43 在最后 1000 步的平均合作率仍为 0.0316%，因为第 9001–9056 步还有少量合作；"
            "因此末段均值大于零不能直接当作终点仍有合作。"
            "第二组是 b=1.03、λ=1、P=0.1：两个种子的第 4001–5000 步合作率均值为 13.07548%，"
            f"最后 1000 步均值为 {cooperation(grouped, (1.03, 1.0, 0.1))}。"
            "其中 seed=43 从第 8169 步起持续全 D，seed=42 在第 10000 步仍有 4.52% 合作者。"
            "这也说明均值曲线可能同时混合存续和灭绝两类路径。"
            "已有 10000 步确实揭示了额外动态，但每组仍只有两个种子，不估计置信区间，"
            "也不将有限时间内的终点存续解释成长期稳定。"
        ),
        (
            "这轮可以形成三个具体观察。第一，高覆盖与适当采用频率结合时，本批能在较大的 b 下维持高合作。"
            "P=0.9、λ=0.5 在所有已扫 b 的两个种子末段都为 100%。"
            "第二，增加 λ 并非总有利，P=0.9 时 λ=1 的末段均值反而低于 λ=0.5。"
            "第三，相同 P×λ 下覆盖方式仍有差异，但广覆盖低频更好的观察限定于 b=1.01、1.02、1.03。"
            f"本批最大的种子差出现在 b={worst_key[0]:g}、λ={worst_key[1]:g}、P={worst_key[2]:g}，"
            f"两次末段合作差 {spread:.2f} 个百分点，其中一次为零、另一次仍有合作。"
            "下一轮优先给这些分歧与转变区域补种子重复，并在 P=0.8–0.9、λ=0.5 与 1 的区域增加覆盖点。"
            "等 P×λ 的三组配置也值得配对复核；固定建议表的结果随后再用独立生成的建议表检查。"
            "现阶段以探索性描述为主，不从两个种子估计可靠的置信区间或作显著性判断。"
        ),
        (
            "附录展示同一参数下两个种子的末段合作率绝对差，单位为百分点。"
            f"最大差为 {spread:.3f} 个百分点，位置是 b={worst_key[0]:g}、λ={worst_key[1]:g}、P={worst_key[2]:g}；"
            f"seed=42 与 43 分别为 {worst_values[0]:.3f}% 和 {worst_values[1]:.3f}%。"
            "这个指标直接指出这两个观测分歧较大的区域，但差值小不能证明总体方差小。"
            "两次都为零或两次都为 100% 也只代表本次观察一致，不能替代更多独立种子的验证。"
        ),
        (
            "分组收益附录固定 λ=0.5，比较全体、资格组和无资格组相对其配对纯 Fermi 对照的收益变化。"
            "三个指标都是全程每人每步收益差，先按同 seed 配对后再取两次均值。"
            "资格组由初始化时的位置决定，并不是按这次运行实际咨询过的次数分类。"
            "P=0 时资格组为空，P=1 时无资格组为空，相应人均指标保留为 NA；NA 不能填成零。"
            "全体收益差由两个存在的群体按人数加权得到，不能直接平均两个组的人均指标。"
            "组间差异包含空间位置与邻居传播作用，不是控制其他因素后的资格直接因果效应。"
        ),
        (
            "建议表包含全部 512 个 3×3 二元局面，每局面两条回答，共 1024 条。"
            "其中合作建议 583 条，占 56.9336%；两条回答一致的局面为 433/512，即 84.5703%。"
            "中心个体原为合作时，合作建议为 343/512，即 66.9922%；"
            "中心原为背叛时为 240/512，即 46.8750%。"
            "这些比例是按表内局面及回答等权统计，不是模拟过程实际访问局面加权后的合作建议比例。"
            "所有 b 复用同一张表，记录的 b 字段为 null，建议输入没有当前扫描的具体 b。"
            "运行时针对当前局面，从对应的两条固定回答中有放回抽样，仿真重复不会重新生成回答。"
            "因此本次两个种子主要反映固定表下的演化随机性，尚未覆盖独立重新生成建议表的不确定性。"
        ),
        (
            "空间附录使用 seed=42 的实际末态示例，对比 b=1.02、1.1、1.2，"
            "λ=0.5 或 1，以及 P=0.2 或 0.9 的局面。"
            "每张图都是第 10000 步一个实际格点，不是两个种子平均后的空间图，也不代表典型种子。"
            "这些图辅助辨认合作团簇和全合作状态；形成过程应联系动态页，普遍性应联系两个种子统计。"
            "单张空间快照不支持机制或长期稳定性的独立证明。"
        ),
    ]
    return notes, worst_key, spread


def build_deck(report_dir: Path, batch_dir: Path):
    # The WSL cwd resolves to this repository's older mixed-case spelling;
    # writes are permitted through its verified lowercase alias. Keep output
    # paths lexical and only rebase when both roots identify the same directory.
    report_dir = Path(report_dir).absolute()
    repository_root = Path(__file__).resolve().parents[1]
    lowercase_root = Path(str(repository_root).lower())
    if (lowercase_root.is_dir() and lowercase_root.samefile(repository_root)
            and report_dir.is_relative_to(repository_root)):
        report_dir = lowercase_root / report_dir.relative_to(repository_root)
    config, grouped = read_inputs(batch_dir)
    figure_dir = report_dir / "figures"
    for name in MAIN_FIGURES + APPENDIX_FIGURES:
        if not (figure_dir / name).is_file():
            raise FileNotFoundError(f"Missing report figure: {figure_dir / name}")
    notes, worst_key, spread = make_notes(config, grouped)
    title = "Qwen3.8 建议：覆盖率与采用频率如何影响合作？"
    source_label = f"{REPORT_DATE}  ·  {batch_dir.name}  ·  每参数 2 个种子；固定建议表"
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(16), Inches(9)
    prs.core_properties.title = title
    prs.core_properties.subject = "308 组参数、616 次运行；10000 步 Qwen 建议扫描"
    prs.core_properties.author = "LLM_Advice"
    prs.core_properties.keywords = "Qwen3.8,空间合作,资格覆盖率,采用概率,2次重复,配对收益"

    slide = new_slide(prs)
    header(slide, f"组会汇报  /  Qwen 参数扫描  /  {REPORT_DATE}", title)
    text(slide, 0.94, 2.10, 14.1, 0.60,
         "P 决定多少位置有建议资格；λ 决定这些位置多频繁采用建议。", 24, MUTED)
    rectangle(slide, 0.9, 3.00, 8.75, 3.92, PALE)
    text(slide, 1.25, 3.25, 8.0, 0.5, "扫描设置", 24, NAVY, True)
    text(slide, 1.25, 3.95, 8.0, 2.75,
         "50 × 50 格点｜周期边界｜四近邻｜同步更新\n"
         "10,000 步；初始合作概率 50%；K = 0.1\n"
         "b = 1.01 / 1.02 / 1.03 / 1.04 / 1.05 / 1.1 / 1.2\n"
         "λ = 0 / 0.1 / 0.5 / 1；P = 0–1，步长 0.1\n"
         "模型记录：Qwen3.8-27B-f16.gguf", 21)
    rectangle(slide, 10.0, 3.0, 5.1, 3.92, NAVY)
    text(slide, 10.4, 3.28, 4.25, 0.7, "308 组 × 2 次", 32, WHITE, True)
    text(slide, 10.4, 4.22, 4.3, 2.35,
         "616 次实验 + 对应 Fermi 对照\nseed = 42 / 43\n末段均值：第 9001–10000 步\n前期窗口：第 1–200 步", 20, WHITE)
    text(slide, 0.95, 7.23, 14.1, 0.85,
         "所有运行复用一张建议表。两次均值与范围用于描述，不是置信区间或稳定性证据。", 20, TEAL, True)
    footer(slide, 1, source_label)
    add_notes(slide, notes[0])
    for index, name in enumerate(MAIN_FIGURES, start=1):
        figure_slide(prs, figure_dir / name, notes[index])

    slide = new_slide(prs)
    header(slide, "当前观察与下一步", "覆盖与频率共同起作用，采用更多并非总是更好")
    conclusions = (
        ("01", "高覆盖配合 λ = 0.5，出现全合作末段",
         "P = 0.9：所有已扫 b、两个种子，末 1000 步合作率均为 100%。"),
        ("02", "相同覆盖率下，λ = 1 未必优于 λ = 0.5",
         f"b = 1.2，P = 0.9：λ = 0.5 为 100%，λ = 1 为 {cooperation(grouped, (1.2, 1.0, 0.9))}。"),
        ("03", "相同 P×λ 下，广覆盖低频在低 b 的三组中更好",
         "观察限定于 b = 1.01 / 1.02 / 1.03；b ≥ 1.04 的三种配置末段均无合作。"),
    )
    for index, (number, heading, body) in enumerate(conclusions):
        y = 2.22 + index * 1.46
        rectangle(slide, 0.9, y, 14.2, 1.21, PALE)
        text(slide, 1.18, y + 0.22, 0.8, 0.65, number, 29, TEAL, True)
        text(slide, 2.05, y + 0.15, 12.5, 0.42, heading, 23, NAVY, True)
        text(slide, 2.05, y + 0.69, 12.5, 0.40, body, 19, INK)
    text(slide, 0.98, 6.97, 14.0, 0.55,
         "下一步：优先增加转变区域的种子重复；细化 P = 0.8–0.9；独立重建建议表。", 21, TEAL, True)
    text(slide, 0.98, 7.64, 14.0, 0.45,
         f"最大种子差 {spread:.2f} 个百分点（b={worst_key[0]:g}，λ={worst_key[1]:g}，P={worst_key[2]:g}）；n=2 尚不足以判断稳定性。",
         18, MUTED)
    footer(slide, 8, source_label)
    add_notes(slide, notes[7])
    for index, name in enumerate(APPENDIX_FIGURES, start=8):
        figure_slide(prs, figure_dir / name, notes[index])

    report_dir.mkdir(parents=True, exist_ok=True)
    destination = report_dir / "qwen_scan_20261027.pptx"
    prs.save(destination)
    titles = (
        "实验设置与统计口径", "b × P 合作率全景", "覆盖率曲线与两次观测范围",
        "全体每人每步收益：相对配对 Fermi 的变化", "相同 P×λ 的不同覆盖方式",
        "代表参数下的完整动态", "从 5000 步延长到 10000 步：后期灭绝与继续衰减", "三个观察与下一步",
        "附录：两种子的末段合作差", "附录：全体、资格组、无资格组收益",
        "附录：Qwen 建议表的条件行为", "附录：seed=42 最终空间状态",
    )
    script = report_dir / "组会讲稿.txt"
    blocks = [
        f"组会讲稿｜{REPORT_DATE}｜Qwen3.8 参数扫描\n数据来源：{batch_dir.name}\n"
        "建议：前 8 页主讲，后 4 页按需补充。图中范围来自两个种子，不是置信区间。\n"
        "模型记录：Qwen3.8-27B-f16.gguf；所有运行复用一张固定建议表。"
    ]
    blocks.extend(f"第 {index} 页｜{slide_title}\n{note}"
                  for index, (slide_title, note) in enumerate(zip(titles, notes), start=1))
    script.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return destination, script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True,
                        help="Scan directory containing summary.csv and scan_config.json")
    args = parser.parse_args()
    for path in build_deck(args.report_dir, args.batch_dir.resolve()):
        print(path)


if __name__ == "__main__":
    main()
