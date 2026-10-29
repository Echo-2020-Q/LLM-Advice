#!/usr/bin/env python3
"""Build the GPT scan / Qwen comparison presentation and Chinese speaker notes.

This report concerns two completed, matched parameter scans and one empirical
advice table per recorded model. Input simulations are never modified.
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


REPORT_DATE = "2026-10-28"
SCAN_DATE = "2026-10-27"
B_VALUES = (1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.2)
LAMBDA_VALUES = (0.0, 0.1, 0.5, 1.0)
P_VALUES = tuple(i / 10 for i in range(11))
SEEDS = (42, 43)
COOP = "tail_mean_cooperation_rate"
PAYOFF = "delta_population_time_mean_payoff"
ROOT = Path(__file__).resolve().parents[1]
MAIN_FIGURES = (
    "01_gpt_cooperation.png", "02_gpt_coverage.png", "03_gpt_welfare.png",
    "04_cooperation_gap.png", "05_payoff_gap.png", "06_dynamics.png", "07_advisor.png",
)
APPENDIX_FIGURES = (
    "08_gpt_repeat_spread.png", "09_group_welfare.png", "10_equal_intensity.png",
    "11_survival.png", "12_spatial.png",
)


def read_scan(batch_dir: Path, expected_model: str):
    config = json.loads((batch_dir / "scan_config.json").read_text(encoding="utf-8-sig"))
    with (batch_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if (len(rows) != 616 or config.get("completed_runs") != 616
            or config.get("status") != "completed"
            or any(row.get("status") != "completed" for row in rows)):
        raise ValueError("The presentation requires two complete 616-run scans.")
    grouped = defaultdict(list)
    indexed = {}
    for row in rows:
        key = (float(row["b"]), float(row["lambda_prob"]),
               float(row["llm_access_fraction"]), int(row["seed"]))
        if key in indexed:
            raise ValueError(f"Duplicate matched key: {key}")
        indexed[key] = row
        grouped[key[:3]].append(row)
        if (int(row["steps"]) != 10000
                or (int(row["tail_start_round"]), int(row["tail_end_round"])) != (9001, 10000)
                or (int(row["early_start_round"]), int(row["early_end_round"])) != (1, 200)):
            raise ValueError("This narrative expects 10,000 steps and the recorded payoff windows.")
        for experimental, control, delta in (
            (COOP, "control_tail_mean_cooperation_rate", "delta_tail_mean_cooperation_rate"),
            ("population_time_mean_payoff", "control_population_time_mean_payoff", PAYOFF),
        ):
            if not math.isclose(float(row[experimental]) - float(row[control]),
                                float(row[delta]), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Invalid paired difference for {key}: {delta}")
    expected = {(b, lam, p, seed) for b in B_VALUES for lam in LAMBDA_VALUES
                for p in P_VALUES for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError("The required 7 × 4 × 11 × 2 scan design is incomplete.")
    for key, group in grouped.items():
        if (len(group) != 2 or {int(row["repeat_index"]) for row in group} != {1, 2}
                or {int(row["repeats"]) for row in group} != {2}):
            raise ValueError(f"Unexpected repeat metadata: {key}")
        group.sort(key=lambda row: int(row["seed"]))
    fixed = config["fixed_parameters"]
    if (fixed["size"] != 50 or fixed["initial_cooperation"] != 0.5
            or fixed["boundary"] != "periodic" or fixed["update_mode"] != "synchronous"
            or config["scan_values"]["k"] != [0.1]
            or config["paired_fermi_control"] is not True):
        raise ValueError("The presentation settings do not match the experiment.")
    models = {source["model"].replace("\\", "/").rsplit("/", 1)[-1]
              for source in config["advisor_sources"]}
    if models != {expected_model}:
        raise ValueError(f"Expected recorded model {expected_model}; found {models}")
    if any(source["n"] != 2 or source["case_count"] != 512 or source["b"] is not None
           for source in config["advisor_sources"]):
        raise ValueError("Unexpected empirical advice-table metadata.")
    return config, dict(grouped), indexed


def read_inputs(gpt_dir: Path, qwen_dir: Path):
    g_config, g_groups, g_runs = read_scan(gpt_dir, "gpt-5.5")
    q_config, q_groups, q_runs = read_scan(qwen_dir, "Qwen3.8-27B-f16.gguf")
    for field in ("fixed_parameters", "scan_values", "steps", "repeats",
                  "tail_average_steps", "early_average_steps", "paired_fermi_control"):
        if g_config[field] != q_config[field]:
            raise ValueError(f"The experiment settings are not matched: {field}")
    prompt_hashes = {source["system_prompt_sha256"]
                     for config in (g_config, q_config) for source in config["advisor_sources"]}
    if len(prompt_hashes) != 1:
        raise ValueError("The two advice sources have different recorded system prompts.")
    for key in g_runs:
        for field in ("initial_cooperation_rate", "control_tail_mean_cooperation_rate",
                      "control_population_time_mean_payoff", "control_final_cooperation_rate"):
            if not math.isclose(float(g_runs[key][field]), float(q_runs[key][field]),
                                rel_tol=1e-11, abs_tol=1e-11):
                raise ValueError(f"Initial/control summary differs across models: {key}, {field}")
    return g_config, q_config, g_groups, q_groups, g_runs, q_runs


def values(grouped, key, field=COOP, scale=1.0):
    return [scale * float(row[field]) for row in grouped[key]]


def average(grouped, key, field=COOP, scale=1.0):
    return statistics.mean(values(grouped, key, field, scale))


def cooperation(grouped, key):
    return f"{average(grouped, key, scale=100):.2f}%"


def payoff(grouped, key):
    return f"{average(grouped, key, field=PAYOFF):+.3f}"


def comparison_counts(g_runs, q_runs):
    counts = {"qwen_higher": 0, "gpt_higher": 0, "equal": 0}
    for key, row in g_runs.items():
        if key[1] == 0 or key[2] == 0:
            continue
        difference = float(row[COOP]) - float(q_runs[key][COOP])
        label = "equal" if abs(difference) <= 1e-10 else (
            "gpt_higher" if difference > 0 else "qwen_higher")
        counts[label] += 1
    if sum(counts.values()) != 420:
        raise ValueError("Expected 420 active matched runs.")
    return counts


def advice_statistics(config):
    source = config["advisor_sources"][0]
    batch_name = source["batch_dir"].replace("\\", "/").rsplit("/", 1)[-1]
    source_path = ROOT / "logs" / "GPT_Multistepsa" / batch_name / "results.csv"
    with source_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    states = defaultdict(list)
    for row in rows:
        if row["status"] != "success" or row["recommendation"] not in {"C", "D"}:
            raise ValueError(f"Unexpected advice response in {source_path}")
        states[row["grid_key"]].append(row["recommendation"])
    if len(rows) != 1024 or len(states) != 512 or any(len(v) != 2 for v in states.values()):
        raise ValueError("Expected 512 states with two empirical answers per state.")
    d4 = [row for row in rows if row["grid_key"][4] == "D"
          and all(row["grid_key"][i] == "C" for i in (1, 3, 5, 7))]
    if len(d4) != 32:
        raise ValueError("Expected 32 responses for center D with four cardinal C neighbors.")
    return {
        "cooperation_pct": 100 * sum(row["recommendation"] == "C" for row in rows) / len(rows),
        "agreement_pct": 100 * sum(v[0] == v[1] for v in states.values()) / len(states),
        "d4_c_pct": 100 * sum(row["recommendation"] == "C" for row in d4) / len(d4),
        "all_c": states["C" * 9], "all_d": states["D" * 9],
    }


def make_notes(g_config, q_config, g_groups, q_groups, g_runs, q_runs):
    counts = comparison_counts(g_runs, q_runs)
    group_counts = {"qwen_both": 0, "gpt_both": 0, "equal_both": 0, "other": 0}
    for b, lam, p in g_groups:
        if lam == 0 or p == 0:
            continue
        differences = [float(g_runs[b, lam, p, seed][COOP])
                       - float(q_runs[b, lam, p, seed][COOP]) for seed in SEEDS]
        if all(value < -1e-10 for value in differences):
            label = "qwen_both"
        elif all(value > 1e-10 for value in differences):
            label = "gpt_both"
        elif all(abs(value) <= 1e-10 for value in differences):
            label = "equal_both"
        else:
            label = "other"
        group_counts[label] += 1
    if group_counts != {"qwen_both": 126, "gpt_both": 0, "equal_both": 80, "other": 4}:
        raise ValueError("The paired parameter-group narrative needs updating.")
    worst_key = max(g_groups, key=lambda key: max(values(g_groups, key)) - min(values(g_groups, key)))
    worst_values = values(g_groups, worst_key, scale=100)
    spread = max(worst_values) - min(worst_values)
    g_advice, q_advice = advice_statistics(g_config), advice_statistics(q_config)
    active_g = [row for key, row in g_runs.items() if key[1] > 0 and key[2] > 0]
    active_q = [row for key, row in q_runs.items() if key[1] > 0 and key[2] > 0]
    extinct_g = sum(float(row["final_cooperation_rate"]) == 0 for row in active_g)
    extinct_q = sum(float(row["final_cooperation_rate"]) == 0 for row in active_q)
    equal_keys = ((1.02, 1.0, 0.1), (1.02, 0.5, 0.2), (1.02, 0.1, 1.0))
    equal_g = "、".join(cooperation(g_groups, key) for key in equal_keys)
    equal_q = "、".join(cooperation(q_groups, key) for key in equal_keys)
    # The strongest qualitative observations are guarded so a future run does
    # not silently inherit this scan-specific narration.
    if not all(float(row[COOP]) == 1 for key in ((1.01, 0.5, 1.0), (1.02, 0.5, 1.0))
               for row in g_groups[key]):
        raise ValueError("The GPT all-cooperation observation changed.")
    if any(values(g_groups, (b, 1.0, 1.0)) != values(g_groups, (1.01, 1.0, 1.0)) for b in B_VALUES):
        raise ValueError("The no-Fermi GPT endpoint observation changed.")
    if not (average(g_groups, (1.02, 1.0, 0.6)) > 0.89
            and average(g_groups, (1.02, 1.0, 0.7)) < 0.02
            and g_advice["d4_c_pct"] == 15.625 and q_advice["d4_c_pct"] == 90.625):
        raise ValueError("The key comparison narrative changed.")
    notes = [
        (
            "本次重点分析新完成的 GPT 扫描，并与此前 Qwen 扫描按相同参数及种子逐一匹配。"
            "配置记录的模型分别为 gpt-5.5 和 Qwen3.8-27B-f16.gguf。每批都有 7 个 b、4 个 λ、11 个 P，"
            "共 308 组参数，每组 seed=42 和 43 两次运行，完成 616 次实验及对应纯 Fermi 对照。"
            "两批的格点、边界、同步更新、初态概率、K、扫描网格、种子和统计窗口一致；"
            "配对初态及 Fermi 对照摘要一致，建议源的系统提示哈希也一致。"
            "模型身份来自本地元数据，本次没有独立核验服务端模型。"
            "P 是初始化时固定有资格采用建议的位置比例；λ 是资格位置更新时采用建议的概率。"
            "末段合作取第 9001–10000 步均值，收益主指标取第 1–10000 步每人每步均值。"
            "每模型本次只对应一张固定建议表，每个 3×3 局面两条回答；仿真重复不会重新生成建议。"
            "因此下面比较的是这两张经验建议表下的演化结果，不能据此给模型作一般能力排名。"
            "每参数两个种子的范围是实际观测范围，不是置信区间；两批总运行数也不能当成独立模型重复。"
        ),
        (
            "先单独看 GPT 的完整参数图。四张热图分别固定 λ，横轴 P，纵轴 b，颜色表示两次末段合作率均值。"
            "λ=0 或 P=0 没有实际建议干预，可作为纯 Fermi 参照。"
            "GPT 并非在所有参数下都无法维持合作：b=1.01 或 1.02，P=1、λ=0.5 时，两次运行末段都为 100%。"
            "但这一高合作区域明显依赖 b、覆盖率与采用频率的组合。"
            f"P=1、λ=1 时，各 b 的两个末段值均为 {values(g_groups, (1.01, 1.0, 1.0), scale=100)[0]:.2f}% 和 "
            f"{values(g_groups, (1.01, 1.0, 1.0), scale=100)[1]:.4f}%，均值 {cooperation(g_groups, (1.01, 1.0, 1.0))}。"
            "此时所有个体都按固定表更新，不再走 Fermi 分支，表中又没有当前扫描的具体 b，"
            "所以合作轨迹对 b 不敏感是该实验设置的结果，并不是收益博弈的一般性质。"
        ),
        (
            "覆盖曲线展示 GPT 在每个 b 下的 P 变化，并保留两个种子的原值及范围。"
            "本批最突出的现象是扩大建议覆盖并不保证更高的合作。"
            f"b=1.02、λ=1 时，P=0.6 的末段合作均值为 {cooperation(g_groups, (1.02, 1.0, 0.6))}，"
            f"增加到 P=0.7 后降到 {cooperation(g_groups, (1.02, 1.0, 0.7))}。"
            "P=0.6 的两个值分别为 93.28976% 和 86.147%；P=0.7 则为 2.10428% 和 1.2274%，"
            "所以这次观察到的下降并非只由一个极端种子拉低均值。"
            "但是 P 仅按 0.1 间隔扫描、每点只有两次，尚不能确定精确转变位置或判断总体稳定性。"
            "另外，b=1.02、P=1 时，λ=0.5 达到全合作，λ=1 只有约 9.86%，采用频率也存在非单调关系。"
        ),
        (
            "GPT 收益图使用全体每人每步收益的配对差：每个实验先减去相同参数和种子的纯 Fermi 对照，再取两次均值。"
            "数值覆盖第 1–10000 步，不含初始第 0 步。正值表示高于对应 Fermi，负值表示低于对应 Fermi。"
            f"b=1.02、λ=1 时，P=0.6 的收益差为 {payoff(g_groups, (1.02, 1.0, 0.6))}，"
            f"P=0.7 为 {payoff(g_groups, (1.02, 1.0, 0.7))}。这与前页的合作下降同时出现，"
            "说明扩大覆盖后的合作损失也对应本批的全程收益损失。"
            f"b=1.01、P=λ=1 时收益差为 {payoff(g_groups, (1.01, 1.0, 1.0))}，"
            "因此不能把有 LLM 建议等同于必然增加群体收益。"
            "收益差也包含邻居传播和整个演化路径变化，不是单次咨询建议的直接收益。"
        ),
        (
            "现在进入两批比较。合作差图统一定义为 GPT 减 Qwen，单位是百分点；负值代表本批 Qwen 表的合作率更高。"
            "先在相同 b、λ、P、seed 下计算差，再对两个种子取平均，避免把独立的组均值当成更多重复。"
            f"排除 P=0 或 λ=0 后有 210 组活跃参数：{group_counts['qwen_both']} 组两个种子都由 Qwen 更高，"
            f"{group_counts['gpt_both']} 组两个种子都由 GPT 更高，{group_counts['equal_both']} 组两次都相同，另有 4 组是混合情况。"
            f"对应 420 对逐次运行，Qwen 更高 {counts['qwen_higher']} 对、GPT 更高 {counts['gpt_higher']} 对、相同 {counts['equal']} 对。"
            "这些计数描述本次扫描网格及两张表，不能当作 420 次独立模型评测或总体胜率。"
            f"例如 b=1.2、P=0.9、λ=0.5，GPT 为 {cooperation(g_groups, (1.2, 0.5, 0.9))}，"
            f"Qwen 为 {cooperation(q_groups, (1.2, 0.5, 0.9))}，形成 100 个百分点的差异。"
        ),
        (
            "收益差图也统一为 GPT 减 Qwen，单位是每人每步收益，覆盖整个 10000 步。"
            "它与前面的各自相对 Fermi 收益图回答不同问题：负值仅表示 GPT 本次低于 Qwen，"
            "不自动表示 GPT 低于 Fermi，也不表示 GPT 的原始收益为负。"
            "由于两批相同键的 Fermi 对照一致，两个模型的配对收益增量之差，等于其原始人均每步收益之差。"
            "末段合作差与全程收益差不必完全一致，因为形成合作的速度、早期路径以及相同合作率下的空间结构也影响收益。"
            "图上只应作当前参数和两个种子的描述性对照；本轮没有独立重建多张建议表，模型层面的不确定性尚未测量。"
        ),
        (
            "动态页选取六个诊断切片，把 GPT、Qwen 与 Fermi 的时间轨迹放到同一张图中。"
            "线条是两个种子的均值，范围只表示这两条实际轨迹之间的距离；均值曲线并不是一次真实运行。"
            "观察重点包括 GPT 在扩大覆盖后高合作消失的过程、两张表在高 b 下的分化，以及同一参数下两个种子走向不同终点。"
            "应同时看初期下降、后续增长或灭绝与尾部变化；一个终点值或末段均值不能完整描述演化路径。"
            "如某曲线仍在变化，增加 STEPS 可以检查后期趋势，但单纯延长两条轨迹不能替代补充独立种子。"
            "所有结果都限定在本批有限的 10000 步内，不宣称已经证明长期稳态。"
        ),
        (
            "建议表页用于寻找行为差异的线索。两张表都包含全部 512 个 3×3 二元局面、每局面两条回答，"
            "当前扫描的具体 b 没有输入建议表，所有 b 复用同一张表。横轴只聚合四个直接邻居的合作数，"
            "仍将各个对角位置配置等权纳入；中心原为 C 与 D 分开统计。"
            f"全表等权合作建议比例为 GPT {g_advice['cooperation_pct']:.2f}%、Qwen {q_advice['cooperation_pct']:.2f}%；"
            f"两条回答一致的局面比例为 GPT {g_advice['agreement_pct']:.2f}%、Qwen {q_advice['agreement_pct']:.2f}%。"
            f"最明显的一类是中心 D、四个直接邻居全 C：GPT 给 C 的比例为 {g_advice['d4_c_pct']:.3f}%，"
            f"Qwen 为 {q_advice['d4_c_pct']:.3f}%。这提示两张表对被合作者包围的背叛者，转向合作的倾向不同。"
            "但这些是表内局面等权的比例，不是仿真实际访问局面加权的频率；不能单凭这张图就证明演化差异由这一类局面导致。"
            "机制验证需要增加局面访问统计，或只替换特定局面建议作受控消融。"
            "还需注意生成设置没有完全受控：两边的请求都未显式指定 temperature、top_p 和生成随机种子；"
            "Qwen 显式记录 reasoning_effort=medium，GPT 没有记录对应设置，不能假定两个后端默认相同。"
        ),
        (
            "本轮最适合汇报三点。第一，GPT 固定建议表的作用强烈依赖参数，扩大覆盖和提高采用频率都可能出现非单调变化。"
            f"b=1.02、λ=1，P=0.6 到 0.7 的末段合作从 {cooperation(g_groups, (1.02, 1.0, 0.6))} "
            f"降至 {cooperation(g_groups, (1.02, 1.0, 0.7))}，收益也由高于 Fermi 变成低于 Fermi。"
            "第二，在本次 210 组活跃参数中，Qwen 在 126 组的两个种子均有更高末段合作，GPT 没有相反的整组情况；"
            "这是一组匹配扫描中两张固定建议表的差异，不能泛化为模型能力排名。"
            "第三，每点两个种子仍会掩盖分岔，例如 GPT 的 b=1.03、λ=0.5、P=1，两次末段合作分别为 0 和 35.14%。"
            "下一轮优先在 b=1.02、λ=1 的 P=0.6–0.7 之间加密覆盖点，给这些转变点及大种子差区域补到 10–20 次重复。"
            "对尾部仍漂移的参数再延长 STEPS，并独立生成多张建议表，以区分演化随机性与建议表生成的不确定性。"
        ),
        (
            "附录展示 GPT 两个种子的末段合作率绝对差，单位为百分点，不是标准误或置信区间。"
            f"最大差出现在 b={worst_key[0]:g}、λ={worst_key[1]:g}、P={worst_key[2]:g}，"
            f"两个末段合作率为 {worst_values[0]:.5f}% 与 {worst_values[1]:.5f}%，相差 {spread:.5f} 个百分点。"
            "在这个参数点，一个种子很早就进入全 D，另一个在 10000 步内仍保留合作，均值会把两类路径混在一起。"
            "差值大的点应优先补重复；差值小、两次都为零或两次都为 100%，也不能证明总体方差小或无限时间稳定。"
        ),
        (
            "分组收益附录固定 GPT 的 λ=1，分别展示全体、资格组和无资格组相对同种子 Fermi 的收益变化。"
            "三个指标都是全程每人每步收益的配对差，两次重复先各自计算差值，再取均值。"
            "资格由初始化时的固定位置决定，并非按实际咨询次数划分。P=0 时资格组为空，P=1 时无资格组为空，"
            "空组的人均收益是 NA，不应填成零。"
            "全体差值应由存在的两组按实际人数加权；不能直接平均两组的人均值。"
            "资格组和无资格组都受到邻居及路径变化影响，组间差不能解释为单次咨询的直接因果效应。"
        ),
        (
            "等强度附录分别在 GPT 和 Qwen 下比较 P×λ=0.1 的三个配置，依次为 P=0.1、λ=1；P=0.2、λ=0.5；P=1、λ=0.1。"
            "这个乘积代表每个位置每步采用建议的平均概率尺度，不保证每步实际采用次数相同，也不消除固定资格位置的空间结构差异。"
            f"b=1.02 时，GPT 的三个末段合作均值依次为 {equal_g}，Qwen 为 {equal_q}。"
            "在这几个低 b 配置中，更多位置低频采用的均值较高，但 GPT 的两个种子在部分点存在明显分歧，"
            "因此不能把三点均值排序推广成普遍的覆盖率规律。"
            "b=1.03 时，GPT 的三种配置末段均无合作，而 Qwen 仍有差异；相同 P×λ 也不能消除建议表行为的差别。"
        ),
        (
            "存续图直接数每组两次运行中，第 10000 步仍存在至少一个合作者的次数，只能是 0/2、1/2 或 2/2。"
            f"在 P>0 且 λ>0 的 420 次运行中，GPT 的终点全 D 有 {extinct_g} 次，Qwen 有 {extinct_q} 次。"
            "这些次数是扫描网格的描述，参数点不是可交换的总体样本，不应当作模型总体灭绝概率。"
            "终点仍有合作、末 1000 步合作均值大于零、长期稳定是三个不同判断。"
            "例如某运行在最后窗口中途灭绝，末段平均仍可能大于零，但终点存续计数为零；即使 2/2 存续，也不能证明长期不会灭绝。"
        ),
        (
            "空间附录在相同参数下同时展示两张建议表、两个种子的第 10000 步实际格点。"
            "每张小图对应一个真实运行，而不是种子平均后的虚拟图。"
            "它有助于辨认合作团簇、稀疏残存与全合作或全背叛状态，并直观看出同参数下模型和种子的差异。"
            "不能只选其中一幅当作典型结果，也不能从一张末态快照证明形成机制；应结合前面的轨迹、两个种子范围和完整参数图阅读。"
        ),
    ]
    if len(notes) != 14:
        raise AssertionError("Every slide requires exactly one speaker-note block.")
    return notes, worst_key, spread


def writable_report_path(report_dir: Path):
    # Keep the verified lowercase repository alias instead of resolving output
    # paths through WSL's historical mixed-case mount spelling.
    report_dir = Path(report_dir)
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    lowercase_root = Path(str(ROOT).lower())
    if (lowercase_root.is_dir() and lowercase_root.samefile(ROOT)
            and report_dir.is_relative_to(ROOT)):
        report_dir = lowercase_root / report_dir.relative_to(ROOT)
    return report_dir


def build_deck(report_dir: Path, batch_dir: Path, qwen_dir: Path):
    report_dir = writable_report_path(report_dir)
    inputs = read_inputs(batch_dir, qwen_dir)
    _, _, g_groups, q_groups, _, _ = inputs
    notes, worst_key, spread = make_notes(*inputs)
    figure_dir = report_dir / "figures"
    for name in MAIN_FIGURES + APPENDIX_FIGURES:
        if not (figure_dir / name).is_file():
            raise FileNotFoundError(f"Missing report figure: {figure_dir / name}")
    title = "GPT 建议扫描：覆盖增加，合作一定增加吗？"
    source_label = (f"{SCAN_DATE} 两批匹配扫描  ·  gpt-5.5 / Qwen3.8  ·  "
                    "每参数 n=2；每模型一张固定建议表")
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(16), Inches(9)
    prs.core_properties.title = title
    prs.core_properties.subject = "GPT 参数扫描及与 Qwen 固定建议表的匹配比较"
    prs.core_properties.author = "LLM_Advice"
    prs.core_properties.keywords = "gpt-5.5,Qwen3.8,空间合作,覆盖率,采用概率,固定建议表,两次重复"

    slide = new_slide(prs)
    header(slide, f"组会汇报  /  GPT 参数扫描与 Qwen 对比  /  {REPORT_DATE}", title)
    text(slide, 0.94, 2.12, 14.1, 0.65,
         "同参数、同种子、同提示；比较两张固定建议表下的合作与收益。", 24, MUTED)
    rectangle(slide, 0.9, 3.0, 8.75, 3.92, PALE)
    text(slide, 1.25, 3.25, 8.0, 0.5, "两批共同设置", 24, NAVY, True)
    text(slide, 1.25, 3.95, 8.0, 2.75,
         "50 × 50 格点｜周期边界｜四近邻｜同步更新\n"
         "10,000 步；初始合作概率 50%；K = 0.1\n"
         "b = 1.01 / 1.02 / 1.03 / 1.04 / 1.05 / 1.1 / 1.2\n"
         "λ = 0 / 0.1 / 0.5 / 1；P = 0–1，步长 0.1\n"
         "P：固定资格位置比例；λ：资格位置采用建议的概率", 20)
    rectangle(slide, 10.0, 3.0, 5.1, 3.92, NAVY)
    text(slide, 10.4, 3.28, 4.25, 0.7, "每批 308 组 × 2 次", 29, WHITE, True)
    text(slide, 10.4, 4.22, 4.3, 2.35,
         "每模型 616 次实验 + Fermi 对照\nseed = 42 / 43\n"
         "末段合作：第 9001–10000 步\n全程收益：第 1–10000 步", 20, WHITE)
    text(slide, 0.95, 7.22, 14.1, 0.9,
         "模型记录：gpt-5.5 / Qwen3.8-27B-f16.gguf。结果限于本次两张表与两个种子。", 20, TEAL, True)
    footer(slide, 1, source_label)
    add_notes(slide, notes[0])

    for index, name in enumerate(MAIN_FIGURES, start=1):
        figure_slide(prs, figure_dir / name, notes[index])

    slide = new_slide(prs)
    header(slide, "当前观察与下一步", "GPT 的响应非单调，两张建议表产生不同演化结果")
    conclusions = (
        ("01", "扩大覆盖可能使合作与收益同时下降",
         f"GPT：b=1.02、λ=1，P 从 0.6 到 0.7，末段合作由 {cooperation(g_groups, (1.02, 1.0, 0.6))} 降至 {cooperation(g_groups, (1.02, 1.0, 0.7))}。"),
        ("02", "本次匹配扫描中，Qwen 表支持合作的区域更广",
         "210 组活跃参数中，126 组两个种子均为 Qwen 更高；没有两次均为 GPT 更高的组。"),
        ("03", "两个种子的均值仍可能掩盖存续与灭绝的差异",
         f"GPT：b={worst_key[0]:g}、λ={worst_key[1]:g}、P={worst_key[2]:g}，两次末段合作相差 {spread:.2f} 个百分点。"),
    )
    for index, (number, heading, body) in enumerate(conclusions):
        y = 2.22 + index * 1.46
        rectangle(slide, 0.9, y, 14.2, 1.21, PALE)
        text(slide, 1.18, y + 0.22, 0.8, 0.65, number, 29, TEAL, True)
        text(slide, 2.05, y + 0.15, 12.5, 0.42, heading, 23, NAVY, True)
        text(slide, 2.05, y + 0.69, 12.5, 0.40, body, 18.5, INK)
    text(slide, 0.98, 6.97, 14.0, 0.55,
         "下一步：加密 GPT 转变区；重点补 10–20 次重复；独立重建多张建议表。", 21, TEAL, True)
    text(slide, 0.98, 7.64, 14.0, 0.45,
         "本次是固定表下的探索性比较；n=2 的范围不是置信区间，也不足以给模型作一般能力排名。", 18, MUTED)
    footer(slide, 9, source_label)
    add_notes(slide, notes[8])
    for index, name in enumerate(APPENDIX_FIGURES, start=9):
        figure_slide(prs, figure_dir / name, notes[index])

    report_dir.mkdir(parents=True, exist_ok=True)
    destination = report_dir / "gpt_qwen_comparison_20261028.pptx"
    prs.save(destination)
    titles = (
        "两批实验设置与比较口径", "GPT：b × P 合作率全景", "GPT：覆盖率曲线与两次观测",
        "GPT：相对 Fermi 的全程人均收益", "合作率差：GPT 减 Qwen", "全程收益差：GPT 减 Qwen",
        "代表参数下的完整演化动态", "两张建议表的条件行为", "三个观察与下一步",
        "附录：GPT 两个种子的末段合作差", "附录：GPT 三个群体的配对收益", "附录：相同 P×λ 的覆盖方式",
        "附录：第 10000 步仍有合作的次数", "附录：两张表、两个种子的实际末态",
    )
    if len(prs.slides) != len(notes) or len(notes) != len(titles):
        raise AssertionError("Slides, titles, and speaker notes must have matching lengths.")
    script = report_dir / "组会讲稿.txt"
    blocks = [
        f"组会讲稿｜{REPORT_DATE}｜GPT 参数扫描与 Qwen 对比\n"
        f"GPT 数据来源：{batch_dir.name}\nQwen 数据来源：{qwen_dir.name}\n"
        "前 9 页主讲，后 5 页附录。每参数两个种子，范围不是置信区间。\n"
        "模型记录：gpt-5.5 / Qwen3.8-27B-f16.gguf；每模型一张固定经验建议表，每局面两条回答。"
    ]
    blocks.extend(f"第 {index} 页｜{slide_title}\n{note}"
                  for index, (slide_title, note) in enumerate(zip(titles, notes), start=1))
    script.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return destination, script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True, help="GPT scan directory")
    parser.add_argument("--qwen-dir", type=Path, required=True, help="Matching Qwen scan directory")
    args = parser.parse_args()
    for path in build_deck(args.report_dir, args.batch_dir, args.qwen_dir):
        print(path)


if __name__ == "__main__":
    main()
