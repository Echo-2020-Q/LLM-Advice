#!/usr/bin/env python3
"""Build a Chinese meeting deck comparing three fixed local advice tables.

Source logs are read only. This report does not contain DeepSeek SPD outcomes.
Requires python-pptx and figures created by visualize_three_advice.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

from build_group_meeting_deck import (
    INK, MUTED, NAVY, PALE, TEAL, WHITE,
    add_notes, figure_slide, footer, header, new_slide, rectangle, text,
)


REPORT_DATE = "2026-10-29"
SOURCES = {
    "DeepSeek": "20261029_121229_Deepseek1029",
    "GPT": "20261016_173605_GPT5.5",
    "Qwen": "20261024_122849_Qwen1024",
}
EXPECTED_MODELS = {
    "DeepSeek": "deepseek-v4.1-flash",
    "GPT": "gpt-5.5",
    "Qwen": "/home/tianyuxuan/qwen/Qwen3.8-27B-f16.gguf",
}
EXPECTED_N = {"DeepSeek": 1, "GPT": 2, "Qwen": 2}
FIGURES = (
    ("01_overview.png", "总体合作建议与保持原行动"),
    ("02_transitions.png", "保留合作与转为合作是两个方向"),
    ("03_neighbors.png", "四个直接邻居如何对应合作建议"),
    ("04_table_distance.png", "同局面的经验合作频率距离"),
    ("05_cases.png", "四个实际局面的建议差异"),
    ("06_diagonals.png", "附录：直接邻居与对角邻居分层"),
    ("07_repeat_agreement.png", "附录：两次回答的一致性"),
    ("08_paired_choices.png", "附录：DeepSeek 与另外两表的回答配对"),
)


def writable_path(path: Path) -> Path:
    """Preserve the verified lowercase alias used by the workspace sandbox."""
    path = path.absolute()
    root = Path(__file__).resolve().parents[1]
    lower = Path(str(root).lower())
    if lower.is_dir() and lower.samefile(root) and path.is_relative_to(root):
        path = lower / path.relative_to(root)
    return path


def read_inputs(log_root: Path):
    expected_keys = {"".join(chars) for chars in itertools.product("CD", repeat=9)}
    tables = {}
    for model, directory in SOURCES.items():
        path = log_root / directory
        config = json.loads((path / "config.json").read_text(encoding="utf-8-sig"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8-sig"))
        with (path / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        n = EXPECTED_N[model]
        if (config["model"] != EXPECTED_MODELS[model] or config["n"] != n
                or config["case_count"] != 512 or len(rows) != 512 * n
                or summary["status"] != "completed"
                or summary["successful_requests"] != 512 * n
                or config["b"] is not None or config["b_constraint"] != "1 < b < 2"):
            raise ValueError(f"Unexpected source metadata for {model}; revise the narrative.")
        grouped = {}
        for row in rows:
            key = row["grid_key"]
            if (row["status"] != "success" or row["recommendation"] not in "CD"
                    or len(row["recommendation"]) != 1
                    or row["current_action"] != key[4]
                    or int(row["cooperating_neighbors"]) != sum(key[i] == "C" for i in (1, 3, 5, 7))):
                raise ValueError(f"Unexpected answer or neighborhood in {model}.")
            grouped.setdefault(key, []).append(row)
        if set(grouped) != expected_keys or any(
                len(group) != n or {int(row["repetition"]) for row in group} != set(range(1, n + 1))
                for group in grouped.values()):
            raise ValueError(f"Incomplete or duplicate states for {model}.")
        table = {key: statistics.mean(row["recommendation"] == "C" for row in group)
                 for key, group in grouped.items()}
        tables[model] = {
            "config": config, "rows": rows, "table": table,
            "overall": statistics.mean(table.values()),
            "retain": statistics.mean(value for key, value in table.items() if key[4] == "C"),
            "convert": statistics.mean(value for key, value in table.items() if key[4] == "D"),
            "same": None if n == 1 else sum(value in (0, 1) for value in table.values()),
            "prompt_hash": hashlib.sha256(config["system_prompt"].encode()).hexdigest(),
        }
    if len({data["prompt_hash"] for data in tables.values()}) != 1:
        raise ValueError("The prompts differ; revise the comparison's setup notes.")
    return tables


def pct(value):
    return f"{100 * value:.2f}%"


def neighbor_rate(tables, model, center, count):
    selected = [value for key, value in tables[model]["table"].items()
                if key[4] == center and sum(key[i] == "C" for i in (1, 3, 5, 7)) == count]
    return statistics.mean(selected)


def setup_note(tables, audit):
    gpt_aliases = "、".join(f"{name}（{count} 条）" for name, count in
                           audit["sources"]["GPT"]["response_model_counts"].items())
    return (
        "本次比较三批已经生成的固定建议表，数据来自用户指定的三个日志目录。"
        "三批都完整覆盖全部 512 个 3×3 二元局面，没有合并旋转或镜像；所有最终保留回答均成功。"
        "系统提示词一致，都只给出 1<b<2 的范围，没有输入某个数值 b。"
        "DeepSeek 每局面一条回答，共 512 条；GPT 和 Qwen 每局面两条，共各 1024 条。"
        "图中的 GPT 是用户指定的 GPT5.5 日志批次，模型名称均应结合配置和响应元数据理解，"
        "不能仅凭目录名称确认服务端身份。"
        f"GPT 全部请求的模型名是 gpt-5.5；响应模型标记为 {gpt_aliases}。"
        "这些是多个服务方别名或版本标记，底层同一性未核实，不能据此直接断言混合了不同模型。"
        "GPT 请求未显式指定推理强度或采样温度，响应元数据均记录 reasoning.effort=medium、"
        "temperature=1.0、top_p=0.98。DeepSeek 和 Qwen 的响应没有温度记录，"
        "缺失字段不代表默认值相同。Qwen 请求记录 reasoning_effort=medium。"
        "DeepSeek 请求输出上限为 51200，GPT 和 Qwen 为 10240。"
        "这些比例按每个局面等权计算，不是空间演化实际访问的局面频率。"
        "输出预算、服务接口和推理设置未完全统一；所以比较对象是这三批已保存的输出，"
        "不能把差异直接归因为模型能力。这里没有 DeepSeek 的 SPD 仿真结果。"
    )


def conclusion_note(tables):
    d, g, q = (tables[key] for key in ("DeepSeek", "GPT", "Qwen"))
    return (
        f"本次 DeepSeek 合作建议占比 {pct(d['overall'])}，处于 GPT {pct(g['overall'])} 与 Qwen {pct(q['overall'])} 之间。"
        "但更有解释力的是两个方向："
        f"本人当前为 C 时，DeepSeek 有 {pct(d['retain'])} 的回答继续建议 C；"
        f"本人当前为 D 时，只有 {pct(d['convert'])} 建议转为 C。"
        "因此这张 DeepSeek 表相对更偏向维持已有合作、较少主动把背叛转成合作。"
        "这是一条局部策略的描述，并不等于已经观察到总体合作更高或更低。"
        "下一步优先补齐每个局面的回答次数，并独立重建多张建议表；"
        "若直接沿用同局面回复中有放回抽样的 SPD 实现，DeepSeek 每局面只有一条固定回答，"
        "给定局面下的建议抽样不会改变动作；GPT 与 Qwen 的混合回答局面仍含表内抽样随机性。"
        "这并不说明真实模型是确定性的，也不表示 DeepSeek 仿真中的采用或 Fermi 更新没有随机性。"
        "所以应先统一每局面回答数，避免把表的采样分辨率差异当成模型差异。"
        "在同样的 b、P、λ、初态与配对 Fermi 对照下运行 DeepSeek，"
        "同时记录实际访问的局面和 C→D、D→C 转换，检验本次提出的机制线索。"
        "仅增加 SPD 种子会评估固定表下的演化随机性，不能代替重新采样建议表。"
    )


def figure_notes(tables):
    """Descriptions are keyed by the final figure filenames."""
    d, g, q = (tables[key] for key in ("DeepSeek", "GPT", "Qwen"))
    def keep(model):
        return statistics.mean(value if key[4] == "C" else 1-value
                               for key, value in tables[model]["table"].items())

    def distance(a, b, center=None):
        return statistics.mean(abs(value - tables[b]["table"][key])
                               for key, value in tables[a]["table"].items()
                               if center is None or key[4] == center)

    def pairs(other):
        return {(dc, oc): sum(d["table"][key] == dc and tables[other]["table"][key] == oc
                             for key in d["table"])
                for dc in (0, 1) for oc in (0, .5, 1)}

    dg, dq = pairs("GPT"), pairs("Qwen")
    return {
        "01_overview.png": (
            "首先按每个局面等权统计合作建议。"
            f"DeepSeek 为 194/512，即 {pct(d['overall'])}；GPT 为 280/1024，即 {pct(g['overall'])}；"
            f"Qwen 为 583/1024，即 {pct(q['overall'])}。"
            "DeepSeek 的总体合作建议比例位于另外两张表之间。"
            f"但保持本人原行动的比例，DeepSeek 是 {pct(keep('DeepSeek'))}，"
            f"GPT 是 {pct(keep('GPT'))}，Qwen 是 {pct(keep('Qwen'))}。"
            "保持原行动同时包括 C→C 和 D→D，不应误读成持续合作。"
            "这些是本批输出的表内经验比例，不能外推为运行中实际触发的频率。"
        ),
        "02_transitions.png": (
            "把中心当前行动与建议行动交叉，才能看到总体合作比例背后的方向差别。"
            f"当前为 C 时，DeepSeek 建议 C 为 {pct(d['retain'])}，"
            f"高于 GPT 的 {pct(g['retain'])} 和 Qwen 的 {pct(q['retain'])}。"
            f"当前为 D 时，DeepSeek 建议转 C 仅 {pct(d['convert'])}，"
            f"GPT 为 {pct(g['convert'])}，Qwen 为 {pct(q['convert'])}。"
            "因此这张 DeepSeek 表主要表现为保留原来的合作，而不是更积极地把背叛转为合作。"
            "各行以相同的本人当前行动为条件归一化，每个条件包含 256 个局面。"
        ),
        "03_neighbors.png": (
            "这里仅按上、下、左、右四个直接邻居中的合作者数量分组，"
            "分别观察本人当前为 C 和 D 的情况。每条线上五个横坐标的局面数为 16、64、96、64、16。"
            f"本人 C、四邻居全 D 时，DeepSeek 仍有 {pct(neighbor_rate(tables, 'DeepSeek', 'C', 0))} 的回答建议 C，"
            "GPT 和 Qwen 在本批对应回答中均为 0。"
            f"本人 D、四邻居全 C 时，DeepSeek 建议 C 为 {pct(neighbor_rate(tables, 'DeepSeek', 'D', 4))}，"
            f"GPT 为 {pct(neighbor_rate(tables, 'GPT', 'D', 4))}，"
            f"Qwen 为 {pct(neighbor_rate(tables, 'Qwen', 'D', 4))}。"
            "这给出了后续检验合作维持与扩张的两类关键局面。曲线连线仅帮助阅读；"
            "并非经过拟合的行为模型，也没有用少量重复构造置信区间。"
        ),
        "04_table_distance.png": (
            "每个局面先计算表内建议 C 的经验频率 q(s)，再在相同局面间取两个表的频率绝对差并平均。"
            "DeepSeek 的 q 只有 0 或 1，GPT 和 Qwen 的 q 为 0、0.5 或 1，混合回答保留为 0.5。"
            f"整体距离 DeepSeek–GPT 为 {100*distance('DeepSeek','GPT'):.2f} 个百分点，"
            f"DeepSeek–Qwen 为 {100*distance('DeepSeek','Qwen'):.2f} 个百分点。"
            "按照这一次的经验频率距离，DeepSeek 整体更接近 GPT。"
            f"但本人为 C 时，这两个距离分别为 {100*distance('DeepSeek','GPT','C'):.2f} 和 "
            f"{100*distance('DeepSeek','Qwen','C'):.2f} 个百分点，反而更接近 Qwen。"
            "这是本次固定表的描述性距离，不是模型能力、真实策略概率或宏观演化结果的距离；"
            "每局面样本数不同也影响经验频率的分辨率。"
        ),
        "05_cases.png": (
            "这些格图展示原始记录中的具体局面和回答，而非人工构造的平均局面。"
            "例如 CCC/CDC/CCC，即本人 D、周围八格均 C，DeepSeek 建议 D，GPT 两答为 C 和 D，Qwen 两答均 C。"
            "另一个例子 CDC/DCD/CDC，即本人 C、四直接邻居全 D、四角全 C，"
            "DeepSeek 建议 C，而 GPT 与 Qwen 两答均 D。"
            "两个仅改变对角位置、互为左右镜像的局面，DeepSeek 单次回答也可能不同。"
            "由于每局面只有一次 DeepSeek 观测，这种差异无法区分稳定位置偏好与抽样波动。"
            "三张表在全 C 局面都建议 C、全 D 局面都建议 D；这类建议本身不会打破两个极端状态。"
        ),
        "06_diagonals.png": (
            "附录进一步同时按直接邻居合作者数和对角邻居合作者数分层，分别看本人当前为 C 与 D。"
            "对角邻居不直接与本人结算四近邻收益，但作为 3×3 观察输入的一部分，仍可能影响建议。"
            "每个格子的局面数为两个组合数的乘积；深浅表达该层已保存回答中的合作建议比例。"
            "完整枚举提供了不同直接与对角数量的交叉对比，但没有证明模型采用某一稳定规则。"
            "相同数量仍可能有不同空间排列，尤其不能用 DeepSeek 每局面一次的回答，"
            "将几何差异和生成随机性完全分开。"
        ),
        "07_repeat_agreement.png": (
            f"GPT 的两次回答一致局面为 {g['same']}/512，即 {pct(g['same']/512)}；"
            f"Qwen 为 {q['same']}/512，即 {pct(q['same']/512)}。"
            "一致包含两次均 C 与两次均 D，不代表更合作、更准确或者长期稳定。"
            "DeepSeek 每局面只有一次，因此重复一致性没有可比估计，必须标为 NA，不能写成 100%。"
            "GPT 和 Qwen 的两答仍然很少，混合或一致只能作为这次采样的事实，不能精确估计真实选择概率。"
            "如果在后续 SPD 中从每个局面的固定回答里有放回抽样，DeepSeek n=1 对应固定的局面到动作映射，"
            "GPT 有 64 个、Qwen 有 79 个混合回答局面，其表内建议抽样仍会变化。"
            "这属于表的结构差异，不是对真实模型确定性的判断；整个仿真仍可由采用与 Fermi 等环节产生随机性。"
            "后续既应增加每个局面的回复数，也应独立生成多张表，以区分表采样和 SPD 演化的随机性。"
        ),
        "08_paired_choices.png": (
            "最后按同一局面配对：行是 DeepSeek 单次回答 C 或 D，列是另一张表的两答均 D、"
            "一 C 一 D、两答均 C，每个面板总数为 512。"
            f"DeepSeek C / GPT 两答 D 的局面有 {dg[1,0]} 个，反过来 DeepSeek D / GPT 两答 C 有 {dg[0,1]} 个。"
            f"DeepSeek C / Qwen 两答 D 有 {dq[1,0]} 个，DeepSeek D / Qwen 两答 C 有 {dq[0,1]} 个。"
            "混合回答单独保留，不用人为多数决策破平局。"
            "这样可直接看见两张表方向完全相反的位置，但 DeepSeek 单次回答仍可能受生成随机性影响。"
        ),
    }


def build_deck(report_dir: Path, log_root: Path):
    report_dir = writable_path(report_dir)
    tables = read_inputs(log_root)
    audit = json.loads((report_dir / "data" / "audit.json").read_text(encoding="utf-8"))
    if (audit["status"] != "passed" or audit["records_validated"] != 2560
            or not audit["all_system_prompts_identical"]
            or not audit["all_paired_user_prompts_identical"]):
        raise ValueError("The raw-record audit is incomplete or the prompts differ.")
    for model, data in tables.items():
        if audit["sources"][model]["cooperation_fraction"] != data["overall"]:
            raise ValueError(f"The raw-record audit differs from current results.csv: {model}.")
    if not FIGURES:
        raise ValueError("The report's figure manifest has not been finalized.")
    for name, _ in FIGURES:
        if not (report_dir / "figures" / name).is_file():
            raise FileNotFoundError(report_dir / "figures" / name)
    notes_by_figure = figure_notes(tables)
    if set(notes_by_figure) != {name for name, _ in FIGURES}:
        raise ValueError("Every figure needs matching speaker notes.")

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(16), Inches(9)
    prs.core_properties.title = "DeepSeek、GPT 与 Qwen：局部建议策略有什么差别？"
    prs.core_properties.subject = "三批固定建议输出；512 个同局面配对比较"
    prs.core_properties.author = "LLM_Advice"
    prs.core_properties.keywords = "DeepSeek,GPT,Qwen,建议表,空间合作,条件策略"
    source_label = f"{REPORT_DATE} · 三批固定建议表；DeepSeek n=1，GPT / Qwen n=2；未包含 DeepSeek SPD 结果"
    slide = new_slide(prs)
    header(slide, f"组会材料 / 建议输出对比 / {REPORT_DATE}",
           "DeepSeek、GPT 与 Qwen：局部建议策略有什么差别？")
    text(slide, 0.95, 2.1, 14.0, 0.65,
         "同一套 512 个局面，先比较“在什么情况下建议合作”。", 25, MUTED)
    rectangle(slide, 0.9, 3.02, 8.55, 3.7, PALE)
    text(slide, 1.27, 3.3, 7.8, 0.48, "共同条件", 25, NAVY, True)
    text(slide, 1.27, 4.03, 7.85, 2.25,
         "3×3 二元局面；本人位于中心\n直接相互作用：上、下、左、右四邻居\n"
         "相同系统提示词；只告知 1 < b < 2\n全部局面等权；保留旋转与镜像", 22)
    rectangle(slide, 9.8, 3.02, 5.3, 3.7, NAVY)
    text(slide, 10.18, 3.3, 4.5, 0.55, "样本量不相同", 28, WHITE, True)
    text(slide, 10.18, 4.13, 4.55, 2.15,
         "DeepSeek：512 × 1 条\nGPT：512 × 2 条\nQwen：512 × 2 条\n最终保留回答均成功", 22, WHITE)
    text(slide, 0.97, 7.04, 14.0, 0.56,
         "DeepSeek n=1：重复一致性无法估计；局部建议差异尚不能推出长期演化表现。",
         21, TEAL, True)
    text(slide, 0.97, 7.78, 14.0, 0.36,
         "GPT 响应含多个别名 / 版本标记，底层同一性未核实；生成设置详情见本页讲稿。", 16, MUTED)
    footer(slide, 1, source_label)
    first_note = setup_note(tables, audit)
    add_notes(slide, first_note)
    script_entries = [("比较范围与共同条件", first_note)]

    # The figure manifest specifies which pages belong in the appendix.
    main_figures = [(name, title) for name, title in FIGURES if not title.startswith("附录")]
    appendix = [(name, title) for name, title in FIGURES if title.startswith("附录")]
    for name, title in main_figures:
        figure_slide(prs, report_dir / "figures" / name, notes_by_figure[name])
        script_entries.append((title, notes_by_figure[name]))

    slide = new_slide(prs)
    header(slide, "结论与下一步", "DeepSeek 更常维持已有合作，很少建议从 D 转为 C")
    d, g, q = (tables[key] for key in ("DeepSeek", "GPT", "Qwen"))
    items = (
        ("01", "总体占比处于两者之间",
         f"DeepSeek {pct(d['overall'])}；GPT {pct(g['overall'])}；Qwen {pct(q['overall'])}。"),
        ("02", "把“维持合作”与“转为合作”分开看",
         f"DeepSeek 本人 C 时建议 C 为 {pct(d['retain'])}；本人 D 时仅 {pct(d['convert'])}。"),
        ("03", "局部建议差异提供下一轮实验的检验方向",
         "补齐建议重复，独立重建建议表，再用相同参数与种子运行 DeepSeek SPD。"),
    )
    for index, (number, title, body) in enumerate(items):
        y = 2.3 + index * 1.43
        rectangle(slide, 0.9, y, 14.2, 1.20, PALE)
        text(slide, 1.18, y + 0.25, 0.8, 0.55, number, 29, TEAL, True)
        text(slide, 2.06, y + 0.14, 12.45, 0.44, title, 23, NAVY, True)
        text(slide, 2.06, y + 0.68, 12.45, 0.39, body, 20, INK)
    text(slide, 0.97, 7.02, 14.0, 0.95,
         "重点记录实际访问局面与 C→D / D→C 转换，检验合作的维持、扩张和灭绝过程。\n"
         "当前结果属于三张固定表的描述，尚不足以作模型能力排名。", 21, TEAL, True)
    footer(slide, len(prs.slides), source_label)
    last_note = conclusion_note(tables)
    add_notes(slide, last_note)
    script_entries.append(("三个观察与下一步", last_note))
    for name, title in appendix:
        figure_slide(prs, report_dir / "figures" / name, notes_by_figure[name])
        script_entries.append((title, notes_by_figure[name]))

    report_dir.mkdir(parents=True, exist_ok=True)
    destination = report_dir / "deepseek_gpt_qwen_advice_20261029.pptx"
    prs.save(destination)
    script = report_dir / "组会讲稿.txt"
    blocks = [f"组会讲稿｜{REPORT_DATE}｜DeepSeek、GPT、Qwen 建议输出对比\n"
              f"前 {2 + len(main_figures)} 页主讲，其后为附录。\n"
              "所有统计按已保存局面等权计算；每局面 DeepSeek 1 条，GPT 和 Qwen 各 2 条。\n"
              "数据来源：\n" + "\n".join(f"{model}：{path}" for model, path in SOURCES.items())]
    blocks.extend(f"第 {index} 页｜{title}\n{note}"
                  for index, (title, note) in enumerate(script_entries, start=1))
    script.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return destination, script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path,
                        default=Path("reports/advice_deepseek_gpt_qwen_20261029"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/GPT_Multistepsa"))
    args = parser.parse_args()
    for path in build_deck(args.report_dir, args.log_root):
        print(path)


if __name__ == "__main__":
    main()
