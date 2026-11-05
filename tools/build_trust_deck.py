#!/usr/bin/env python3
"""Create a Chinese meeting deck for the completed adaptive-trust scan.

Figures are generated separately by the trust visualization script. This builder
reads existing records only, validates the scan-specific narrative, and exports
editable Chinese slide text, notes, and a concise Chinese interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches

from build_group_meeting_deck import (
    INK, MUTED, NAVY, PALE, TEAL, WHITE,
    add_notes, footer, header, new_slide, rectangle, text,
)


NEW_SCAN = "main/SPD/results/scan_20261029_183749_initial"
OLD_SCAN = "main/SPD/results/scan_20261027_121246_2次gpt微扫描"
FIGURES = (
    ("01_cooperation.png", "合作边界向更高 b 移动，但 b=1.04 依赖种子"),
    ("02_fixed_comparison.png", "相较固定 λ=0.5：23 组改善，47 组持平"),
    ("03_coverage.png", "本批随 P 增加而改善；自适应并非始终最好"),
    ("04_trust.png", "合作、信任与实际建议使用率共同分化"),
    ("10_cooperation_dynamics.png", "合作轨迹：恢复、持续混合与灭绝三种路径"),
    ("11_trust_dynamics.png", "信任回落后，实际建议使用率随之分化"),
    ("06_trust_distribution.png", "均值之外：查看单次运行内的信任分布"),
    ("07_seed_dependence.png", "b=1.04 的均值掩盖了两种不同结局"),
    ("08_payoff.png", "末段收益改善，不代表全程或初期都受益"),
    ("09_feedback_caveat.png", "持续正反馈不一定意味着收益仍在提高"),
)


def writable_path(path: Path) -> Path:
    path = path.absolute()
    root = Path(__file__).resolve().parents[1]
    lower = Path(str(root).lower())
    if lower.is_dir() and lower.samefile(root):
        prefix = str(root).casefold()
        if str(path).casefold() == prefix or str(path).casefold().startswith(prefix + "/"):
            return lower.joinpath(*path.parts[len(root.parts):])
    return path


def load_inputs(repo: Path):
    current = pd.read_csv(repo / NEW_SCAN / "summary.csv")
    previous = pd.read_csv(repo / OLD_SCAN / "summary.csv")
    config = json.loads((repo / NEW_SCAN / "scan_config.json").read_text())
    expected = {(b, p / 10, seed)
                for b in (1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.2)
                for p in range(11) for seed in (42, 43)}
    actual = set(current[["b", "llm_access_fraction", "seed"]].itertuples(index=False, name=None))
    if len(current) != 154 or actual != expected or not current.status.eq("completed").all():
        raise ValueError("Narrative requires the complete 154-run adaptive scan.")
    constants = {
        "steps": 10000, "initial_trust": .5, "trust_learning_rate": .1,
        "aspiration_learning_rate": .1, "trust_payoff_window": 3,
        "tail_start_round": 9001, "tail_end_round": 10000,
        "early_average_steps": 200,
    }
    for field, value in constants.items():
        if not current[field].eq(value).all():
            raise ValueError(f"Narrative setting changed: {field}")
    if {source["model"] for source in config["advisor_sources"]} != {"gpt-5.5"}:
        raise ValueError("This narrative expects the recorded GPT advice table.")
    grouped = current.groupby(["b", "llm_access_fraction"]).mean(numeric_only=True)
    old_grouped = previous.groupby(["b", "llm_access_fraction", "lambda_prob"]).mean(numeric_only=True)
    active = grouped.copy().reset_index().query("llm_access_fraction > 0").set_index(["b", "llm_access_fraction"])
    fixed_half = old_grouped.xs(.5, level="lambda_prob")
    gap = (active.tail_mean_cooperation_rate - fixed_half.tail_mean_cooperation_rate).dropna()
    counts = ((gap > 1e-12).sum(), (gap.abs() <= 1e-12).sum(), (gap < -1e-12).sum())
    if counts != (23, 47, 0):
        raise ValueError(f"Adaptive/fixed comparison changed: {counts}")
    if not np.allclose(current.delta_tail_mean_cooperation_rate,
                       current.tail_mean_cooperation_rate - current.control_tail_mean_cooperation_rate):
        raise ValueError("Paired cooperation differences do not match.")
    if not np.isclose(grouped.loc[(1.04, 1), "tail_mean_cooperation_rate"], .4186062):
        raise ValueError("Critical seed-split narrative changed.")
    return current, previous, grouped, old_grouped


def make_notes(current, grouped, old_grouped):
    c = lambda b, p: 100 * grouped.loc[(b, p), "tail_mean_cooperation_rate"]
    f = lambda b, p, lam: 100 * old_grouped.loc[(b, p, lam), "tail_mean_cooperation_rate"]
    return [
        "今天汇报加入个体信任后的第一次扫描。使用同一张 GPT 建议表：512 个三乘三局面，"
        "每个局面两条固定回复，模拟中有放回抽样。本次没有重新调用模型。"
        "P 是初始化时固定的有建议资格的位置比例；这些个体的建议使用概率为各自信任 τ_i，"
        "初始均为 0.5。信任学习率 α 和收益参考学习率 β 都是 0.1，反馈收益取最近三轮已完成收益的均值。"
        "每次采用建议后，若该均值高于旧参考 A_i，τ_i 增加 α(1−τ_i)；低于则乘以 1−α；相等不变。"
        "未采用建议时不更新信任，但所有个体都更新收益参考：A_i←0.9A_i+0.1平均收益。"
        "比较依据是旧参考收益，并非单纯比较本轮与上一轮。"
        "50乘50周期格点、四近邻、同步更新、K=0.1，初始合作概率0.5。"
        "扫描7个b和11个P，共77组，每组seed42、43两次，运行10000步。末段指9001至10000步。"
        "所有区间是这两个种子的范围，不是置信区间；以下结果是初步描述。",

        "先看合作率全图。本批随P增大，末段合作率在两个种子内都不下降。"
        "b=1.01从P=0.7起两次运行都达到全合作；b=1.02从P=0.9起达到全合作。"
        f"b=1.03在P≤0.6全部灭绝，P=0.7时平均合作率{c(1.03,.7):.2f}%，P=1全合作。"
        "b=1.04只有P=0.9和1出现存活，而且两个种子结局分裂；b≥1.05所有P都灭绝。"
        "这些是当前离散网格上的观察，不能直接称为精确相变边界或稳定平衡。",

        "以固定λ=0.5作主要对照，因为它与个体信任的初始使用概率相同。"
        "排除P=0后共有70个参数组；自适应末段合作率均值在23组更高、47组相同，没有均值更低的组。"
        f"最大例子是b=1.03、P=1：固定λ=0.5为{f(1.03,1,.5):.2f}%，自适应为100%，提高{100-f(1.03,1,.5):.2f}个百分点。"
        "相对配对Fermi，末段均值26组提高、44组持平。建议表和主要实验设置一致，但旧扫描没有保存初始网格及资格掩码，"
        "因此不声称已经逐格验证同初态，也不把跨运行差异解释成单次建议的因果收益。",

        "这一页加入固定λ=1，避免仅凭λ=0.5对照就判断自适应普遍最好。"
        f"b=1.02、P=0.6时，自适应{c(1.02,.6):.2f}%，固定λ=1为{f(1.02,.6,1):.2f}%。"
        f"b=1.01、P=0.5时，自适应{c(1.01,.5):.2f}%，固定λ=1为{f(1.01,.5,1):.2f}%。"
        "而在高覆盖率的低b区域，自适应可以避开固定λ=1的合作骤降。"
        "b≥1.05、P=1时，自适应为0，固定λ=1仍约9.86%。"
        "与旧网格中每格最好的λ比较，70组中14组更高、29组持平、27组更低。"
        "这里的最好是事后选出的描述性上界；核心结论是自适应改变了有效使用强度及其个体分布。",

        "有资格者的平均信任从初始0.5分化：合作成功的区域维持较高信任，灭绝区域降到很低。"
        "在灭绝区域，尾段平均信任大致为0.2%至2%；在全合作区域接近1。"
        "图中的建议使用率以有资格者为分母；换成全体格点的实际使用比例，还要乘P。"
        "例如P相同，信任下降会使实际建议使用远低于固定λ=0.5；不能只用P描述暴露程度。"
        "信任和合作共同变化不证明单向因果。P=0没有有资格者，因此信任指标应为空而非0。",

        "轨迹说明末段均值背后存在不同过程。初期合作和信任通常先下降，成功运行随后逐步恢复；"
        "另一些运行在信任较低时进入全背叛吸收态。b=1.01、P=1、seed42在615步首次全合作；"
        "b=1.03、P=1、seed42在2599步首次全合作。"
        "b=1.04、P=1、seed42曾暂时保持合作，但3848步灭绝。"
        "相反seed43在末段保持约84%合作。已经全背叛的运行在当前建议表和无突变规则下不会自发恢复，"
        "单纯延长这些运行不能验证恢复现象；应把时间预算优先用于仍存活或混合的边界组。",

        "接着看同一批代表性运行的信任与实际建议使用轨迹。初始信任都是0.5，早期明显回落。"
        "在合作恢复的运行中，信任及建议使用率又逐步上升；在灭绝路径中则保持很低。"
        "图中建议使用率以有资格个体为分母，来自每轮实际采用建议的计数。"
        "本轮采用建议的概率使用更新前的信任，记录的信任均值则是本轮反馈后的值，"
        "因此两条轨迹应接近但不必逐点相等。"
        "这些是同一系统中共同变化的量，仍不能仅凭相关性断言信任变化导致合作恢复。",

        "这一页用保存的十个等宽分箱查看个体信任分布，分布来自每次运行内2500P个有资格位置。"
        "它们不是独立实验重复，也不是最终个体信任空间地图，因为本批没有保存逐格信任。"
        "例如b=1.02、P=0.1，两个seed最终低于0.1的个体比例分别为49.2%和65.2%。"
        "而b=1.03、P=0.7，信任分布跨越多个区间。仅凭均值看不出这些个体差异，"
        "仅凭分箱也无法确认每个个体的具体经历。"
        "必须把跨种子的结局差异与同一次运行内的异质性分开。",

        "这是最需要增加重复次数的例子。b=1.04、P=0.9时，seed42在2730步灭绝，"
        "seed43末段合作率35.58%，二者平均17.79%。P=1时seed42在3848步灭绝，"
        "seed43末段83.72%，二者平均41.86%。"
        "因此41.86%不代表某次运行维持约42%的稳定混合态。P=1的存活运行最终平均信任0.812，"
        "64.48%的个体信任高于0.9；灭绝运行平均信任0.00121，全部低于0.1。"
        "两次重复不足以估计存活概率或支持双稳态的正式结论，但已经明确揭示种子敏感性。",

        "收益和合作结论应按时间窗口分别报告。70个P>0参数组中，相对配对Fermi的末段全体人均收益，"
        "26组均值提高、44组持平，没有下降。"
        "全10000步的人均每轮收益差，60组为正、10组为负；负值最小约−0.00159。"
        "前200步则有21组为负、49组为正，所以长期末段改善不代表初期获益。"
        "资格组和无资格组的收益差包含邻域传播与系统演化改变，不能解释成一次咨询的直接回报。"
        "注意图中人均每轮收益与人均累计收益差一个总步数因子，不能混用量纲。",

        "需要先确认反馈定义的一个细节。全合作后每人每轮收益固定为4，收益参考A仍可能略低于4。"
        "图中b=1.01、P=0.7、seed42在7431步进入全合作，末段参考为3.9999999999999982，"
        "差距约1.78乘10的负15次方，每轮却仍有1750次正反馈，即所有有资格个体。"
        "类似地，b=1.03、P=1、seed42末段每轮2500次正反馈。"
        "这是EMA从下方渐近逼近收益、比较严格使用大于小于且没有死区的共同结果；浮点精度会保留极小残差。"
        "它不代表收益仍在增加，也不应仅仅视作整个实验无效。"
        "下一步应固定当前结果作为基线，加入有明确容差的敏感性比较，并决定研究问题到底是收益超过期望，"
        "还是收益比上一轮提高。两个规则对应不同的信任含义。本次分析未更改代码或重跑实验。",

        "下一步按优先级推进。首先明确收益参考的语义，对正负反馈加可记录的数值死区做敏感性实验。"
        "随后在b约1.03到1.04、P约0.7到1重点增加到至少10个seed，尤其是b=1.04、P=0.9和1；"
        "报告存活次数、灭绝时间以及每个seed的结果。"
        "对仍有合作的边界组延长到20000至50000步，检查末段漂移，不把尾段直接称为稳态。"
        "保留固定λ=0.5、λ=1与Fermi对照，进一步加入平均使用强度匹配的同质信任对照，"
        "区分平均咨询减少与个体选择的效果。之后分别扫描初始信任、α、β和收益窗口W，并在Qwen表上重复，"
        "检验规则效果是否依赖特定建议表。当前只得出两seed、单张GPT表上的初步结论。",
    ]


def figure_panel(prs, path, title, page, note):
    slide = new_slide(prs)
    rectangle(slide, .45, .28, .08, .46, TEAL)
    text(slide, .7, .24, 14.6, .6, title, 26, NAVY, True)
    with Image.open(path) as image:
        aspect = image.width / image.height
    max_w, max_h = 15.15, 7.55
    width = min(max_w, max_h * aspect)
    height = width / aspect
    slide.shapes.add_picture(str(path), Inches((16-width)/2),
                             Inches(.87+(max_h-height)/2),
                             width=Inches(width), height=Inches(height))
    text(slide, .7, 8.57, 14.1, .22,
         "GPT 固定建议表 · 每组 2 个种子 · 末段：9001–10000 步 · 种子范围不是置信区间", 10, MUTED)
    text(slide, 15.0, 8.52, .5, .3, str(page).zfill(2), 12, MUTED)
    add_notes(slide, note)
    return slide


def chinese_summary(grouped, old_grouped):
    return """加入个体信任后的初步结果（2026-10-30）

数据：scan_20261029_183749_initial；77组参数，每组seed42、43，共154次运行。
建议来源仍是20261016的GPT固定建议表，每局面2条回复；本次没有重新调用模型。
50×50周期格点、同步更新、K=0.1；初始信任0.5，α=β=0.1，收益反馈窗口3轮。
10000步；末段为9001–10000步，早期为1–200步。n=2，只作描述性比较。

1. 个体信任改变了合作能够存活的参数范围。
b=1.01从P≥0.7起两次运行都全合作；b=1.02从P≥0.9起全合作。
b=1.03在P≤0.6都灭绝，从P=0.7起两次都存活，P=1全合作。
b=1.04仅P=0.9和1出现存活，且只有seed43存活；b≥1.05所有P都灭绝。
随P增加，末段合作率在每个b、每个seed内均不下降。这是当前扫描的现象，不是普遍规律。

2. 相较固定λ=0.5，改善明确；相较所有固定策略，不存在普遍优势。
70个P>0参数组：相对固定λ=0.5，末段合作率均值23组更高、47组持平、0组更低。
b=1.03、P=1从17.57%升至100%，提高82.43个百分点。
但b=1.02、P=0.6：自适应50.49%，固定λ=1为89.72%；b=1.01、P=0.5：64.24%对99.15%。
b≥1.05、P=1：自适应0%，固定λ=1仍约9.86%。应继续保留多个固定策略对照。

3. 不能仅展示两次运行的均值。
b=1.04、P=0.9：两个seed末段合作率为0%和35.58%，均值17.79%。
b=1.04、P=1：两个seed为0%和83.72%，均值41.86%。
seed42分别在第2730、3848步灭绝；这些均值不代表每次运行维持中等合作。
P=1的存活运行最终64.48%个体信任>0.9；灭绝运行全部<0.1。跨种子混合不等于单次运行内的异质性。

4. 末段收益与前期收益不同。
相对配对Fermi，70个P>0参数组的末段全体人均收益：26组改善、44组持平。
全程人均每轮收益差：60组为正、10组为负；前200步：49组为正、21组为负。
因此不能把末段成功表述成全程、初期或每个个体都受益。

5. 信任接近1不代表收益还在提高。
规则比较最近3轮平均收益与旧的指数平滑参考收益，而不是单纯比较前后两轮。
全合作时收益固定4，参考仍可能是3.9999999999999982；严格正负比较持续产生正反馈。
例如b=1.01、P=1、seed42末段每轮约2499次正反馈，但收益已不变。
这是参考的渐近逼近和无容差比较共同造成的行为，应做死区/容差敏感性检验。
本次保留现有代码及原始结果，未重跑模拟。

建议下一步：
① 先明确“超过期望”与“比上一轮提高”的科学含义，验证数值容差对结果的影响。
② 优先在b≈1.03–1.04、P≈0.7–1增加到至少10个seed，重点b=1.04、P=0.9/1。
③ 对仍存活的边界组延长到20000–50000步，报告灭绝时间与尾段漂移；已全背叛吸收的运行不会因延长时间自发恢复。
④ 保留Fermi、固定λ=0.5/1，并增加平均实际使用强度匹配的对照，再分别扫描初始信任、α、β、W及Qwen表。

限制：只有2个seed、一张GPT建议表；旧实验未保存初始网格和资格掩码，不能逐格核验跨批初态。
信任仅保存总体分箱，没有逐格信任轨迹；收益和信任相关性不能当作单次建议的因果作用。
"""


def build(repo: Path, report: Path):
    current, previous, grouped, old_grouped = load_inputs(repo)
    report.mkdir(parents=True, exist_ok=True)
    notes = make_notes(current, grouped, old_grouped)
    (report / "speaker_notes.txt").write_text(
        "个体信任与GPT建议：组会讲稿\n\n" + "\n\n".join(
            f"第{i+1}页｜{title}\n{note}"
            for i, (title, note) in enumerate(zip(
                ["实验设置与主要问题"] + [title for _, title in FIGURES] + ["下一步实验"], notes))
        ) + "\n", encoding="utf-8")
    (report / "分析摘要.txt").write_text(chinese_summary(grouped, old_grouped), encoding="utf-8")
    for name, _ in FIGURES:
        if not (report / "figures" / name).is_file():
            raise FileNotFoundError(f"Generate the trust figures first: {name}")

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(16), Inches(9)
    prs.core_properties.title = "个体信任如何改变GPT建议的效果？"
    prs.core_properties.subject = "2026-10-29 adaptive-trust scan, two seeds per cell"
    prs.core_properties.author = "Research analysis"
    slide = new_slide(prs)
    header(slide, "2026-10-30 · 初步结果", "个体信任如何改变 GPT 建议的效果？")
    text(slide, .95, 2.05, 14, .65, "合作范围扩大；边界组仍对种子敏感", 30, TEAL, True)
    rectangle(slide, .9, 3.0, 8.6, 4.6, PALE)
    text(slide, 1.25, 3.3, 7.95, .55, "此次扫描", 25, NAVY, True)
    text(slide, 1.25, 4.0, 7.95, 2.95,
         "7 个 b × 11 个 P × 2 个 seed = 154 次运行\n"
         "50×50 格点，10000 步；末段取最后 1000 步\n"
         "初始信任 τᵢ=0.5；α=β=0.1；收益窗口 W=3\n"
         "仍使用同一张 GPT 固定建议表（每局面 2 条回复）", 22)
    text(slide, 10.1, 3.35, 5.05, .55, "核心比较", 25, NAVY, True)
    text(slide, 10.1, 4.05, 5.05, 2.7,
         "配对 Fermi 基线\n固定 λ=0.5 与 λ=1\n种子逐一展示\n信任、合作和收益一起观察", 23)
    text(slide, 1.0, 7.78, 14, .35, "n=2：描述性结果；种子范围不是置信区间；末段均值不等于稳态。", 17, MUTED)
    footer(slide, 1, "来源：scan_20261029_183749_initial · 不修改实验、不重新调用模型")
    add_notes(slide, notes[0])
    for index, (name, title) in enumerate(FIGURES, start=1):
        figure_panel(prs, report / "figures" / name, title, index+1, notes[index])

    slide = new_slide(prs)
    header(slide, "下一步", "先验证反馈定义，再集中补足边界实验")
    cards = [
        ("01  反馈规则", "明确收益超过期望的含义\n加入数值死区敏感性检验\n记录正／负／零反馈", .9, 2.35),
        ("02  增加重复", "重点 b≈1.03–1.04，P≈0.7–1\n至少 10 个 seed\n报告存活次数与灭绝时间", 8.2, 2.35),
        ("03  延长仍存活的运行", "边界组扩展至 20000–50000 步\n检查尾段漂移\n已吸收为全背叛者不会自行恢复", .9, 5.2),
        ("04  区分机制", "保留 Fermi、固定 λ=0.5/1\n加入平均使用强度匹配对照\n之后扫描 α、β、W 与其他建议表", 8.2, 5.2),
    ]
    for title, body, x, y in cards:
        rectangle(slide, x, y, 6.9, 2.5, PALE)
        text(slide, x+.3, y+.22, 6.3, .5, title, 25, TEAL, True)
        text(slide, x+.3, y+.9, 6.3, 1.45, body, 21)
    footer(slide, len(FIGURES)+2, "当前结论限于两个种子、一张 GPT 建议表；个体信任的效果仍需独立重复验证。")
    add_notes(slide, notes[-1])
    path = report / "Adaptive_Trust_Group_Meeting.pptx"
    prs.save(path)
    reopened = Presentation(path)
    if len(reopened.slides) != len(FIGURES)+2 or any(not s.notes_slide.notes_text_frame.text.strip() for s in reopened.slides):
        raise RuntimeError("Slide count or speaker notes validation failed.")
    overflow = []
    for index, slide in enumerate(reopened.slides, start=1):
        for shape in slide.shapes:
            if (shape.left < 0 or shape.top < 0 or shape.left+shape.width > reopened.slide_width
                    or shape.top+shape.height > reopened.slide_height):
                overflow.append((index, shape.name))
    if overflow:
        raise RuntimeError(f"Out-of-slide shapes: {overflow}")
    review = report / "review"
    review.mkdir(exist_ok=True)
    (review / "deck_validation.json").write_text(json.dumps({
        "slides": len(FIGURES)+2, "embedded_figures": len(FIGURES), "speaker_notes": len(notes),
        "shapes_outside_slide": overflow,
        "pptx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_summary_sha256": hashlib.sha256((repo / NEW_SCAN / "summary.csv").read_bytes()).hexdigest(),
        "validation": "Package reopened; notes and slide bounds verified. Native rendering checked separately.",
    }, indent=2), encoding="utf-8")
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-dir", type=Path, default=Path("reports/Trust_1030"))
    args = parser.parse_args()
    repo = writable_path(args.repo)
    report = writable_path(args.report_dir)
    build(repo, report)


if __name__ == "__main__":
    main()
