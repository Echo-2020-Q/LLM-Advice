"""运行空间囚徒困境实验，保存逐轮统计、最终策略网格和实验汇总。

直接修改下方“实验参数”区域，然后在 IDE 中运行本文件即可，无需填写命令行参数。
也可在项目根目录运行（不需要安装额外依赖）：
    python main/SPD/run_experiment.py
仍支持命令行覆盖代码中的参数，例如：
    python main/SPD/run_experiment.py --b 1.5 --lambda 0.2
扫描 b、K、λ、P 的所有组合：把 SCAN_ENABLED 改为 True，并修改下方扫描列表。
也可运行：
    python main/SPD/run_experiment.py --scan --b-values 1.5 --k-values 0.1 1 --lambda-values 0 0.2
REPEATS 控制每组参数运行多少次，单组和扫描模式均生效，也可传入 --repeats。
NUM_WORKERS 控制并行进程数，也可传入 --workers；1 为串行，多个独立实验可并行。
扫描子目录默认只保存 history.csv 和 final_grid.csv；SCAN_SAVE_DETAILS=True 可恢复完整记录。
默认配对运行同初态、同资格位置的纯 Fermi 对照，在原有 CSV 中记录分组收益和配对差值。
单值 --b/--k/--lambda/--p/--seed 只用于单组模式；扫描模式使用对应的 --*-values。

大模型建议来自 LLM_BATCH_DIR 中的历史回答，自动识别完整3×3布局或旧的四邻居计数格式。
3×3布局按北到南、西到东的九格策略匹配，中心是本人，不合并旋转或镜像。
四个角仅用于建议观察，收益和 Fermi 模仿仍只使用上下左右四邻居。
这是本地经验分布采样，不读取 API 密钥；固定种子可复现，不消耗 API 费用。
只有固定选中的 P 比例个体可按概率 λ 采纳建议；其余个体始终执行 Fermi 更新。
建议匹配不使用 b，不同 b 的实验共用同一批数据；博弈收益仍使用各组实验的 b。
实际有人可咨询且 λ>0 时使用四邻居周期边界。
λ=0 或实际资格人数为 0 时不读取建议数据。
history.csv 的建议/Fermi/策略变化计数均是本轮计数；round=0 为初始状态。
每次运行创建独立输出目录，不覆盖已有实验。更换 SEED 可进行独立重复实验。
"""

import argparse
import csv
import json
import math
import multiprocessing
import sys
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime
from itertools import product
from pathlib import Path


# ==================== 实验参数：在这里直接修改 ====================
# 不传命令行参数时，使用下列数值；显式传入的命令行参数会覆盖对应数值。

# 方格边长 L，必须为至少 3 的整数；总个体数为 L × L。
SIZE = 50

# 仿真时间步数，必须为非负整数；0 表示只保存初始状态。
# 异步模式每步随机抽取 L² 次个体（有放回）；同步模式每步所有个体更新一次。
STEPS = 10000

# 每组参数独立运行的次数，必须为正整数；单组模式和扫描模式都生效。
# 例如 REPEATS=10 表示每组运行10次，每次都演化 STEPS 步。
# 默认第 r 次使用种子 SEED+r-1（r从1开始），重新初始化策略、资格人群和建议抽样器。
REPEATS = 20

# 同时运行的进程数，必须为正整数；1 为串行，4 表示最多同时运行4次实验。
# 扫描和单组重复均生效；实际进程数不超过总运行次数，只有1次实验时直接串行。
# 每个进程负责一次完整实验及其配对 Fermi 对照，不拆分单次实验内部的格点更新。
# 建议不超过电脑的 CPU 逻辑核心数，并为其他程序留一些资源；进程越多，占用内存越多。
# Windows 实际进程数最多61；固定种子下，修改此参数不改变每次实验的结果。
NUM_WORKERS = 16

# 背叛诱惑收益 b，要求 b > 1；收益矩阵固定为 R=1、S=0、T=b、P=0。
# 个体收益为与所有邻居博弈的收益之和，不是跨时间步累加。
B = 1.5

# Fermi 更新的噪声强度 K，必须为正数。
# K 越小越倾向模仿高收益邻居；K 越大，模仿概率越接近 1/2。
K = 1

# 有咨询资格的个体，每次更新时使用大模型建议的概率 λ，范围 [0, 1]。
# 0 表示纯 Fermi；1 表示有资格者每次更新都抽取建议；无资格者始终执行 Fermi。
# 每次抽样只匹配当前局面；历史回答不耗尽，也不沿用该个体上一次得到的建议。
LAMBDA_PROB = 0.1 # λ

# 具备 LLM 咨询资格的个体比例，范围 [0, 1]；不是收益矩阵中的惩罚收益 P=0。
# 开始时随机选定 floor(P × SIZE²) 个个体，向下取整，整次实验资格保持不变。
# P=1 表示全体都有资格（保持原有行为）；P=0 表示完全不使用 LLM 建议。
# 例：SIZE=50、P=0.2 时，固定 500 个体有资格，各自再按 LAMBDA_PROB 决定是否咨询。
# 资格不会随策略模仿转移；其他个体仍可能通过 Fermi 模仿间接受到建议影响。
P = 0.2

# 本地 LLM 批次目录，读取 config.json 和 results.csv；不会修改原始结果。
# 自动读取批次 observation_format：grid_3x3 按完整九格布局匹配（512种），
# neighbor_counts 或旧批次按自身策略和四邻居计数匹配（10种）。
# 3×3包含本人和周围八格，保留方向；四角只供观察，收益/Fermi仍用四邻居。
# 旧批次的 b 仅作来源记录，不参与建议匹配；各实验的博弈收益仍由 B/SCAN_B_VALUES 决定。
LLM_BATCH_DIR = (Path(__file__).resolve().parents[2] / "logs" / "GPT_Multistepsa"
                 / "20261016_173605_322949_5d224b193871427f8c133150d99a6599")

# 各个体初始选择合作 C 的概率，范围 [0, 1]；否则初始化为背叛 D。
# 0 为全 D，1 为全 C；随机初始化的实际合作比例不一定恰好等于该值。
INITIAL_COOPERATION = 0.5

# 起始随机种子，必须为整数；相同参数和实际种子可复现当前实验的随机过程。
# 例如 SEED=42、REPEATS=3，实际种子为42、43、44；各参数组使用同一组种子作对照。
SEED = 42

# 空间边界："periodic" 为周期边界，上下、左右相连，每人都有四个邻居；
# "open" 为开放边界，角点只有两个邻居、其他边缘点三个，收益不按邻居数归一化。
BOUNDARY = "periodic"

# 更新方式："asynchronous" 为随机异步更新，抽中个体后立即更新策略；
# 每步有放回抽样 L² 次，同一个体可能多次被抽中，也可能未被抽中。
# "synchronous" 为同步更新，所有个体依据同一旧状态决策，然后统一替换策略。
UPDATE_MODE = "synchronous"

# 结果根目录，默认是本文件旁的 results 文件夹；可改为 Path("你的目录")。
# 每次运行会在其下新建独立目录；扫描子目录的文件数量由 SCAN_SAVE_DETAILS 控制。
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# 每隔多少个时间步打印一次进度，必须为非负整数；0 表示不打印中途进度。
# 仅影响控制台显示，逐步统计仍全部保存到 history.csv。
# 多进程时仅由主进程打印每次实验完成的进度，避免各进程的逐步输出混杂。
REPORT_EVERY = 100

# 是否启用参数扫描：False 只运行上方单组参数；True 遍历下面列表的所有组合。
# 总运行次数 = b 值数 × K 值数 × λ 值数 × P 值数 × REPEATS。
SCAN_ENABLED = True

# 扫描时是否保存完整明细；只影响输出文件，不改变仿真过程和汇总统计。
# False：每个子目录只保存 history.csv（逐轮统计）和 final_grid.csv（最终策略网格）。
# 参数、种子和汇总结果统一保存在扫描根目录的 scan_config.json、summary.csv。
# True：额外保存 config.json、initial_grid.csv、llm_access_grid.csv，
# 使用本地建议时还保存 advisor_table.csv、advisor_usage.csv。
# 单组模式（含 REPEATS 重复）始终保存完整明细；此设置仅对之后新运行的扫描生效。
SCAN_SAVE_DETAILS = False

# 要扫描的 b 值，每个值必须 > 1；扫描时替代上方 B。
# 不同 b 共用本地 LLM 建议表，仅改变博弈收益及依赖收益的 Fermi 更新概率。
SCAN_B_VALUES = [1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.2]

# 要扫描的 K 值，每个值必须 > 0；扫描时替代上方 K。
# 不扫描 K 时可写成 [K]，即只使用上方设置的 K。
SCAN_K_VALUES = [0.1]

# 要扫描的 λ 值，每个值必须在 [0, 1] 内；扫描时替代上方 LAMBDA_PROB。
# 不扫描 λ 时可写成 [LAMBDA_PROB]。
# λ=0 为纯 Fermi 对照；λ>0 以对应概率采纳本地 LLM 抽样建议。
SCAN_LAMBDA_VALUES = [0.0, 0.1, 0.5, 1.0]

# 要扫描的资格比例 P，各值须在 [0, 1]；如 [0.0, 0.2, 0.5, 1.0]。
# [P] 表示不扫描此参数，只使用上方 P；每组实验都单独选定并保存固定资格网格。
SCAN_P_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                 0.6, 0.7, 0.8, 0.9, 1.0]


# 扫描种子列表的高级设置：None 表示自动使用 SEED 开始的 REPEATS 个连续整数。
# 通常只需修改 REPEATS，无需修改这里；也可手动指定例如 [42, 100, 200]。
# 手动列表长度必须等于 REPEATS，且种子不能重复；它替代自动种子，不会额外增加次数。
# 每次重复都重新初始化，不继承上一次的网格。单组模式始终从 SEED 自动生成种子。
SCAN_SEEDS = None

# 汇总时取最后多少个更新步计算合作率/收益的时间平均，必须为正整数。
# 不包含第 0 步；不足此数量时使用实际步数。STEPS=0 时仅统计第 0 步。
# 这是有限时间的末段平均，不代表已自动判定达到稳态；单次模式也会记录此统计。
TAIL_AVERAGE_STEPS = 1000

# 是否为每次实验配对运行纯 Fermi 对照；True 会增加一次相同步数的仿真。
# 对照仅把 λ 设为0，保留 b、K、种子、初始网格和固定资格位置，用同一批位置比较收益。
# False 仍统计实验组收益，但 control_ / delta_ 列为空；不会额外调用 LLM 或 API。
PAIRED_FERMI_CONTROL = True

# 前期收益平均窗口：第1步到此步（不足时取实际步数），必须为正整数。
# 末期窗口使用 TAIL_AVERAGE_STEPS；短实验两窗口可能重叠，会在汇总中标记。
# 累计收益按每个完整时间步结束后的收益累加，第0步只展示初态、不计入累计。
# 这些统计不改变 Fermi 使用的当前收益。空组的人均指标为空，不用0替代。
EARLY_AVERAGE_STEPS = 200
# ==================== 实验参数结束 ====================


if __package__ in (None, ""):
    # 同时支持在 IDE 直接运行本文件和 python -m 包运行。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from SPD.llm_advisor import EmpiricalAdvisor, request_advice
    from SPD.simulation import SimulationConfig, SpatialPrisonersDilemma
    from SPD.experiment_metrics import PayoffTracker, GROUPS, HISTORY_FIELDS, SUMMARY_FIELDS
    from SPD.parallel_execution import ParallelRunError, run_parallel
else:
    from .llm_advisor import EmpiricalAdvisor, request_advice
    from .simulation import SimulationConfig, SpatialPrisonersDilemma
    from .experiment_metrics import PayoffTracker, GROUPS, HISTORY_FIELDS, SUMMARY_FIELDS
    from .parallel_execution import ParallelRunError, run_parallel


GAIN_FIELDS = tuple(
    f"{group}_{metric}" for group in GROUPS
    for metric in ("cumulative_mean_payoff", "time_mean_payoff", "early_mean_payoff", "tail_mean_payoff")
) + ("time_mean_cooperation_rate",)
CONTROL_OUTCOME_FIELDS = (
    "final_cooperation_rate", "final_average_payoff", "tail_mean_cooperation_rate", "tail_mean_payoff",
)
STATISTICS_SEMANTICS = {
    "payoffs": "每完整步轮末，每个位置测一次当前邻居收益；累计round1..T，不计round0，不改变更新收益。",
    "groups": "eligible/ineligible按初始化的固定资格位置划分，population为全体；不是按实际咨询次数分组。",
    "cumulative_total_payoff": "组内所有位置、所有观测步的收益之和。",
    "cumulative_mean_payoff": "组内每人累计收益的平均值，等于累计总收益/固定组人数。",
    "time_mean_payoff": "组内每人每步的平均收益，等于人均累计收益/观测步数。",
    "windows": "early为最初EARLY_AVERAGE_STEPS步，tail为最后TAIL_AVERAGE_STEPS步；均排除round0。",
    "empty_values": "空组人均指标、0步时间平均、未观测到的事件时间、关闭的对照记JSON null/CSV空白。",
    "pairing": "对照仅lambda=0，同种子初始化且核验同初态/同资格位置；两模型各自演化完整T步。",
    "randomness": "模型分别持有随机流；干预后随机数消费可能不同，不保证逐更新事件相同。",
    "delta": "所有delta_为实验值减去相同位置分组的纯Fermi对照值；包含邻居传播带来的影响。",
    "extinction": "first_extinction_round为首次观测到全D（含初态0）；仅逐完整步观测，不表示永久吸收。",
    "revivals": "cooperation_revivals记录观测到从全D恢复合作的次数；extinction_events含初始全D。",
    "final_extinction": "final_extinction_start_round仅在末态全D时记录最后连续全D段的起点。",
    "replicates": "每行对应一组参数的一次重复；先取同seed配对差值，格点和时间步不当作独立重复。",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=SIZE, help="方格边长 L（至少 3）")
    parser.add_argument("--steps", type=int, default=STEPS, help="时间步数（非负整数）")
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help="每组参数独立运行次数（正整数），单组和扫描模式均生效")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help="并行进程数（正整数）；1为串行，实际不超过总运行次数")
    # 单值默认延迟到构建配置时读取，以识别扫描时误传的单值参数。
    parser.add_argument("--b", type=float, default=None, help="单组模式的 T=b，默认使用代码中的 B")
    parser.add_argument("--k", type=float, default=None, help="单组模式的 K，默认使用代码中的 K")
    parser.add_argument("--lambda", dest="lambda_prob", type=float, default=None,
                        help="单组模式中有资格者的建议概率 λ，默认使用 LAMBDA_PROB")
    parser.add_argument("--p", dest="llm_access_fraction", type=float, default=None,
                        help="单组模式的 LLM 咨询资格比例 P，范围 [0,1]，默认使用代码中的 P")
    parser.add_argument("--initial-cooperation", type=float, default=INITIAL_COOPERATION,
                        help="各格点独立初始化为 C 的概率，不保证比例恰好相等")
    parser.add_argument("--seed", type=int, default=None, help="单组模式的起始种子，默认使用代码中的 SEED")
    parser.add_argument("--boundary", choices=("periodic", "open"), default=BOUNDARY,
                        help="periodic 周期边界；open 开放边界，边缘邻居较少（不归一化收益）")
    parser.add_argument("--update-mode", choices=("asynchronous", "synchronous"), default=UPDATE_MODE,
                        help="随机异步：每步 L² 次有放回抽样；同步则每步全员更新一次")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="结果根目录；其下自动创建本次运行的独立子目录")
    parser.add_argument("--advisor-batch", type=Path, default=LLM_BATCH_DIR,
                        help="本地 LLM 批次目录；λ=0 或实际资格人数为0时不读取")
    parser.add_argument("--report-every", type=int, default=REPORT_EVERY, help="打印进度的步数间隔；0 表示不打印中途进度")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scan", dest="scan", action="store_true", help="启用参数扫描")
    mode.add_argument("--single", dest="scan", action="store_false", help="只使用单组参数，仍按 repeats 重复")
    parser.set_defaults(scan=SCAN_ENABLED)
    parser.add_argument("--scan-save-details", action=argparse.BooleanOptionalAction,
                        default=SCAN_SAVE_DETAILS,
                        help="扫描时保存完整明细；--no-scan-save-details 仅保留逐轮统计和最终网格")
    parser.add_argument("--b-values", type=float, nargs="+", help="扫描的 b 列表，覆盖 SCAN_B_VALUES")
    parser.add_argument("--k-values", type=float, nargs="+", help="扫描的 K 列表，覆盖 SCAN_K_VALUES")
    parser.add_argument("--lambda-values", type=float, nargs="+", help="扫描的 λ 列表，覆盖 SCAN_LAMBDA_VALUES")
    parser.add_argument("--p-values", type=float, nargs="+", help="扫描的资格比例 P 列表，覆盖 SCAN_P_VALUES")
    parser.add_argument("--seed-values", type=int, nargs="+",
                        help="扫描的显式种子列表，长度必须等于 repeats，覆盖 SCAN_SEEDS")
    parser.add_argument("--tail-average-steps", type=int, default=TAIL_AVERAGE_STEPS,
                        help="末段平均使用的步数；不足时按实际步数，0步实验取初态")
    parser.add_argument("--early-average-steps", type=int, default=EARLY_AVERAGE_STEPS,
                        help="前期收益平均的步数，排除第0步，不足时取实际步数")
    parser.add_argument("--paired-fermi-control", action=argparse.BooleanOptionalAction,
                        default=PAIRED_FERMI_CONTROL, help="配对运行同初态和同资格位置的纯Fermi对照")
    return parser


def write_grid(path, grid):
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(grid)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_advisor_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_advisor(config, batch_dir):
    """纯 Fermi 不加载表；有建议的实验在创建输出目录前验证适用范围。"""
    if config.lambda_prob == 0 or config.llm_access_count == 0:
        return None
    advisor = EmpiricalAdvisor(batch_dir, seed=config.seed)
    advisor.validate_config(config.b, config.boundary)
    return advisor


def repeat_seeds(base_seed, repeats, explicit_seeds=None):
    """一组参数恰好使用 repeats 个互异种子，显式列表替代自动生成。"""
    if explicit_seeds is None:
        if type(base_seed) is not int:
            raise ValueError("起始 SEED 必须是整数。")
        seeds = list(range(base_seed, base_seed + repeats))
    else:
        if not isinstance(explicit_seeds, (list, tuple)) or len(explicit_seeds) != repeats:
            raise ValueError("显式种子列表的长度必须等于 REPEATS / --repeats。")
        if any(type(seed) is not int for seed in explicit_seeds):
            raise ValueError("种子列表中的值必须是整数。")
        seeds = list(explicit_seeds)
    if len(set(seeds)) != repeats:
        raise ValueError("重复实验的种子不能重复。")
    # random.Random 对整数种子使用其绝对值；正负成对会重复初始化/Fermi随机流。
    if len({abs(seed) for seed in seeds}) != repeats:
        raise ValueError("重复实验不能同时使用互为相反数的种子（如 -42 和 42）；建议使用非负起始种子。")
    return seeds


def prepare_configs(args):
    """先验证全部组合，再开始写文件，防止坏参数导致扫描运行到一半才报错。"""
    if (type(args.steps) is not int or args.steps < 0
            or type(args.report_every) is not int or args.report_every < 0):
        raise ValueError("steps 和 report-every 必须为非负整数。")
    if type(args.tail_average_steps) is not int or args.tail_average_steps <= 0:
        raise ValueError("tail-average-steps 必须为正整数。")
    if type(args.early_average_steps) is not int or args.early_average_steps <= 0:
        raise ValueError("early-average-steps 必须为正整数。")
    if type(args.paired_fermi_control) is not bool:
        raise ValueError("PAIRED_FERMI_CONTROL 必须为 True 或 False。")
    if type(args.repeats) is not int or args.repeats <= 0:
        raise ValueError("REPEATS / --repeats 必须为正整数。")
    if type(args.workers) is not int or args.workers <= 0:
        raise ValueError("NUM_WORKERS / --workers 必须为正整数。")
    if type(args.scan) is not bool:
        raise ValueError("SCAN_ENABLED 必须为 True 或 False。")
    if type(args.scan_save_details) is not bool:
        raise ValueError("SCAN_SAVE_DETAILS 必须为 True 或 False。")

    parameters = {
        "size": args.size,
        "b": B if args.b is None else args.b,
        "k": K if args.k is None else args.k,
        "lambda_prob": LAMBDA_PROB if args.lambda_prob is None else args.lambda_prob,
        "llm_access_fraction": P if args.llm_access_fraction is None else args.llm_access_fraction,
        "initial_cooperation": args.initial_cooperation,
        "seed": SEED if args.seed is None else args.seed,
        "boundary": args.boundary,
        "update_mode": args.update_mode,
    }
    if not args.scan:
        if any(value is not None for value in (
            args.b_values, args.k_values, args.lambda_values, args.p_values, args.seed_values,
        )):
            raise ValueError("扫描列表参数需要启用 --scan；单组运行请用 --b/--k/--lambda/--p/--seed。")
        seeds = repeat_seeds(parameters["seed"], args.repeats)
        return [SimulationConfig(**dict(parameters, seed=seed)) for seed in seeds], None

    if any(value is not None for value in (args.b, args.k, args.lambda_prob, args.llm_access_fraction, args.seed)):
        raise ValueError("扫描模式请用 --b-values/--k-values/--lambda-values/--p-values/--seed-values，"
                         "或修改代码中的扫描列表；单值参数仅适用于 --single。")
    scan_values = {
        "b": SCAN_B_VALUES if args.b_values is None else args.b_values,
        "k": SCAN_K_VALUES if args.k_values is None else args.k_values,
        "lambda_prob": SCAN_LAMBDA_VALUES if args.lambda_values is None else args.lambda_values,
        "llm_access_fraction": SCAN_P_VALUES if args.p_values is None else args.p_values,
        "seed": repeat_seeds(SEED, args.repeats,
                             SCAN_SEEDS if args.seed_values is None else args.seed_values),
    }
    for name, values in scan_values.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"扫描参数 {name} 必须是非空列表，例如 [0.1, 0.2]。")
        if len(set(values)) != len(values):
            raise ValueError(f"扫描参数 {name} 的列表不能包含重复值，以免同一参数组被重复计数。")
    configs = []
    for b, k, lambda_prob, access_fraction, seed in product(*scan_values.values()):
        configs.append(SimulationConfig(**dict(
            parameters, b=b, k=k, lambda_prob=lambda_prob,
            llm_access_fraction=access_fraction, seed=seed,
        )))
    return configs, scan_values


def difference(value, control):
    """缺失的对照或空组不填0，避免产生虚假的收益优势。"""
    return value - control if value is not None and control is not None else None


def welfare_history(stats, tracker, control_stats=None, control_tracker=None):
    """同一步的实验、对照和差值合并在一行，不增加扫描子目录文件。"""
    row = asdict(stats)
    values = tracker.observe(stats)
    baseline = control_tracker.observe(control_stats) if control_tracker is not None else {}
    row.update(values)
    for name in ("cooperation_rate", "average_payoff"):
        value = getattr(control_stats, name) if control_stats is not None else None
        row["control_" + name] = value
        row["delta_" + name] = difference(row[name], value)
    for name in HISTORY_FIELDS:
        row["control_" + name] = baseline.get(name)
        row["delta_" + name] = difference(values[name], baseline.get(name))
    return row


def welfare_summary(tracker, control_tracker=None):
    values = tracker.summary()
    baseline = control_tracker.summary() if control_tracker is not None else {}
    result = dict(values)
    result.update({"control_" + name: baseline.get(name) for name in SUMMARY_FIELDS})
    result.update({"delta_" + name: difference(values[name], baseline.get(name)) for name in GAIN_FIELDS})
    steps = values["observed_steps"]
    early_end = values["early_observed_steps"] if steps else None
    tail_start = steps - values["tail_observed_steps"] + 1 if steps else None
    result.update(
        early_start_round=1 if steps else None,
        early_end_round=early_end,
        payoff_tail_start_round=tail_start,
        payoff_tail_end_round=steps if steps else None,
        payoff_windows_overlap=bool(steps and early_end >= tail_start),
    )
    return result


def run_single(config, steps, output, report_every, tail_average_steps, advisor=None,
               parameter_set_id=1, repeat_index=1, repeats=1, save_details=True,
               paired_control=True, early_average_steps=100, quiet=False, cancel_check=None):
    """独立初始化并返回汇总；精简输出仅保存逐轮统计和最终网格。"""
    def check_cancelled():
        if cancel_check is not None and cancel_check():
            raise InterruptedError("批次已停止，取消本次实验。")

    check_cancelled()
    simulation = SpatialPrisonersDilemma(config, advisor=advisor if advisor is not None else request_advice)
    # 让两模型各自用同种子生成初态，不传initial_grid，以保持初始化对随机流的正常消耗。
    control_config = replace(config, lambda_prob=0.0) if paired_control else None
    control = SpatialPrisonersDilemma(control_config) if paired_control else None
    if control is not None and (
        simulation.grid != control.grid or simulation.llm_access_grid != control.llm_access_grid
    ):
        raise RuntimeError("配对对照的初始策略网格或固定资格网格不一致。")
    tracker = PayoffTracker(simulation, early_average_steps, tail_average_steps)
    control_tracker = PayoffTracker(control, early_average_steps, tail_average_steps) if paired_control else None
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "config": asdict(config),
        "parameter_set_id": parameter_set_id,
        "repeat_index": repeat_index,
        "repeats": repeats,
        "steps": steps,
        "tail_average_steps": tail_average_steps,
        "early_average_steps": early_average_steps,
        "paired_fermi_control": paired_control,
        "control_config": asdict(control_config) if control_config is not None else None,
        "statistics_semantics": STATISTICS_SEMANTICS,
        "payoff_matrix": {"R": 1, "S": 0, "T": config.b, "P": 0},
        "advice_rule": "fixed_eligible_subset_then_probability_lambda; others_fermi; None_falls_back_to_fermi",
        "llm_access": {
            "rule": "fixed_random_subset",
            "requested_fraction": config.llm_access_fraction,
            "rounding": "floor(P * size**2); decimal_string_value",
            "eligible_count": config.llm_access_count,
            "actual_fraction": config.llm_access_count / config.size ** 2,
            "grid_file": "llm_access_grid.csv",
            "grid_encoding": "1=eligible, 0=ineligible; no_header; same_positions_as_strategy_grid",
        },
        "advisor": (f"{type(advisor).__module__}.{type(advisor).__qualname__}" if advisor is not None
                    else f"{request_advice.__module__}.{request_advice.__qualname__}"),
        "advisor_source": advisor.metadata() if advisor is not None else {"source": "disabled_or_placeholder"},
        "round_counters": "per_round; round_0_is_initial_state",
        "status": "running",
        "completed_steps": 0,
    }
    config_path = output / "config.json"
    if save_details:
        write_json(config_path, metadata)
    if not quiet:
        print(f"输出目录：{output}", flush=True)
        print(f"LLM 咨询资格：{config.llm_access_count}/{config.size ** 2} 个体，"
              f"P={config.llm_access_fraction}，有资格者每次咨询概率 λ={config.lambda_prob}", flush=True)
    try:
        check_cancelled()
        if save_details:
            write_grid(output / "initial_grid.csv", simulation.grid)
            write_grid(output / "llm_access_grid.csv",
                       [[int(value) for value in row] for row in simulation.llm_access_grid])
            if advisor is not None:
                write_advisor_rows(output / "advisor_table.csv", advisor.table_rows())
        initial = simulation.measure()
        final = initial
        control_final = control.measure() if control is not None else None
        tail = deque(maxlen=tail_average_steps)
        control_tail = deque(maxlen=tail_average_steps)
        totals = dict(advice_requests=0, advice_used=0, fermi_updates=0, strategy_changes=0)
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as file:
            initial_row = welfare_history(initial, tracker, control_final, control_tracker)
            writer = csv.DictWriter(file, fieldnames=list(initial_row))
            writer.writeheader()
            writer.writerow(initial_row)
            for _ in range(steps):
                # 在完整实验/对照步之间响应取消，保证保存的每一行仍是一对完整观测。
                check_cancelled()
                final = simulation.step()
                if control is not None:
                    control_final = control.step()
                    control_tail.append(control_final)
                writer.writerow(welfare_history(final, tracker, control_final, control_tracker))
                tail.append(final)
                for name in totals:
                    totals[name] += getattr(final, name)
                metadata["completed_steps"] = final.round
                if report_every and final.round % report_every == 0:
                    if not quiet:
                        print(f"步数 {final.round}/{steps} | 合作率 {final.cooperation_rate:.4f} "
                              f"| 平均收益 {final.average_payoff:.4f}", flush=True)
                    file.flush()
        write_grid(output / "final_grid.csv", simulation.grid)
        if not tail:
            tail.append(initial)
        if control is not None and not control_tail:
            control_tail.append(control_final)
        summary = dict(
            asdict(config),
            parameter_set_id=parameter_set_id,
            repeat_index=repeat_index,
            repeats=repeats,
            llm_access_count=config.llm_access_count,
            actual_llm_access_fraction=config.llm_access_count / config.size ** 2,
            steps=steps,
            initial_cooperation_rate=initial.cooperation_rate,
            final_cooperation_rate=final.cooperation_rate,
            final_average_payoff=final.average_payoff,
            tail_average_steps=len(tail),
            tail_start_round=tail[0].round,
            tail_end_round=tail[-1].round,
            tail_mean_cooperation_rate=math.fsum(item.cooperation_rate / len(tail) for item in tail),
            tail_mean_payoff=math.fsum(item.average_payoff / len(tail) for item in tail),
            status="completed",
            paired_fermi_control=paired_control,
            early_average_steps=early_average_steps,
        )
        summary.update({"total_" + name: value for name, value in totals.items()})
        summary.update(welfare_summary(tracker, control_tracker))
        baseline_outcomes = {
            "final_cooperation_rate": control_final.cooperation_rate if control is not None else None,
            "final_average_payoff": control_final.average_payoff if control is not None else None,
            "tail_mean_cooperation_rate": (math.fsum(item.cooperation_rate / len(control_tail)
                                                     for item in control_tail) if control is not None else None),
            "tail_mean_payoff": (math.fsum(item.average_payoff / len(control_tail)
                                          for item in control_tail) if control is not None else None),
        }
        for name, value in baseline_outcomes.items():
            summary["control_" + name] = value
            summary["delta_" + name] = difference(summary[name], value)
        metadata["summary"] = summary
        metadata["status"] = "completed"
    except BaseException:
        # 中断或接口错误留下的部分轨迹不能被误认为完整实验。
        metadata["status"] = "failed_or_interrupted"
        raise
    finally:
        try:
            if save_details and advisor is not None:
                write_advisor_rows(output / "advisor_usage.csv", advisor.usage_rows())
        except OSError:
            metadata["status"] = "failed_or_interrupted"
            raise
        finally:
            if save_details:
                write_json(config_path, metadata)

    if not quiet:
        print(f"实验完成：{final.round} 步，最终合作率 {final.cooperation_rate:.4f}，"
              f"末段平均合作率 {summary['tail_mean_cooperation_rate']:.4f}，"
              f"平均收益 {final.average_payoff:.4f}", flush=True)
    return summary


def run_batch(args, configs, scan_values, advisors):
    """执行参数扫描或单组重复；每完成一次立即保存汇总，中断时保留已完成结果。"""
    mode = "scan" if args.scan else "repeat"
    label = "扫描" if args.scan else "重复实验"
    save_details = not args.scan or args.scan_save_details
    actual_workers = min(args.workers, len(configs))
    output = args.output_dir / (mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "mode": mode,
        "save_details": save_details,
        "scan_values": scan_values,
        "repeats": args.repeats,
        "parameter_sets": len(configs) // args.repeats,
        "seed_values": [config.seed for config in configs[:args.repeats]],
        "seed_policy": ("explicit_list" if args.scan and (args.seed_values is not None or SCAN_SEEDS is not None)
                        else "base_seed_plus_repeat_index_minus_one"),
        "fixed_parameters": {
            name: getattr(configs[0], name)
            for name in ("size", "initial_cooperation", "boundary", "update_mode")
        },
        "steps": args.steps,
        "advisor_batch": str(args.advisor_batch.resolve()),
        "advisor_sources": [advisor.metadata() for advisor in advisors if advisor is not None],
        "tail_average_steps": args.tail_average_steps,
        "early_average_steps": args.early_average_steps,
        "paired_fermi_control": args.paired_fermi_control,
        "control_lambda_prob": 0.0 if args.paired_fermi_control else None,
        "statistics_semantics": STATISTICS_SEMANTICS,
        "summary_semantics": "one_row_per_run; tail_excludes_round_0_unless_steps_is_0; counters_are_totals",
        "output_dir_semantics": "summary_paths_relative_to_batch_directory",
        "total_runs": len(configs),
        "total_control_runs": len(configs) if args.paired_fermi_control else 0,
        "requested_workers": args.workers,
        "actual_workers": actual_workers,
        "parallel_start_method": "spawn" if actual_workers > 1 else None,
        "completed_runs": 0,
        "completed_run_ids": [],
        "failed_run_id": None,
        "status": "running",
    }
    config_path = output / (mode + "_config.json")
    write_json(config_path, metadata)
    print(f"{label}开始：{metadata['parameter_sets']} 组参数，每组 {args.repeats} 次，"
          f"共 {len(configs)} 次运行；汇总文件：{output / 'summary.csv'}", flush=True)
    print(f"并行进程数：{actual_workers}（设置值 {args.workers}）；"
          "每个任务包含一次实验及其启用的配对对照。", flush=True)
    if not save_details:
        print("扫描使用精简输出：每个子目录仅保存 history.csv 和 final_grid.csv。", flush=True)
    if args.paired_fermi_control:
        print("每次实验额外配对一条同初态纯Fermi对照轨迹，统计写入同一份 history.csv 和 summary.csv。", flush=True)
    fields = [
        "run_id", "parameter_set_id", "repeat_index", "repeats",
        "b", "k", "lambda_prob", "llm_access_fraction", "llm_access_count",
        "actual_llm_access_fraction", "seed", "size", "boundary", "update_mode",
        "initial_cooperation", "steps", "initial_cooperation_rate", "final_cooperation_rate",
        "final_average_payoff", "tail_average_steps", "tail_start_round", "tail_end_round",
        "tail_mean_cooperation_rate", "tail_mean_payoff", "total_advice_requests",
        "total_advice_used", "total_fermi_updates", "total_strategy_changes", "output_dir", "status",
    ]
    fields += ["paired_fermi_control", "early_average_steps"] + list(SUMMARY_FIELDS)
    fields += ["control_" + name for name in SUMMARY_FIELDS]
    fields += ["delta_" + name for name in GAIN_FIELDS]
    fields += ["early_start_round", "early_end_round", "payoff_tail_start_round",
               "payoff_tail_end_round", "payoff_windows_overlap"]
    fields += [prefix + name for name in CONTROL_OUTCOME_FIELDS for prefix in ("control_", "delta_")]
    tasks = []
    for run_id, config in enumerate(configs, start=1):
        parameter_set_id = (run_id - 1) // args.repeats + 1
        repeat_index = (run_id - 1) % args.repeats + 1
        name = (f"run_{run_id:04d}_b{config.b}_k{config.k}"
                f"_lambda{config.lambda_prob}_p{config.llm_access_fraction}"
                f"_repeat{repeat_index:03d}_seed{config.seed}")
        tasks.append((run_id, dict(
            config=config, steps=args.steps, output=output / name,
            report_every=args.report_every, tail_average_steps=args.tail_average_steps,
            advisor=advisors[run_id - 1], parameter_set_id=parameter_set_id,
            repeat_index=repeat_index, repeats=args.repeats,
            save_details=save_details, paired_control=args.paired_fermi_control,
            early_average_steps=args.early_average_steps,
        )))

    completed_rows = {}

    def save_summary():
        # 只有主进程写汇总；任务即使乱序完成，也按原run_id保存。
        # 临时文件关闭后再替换，避免中断重写时损坏此前已完成的汇总。
        temporary = output / "summary.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(completed_rows[run_id] for run_id in sorted(completed_rows))
        temporary.replace(output / "summary.csv")

    def on_result(run_id, summary):
        name = tasks[run_id - 1][1]["output"].name
        completed_rows[run_id] = dict(summary, run_id=run_id, output_dir=name)
        save_summary()
        metadata["completed_runs"] = len(completed_rows)
        metadata["completed_run_ids"] = sorted(completed_rows)
        write_json(config_path, metadata)
        print(f"[已完成 {len(completed_rows)}/{len(configs)}] run_{run_id:04d}，"
              f"b={summary['b']}, K={summary['k']}, λ={summary['lambda_prob']}, "
              f"P={summary['llm_access_fraction']}, seed={summary['seed']} | "
              f"末段平均合作率 {summary['tail_mean_cooperation_rate']:.4f}", flush=True)

    active_run_id = None
    try:
        save_summary()
        if actual_workers > 1:
            run_parallel(tasks, actual_workers, on_result)
        else:
            for run_id, kwargs in tasks:
                active_run_id = run_id
                print(f"[{run_id}/{len(configs)}] 参数组 {kwargs['parameter_set_id']}，"
                      f"重复 {kwargs['repeat_index']}/{args.repeats}", flush=True)
                on_result(run_id, run_single(**kwargs))
                active_run_id = None
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata["status"] = "failed_or_interrupted"
        metadata["failed_run_id"] = exc.run_id if isinstance(exc, ParallelRunError) else active_run_id
        # Ctrl+C 可能恰好打断某次成功回调的落盘；工作进程退出后再次提交已收集结果。
        # 回调不能重试（可能有其它副作用），这里只恢复主进程掌握的汇总快照。
        try:
            save_summary()
            metadata["completed_runs"] = len(completed_rows)
            metadata["completed_run_ids"] = sorted(completed_rows)
        except Exception as save_error:
            # 例如磁盘写入失败时保留原始异常，避免将实验失败原因替换为收尾错误。
            print(f"汇总收尾保存失败：{save_error}", file=sys.stderr, flush=True)
        raise
    finally:
        write_json(config_path, metadata)
    print(f"{label}完成：{len(configs)} 次运行，结果见 {output / 'summary.csv'}", flush=True)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configs, scan_values = prepare_configs(args)
        if sys.platform == "win32" and min(args.workers, len(configs)) > 61:
            raise ValueError("Windows 最多支持61个并行进程，请减小 NUM_WORKERS / --workers。")
        # 所有组合预检完成后再开始扫描，避免数据缺失或边界不支持造成半批实验。
        advisors = [create_advisor(config, args.advisor_batch) for config in configs]
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        parser.error(str(exc))
    if any(advisor is not None for advisor in advisors):
        formats = sorted({getattr(advisor, "observation_format", "custom")
                          for advisor in advisors if advisor is not None})
        format_names = {"grid_3x3": "完整3×3布局（512种，保留方位）",
                        "neighbor_counts": "自身策略与四邻居计数（10种）", "custom": "自定义建议接口"}
        print(f"本地 LLM 建议：{args.advisor_batch.resolve()}\n"
              f"建议匹配：{'、'.join(format_names[name] for name in formats)}；不使用 b。\n"
              "收益和Fermi模仿使用上下左右四邻居，收益按实验 b 计算。\n"
              "固定资格个体以概率 λ 按当前局面抽取并采纳历史建议；其余更新执行 Fermi。", flush=True)
    else:
        print("本次不使用本地 LLM 建议数据。", flush=True)
    if args.scan or args.repeats > 1:
        run_batch(args, configs, scan_values, advisors)
    else:
        config = configs[0]
        output = args.output_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{config.seed}")
        run_single(config, args.steps, output, args.report_every, args.tail_average_steps, advisor=advisors[0],
                   paired_control=args.paired_fermi_control, early_average_steps=args.early_average_steps)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
