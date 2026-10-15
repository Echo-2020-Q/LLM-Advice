"""运行空间囚徒困境实验，保存配置、逐轮统计以及初始/最终策略网格。

项目根目录运行（不需要安装额外依赖）：
    python main/SPD/run_experiment.py --size 50 --steps 1000 --b 1.5 --k 0.1 --lambda 0.2
或：
    python -m main.SPD.run_experiment --update-mode synchronous --seed 123

大模型接口在 llm_advisor.py，目前返回 None，触发后回退 Fermi。
history.csv 的建议/Fermi/策略变化计数均是本轮计数；round=0 为初始状态。
每次运行创建独立输出目录，不覆盖已有实验。更换 --seed 可进行独立重复实验。
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    # 同时支持在 IDE 直接运行本文件和 python -m 包运行。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from SPD.llm_advisor import request_advice
    from SPD.simulation import SimulationConfig, SpatialPrisonersDilemma
else:
    from .llm_advisor import request_advice
    from .simulation import SimulationConfig, SpatialPrisonersDilemma


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=50, help="方格边长 L（默认 50，至少 3）")
    parser.add_argument("--steps", type=int, default=1000, help="时间步数（默认 1000）")
    parser.add_argument("--b", type=float, default=1.5, help="背叛诱惑收益 T=b，要求 b>1")
    parser.add_argument("--k", type=float, default=0.1, help="Fermi 噪声 K，要求 K>0")
    parser.add_argument("--lambda", dest="lambda_prob", type=float, default=0.1,
                        help="每次更新时听取建议的概率 λ（默认 0.1）")
    parser.add_argument("--initial-cooperation", type=float, default=0.5,
                        help="各格点独立初始化为 C 的概率，不保证比例恰好相等")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--boundary", choices=("periodic", "open"), default="periodic",
                        help="periodic 周期边界；open 开放边界，边缘邻居较少（不归一化收益）")
    parser.add_argument("--update-mode", choices=("asynchronous", "synchronous"), default="asynchronous",
                        help="默认随机异步：每步 L² 次有放回抽样；同步则每步全员更新一次")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results",
                        help="结果根目录；其下自动创建本次运行的独立子目录")
    parser.add_argument("--report-every", type=int, default=100, help="打印进度的步数间隔；0 表示仅打印结束结果")
    return parser


def write_grid(path, grid):
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(grid)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.steps < 0 or args.report_every < 0:
        parser.error("steps 和 report-every 必须为非负整数。")
    try:
        config = SimulationConfig(**{
            key: getattr(args, key) for key in (
                "size", "b", "k", "lambda_prob", "initial_cooperation", "seed", "boundary", "update_mode",
            )
        })
    except ValueError as exc:
        parser.error(str(exc))

    simulation = SpatialPrisonersDilemma(config, advisor=request_advice)
    output = args.output_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{args.seed}")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "config": asdict(config),
        "steps": args.steps,
        "payoff_matrix": {"R": 1, "S": 0, "T": config.b, "P": 0},
        "advice_rule": "probability_lambda_then_advice_else_fermi; None_falls_back_to_fermi",
        "advisor": f"{request_advice.__module__}.{request_advice.__qualname__}",
        "round_counters": "per_round; round_0_is_initial_state",
        "status": "running",
        "completed_steps": 0,
    }
    config_path = output / "config.json"
    config_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_grid(output / "initial_grid.csv", simulation.grid)
    print(f"输出目录：{output}", flush=True)
    print("λ 分支：返回 C/D 时直接采纳，返回 None 时回退 Fermi；当前预留接口返回 None。", flush=True)

    initial = simulation.measure()
    final = initial
    try:
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(asdict(initial)))
            writer.writeheader()
            writer.writerow(asdict(initial))
            for _ in range(args.steps):
                final = simulation.step()
                writer.writerow(asdict(final))
                metadata["completed_steps"] = final.round
                if args.report_every and final.round % args.report_every == 0:
                    print(f"步数 {final.round}/{args.steps} | 合作率 {final.cooperation_rate:.4f} "
                          f"| 平均收益 {final.average_payoff:.4f}", flush=True)
                    file.flush()
        write_grid(output / "final_grid.csv", simulation.grid)
        metadata["status"] = "completed"
    except BaseException:
        # 中断或接口错误留下的部分轨迹不能被误认为完整实验。
        metadata["status"] = "failed_or_interrupted"
        raise
    finally:
        config_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"实验完成：{final.round} 步，最终合作率 {final.cooperation_rate:.4f}，"
          f"平均收益 {final.average_payoff:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
