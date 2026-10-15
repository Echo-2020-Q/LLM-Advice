"""独立重复咨询 GPT-5.5：10 种局面，每种重复 n 次，共 10*n 次请求。

运行：python main/GPT_Multistepsa.py --b 1.5 --n 10
仅预览，不读取密钥、不发送请求：python main/GPT_Multistepsa.py --b 1.5 --n 10 --dry-run
直接在 IDE 中运行时，使用下方 B、N 的默认值。
结果保存在 logs/GPT_Multistepsa/ 的独立批次目录中。
"""

import argparse
import csv
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

if __package__:
    from .GPT_test import APIError as ConfigError, DEFAULT_ENV_FILE, load_config
    from .GPT_onestep_test import LOG_TIMEZONE, save_record
else:
    from GPT_test import APIError as ConfigError, DEFAULT_ENV_FILE, load_config
    from GPT_onestep_test import LOG_TIMEZONE, save_record


B = 1.5
N = 10  # 每种组合重复的次数；N=10 时共有 100 次请求。
MODEL = "gpt-5.5"
MAX_OUTPUT_TOKENS = 1024
TIMEOUT_SECONDS = 120
LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "GPT_Multistepsa"

SYSTEM_PROMPT = """You are an independent decision advisor for an individual participating in a repeated spatial Prisoner's Dilemma game. The game will continue for many rounds.

The game follows these rules:

1. Each player occupies a fixed position in a square lattice and interacts with four neighbors.
2. The game is repeated over many rounds, and players interact with the same neighbors throughout the game.
3. In each round, every player chooses one of two actions: Cooperate (C) or Defect (D).
4. The payoff from each pairwise interaction is:
   - C vs. C: both receive 1.
   - C vs. D: the cooperator receives 0 and the defector receives b.
   - D vs. D: both receive 0.
5. Each player's total payoff is the sum of the payoffs from the four interactions.
6. Players may revise their actions between rounds. Future actions of their neighbors are unknown.

Your objective is to recommend an action that maximizes the focal player's expected cumulative payoff over the entire repeated game

You are an advisor, not a participant in the game.

Recommend either C or D. No explanation is required."""

CSV_FIELDS = [
    "request_index", "repetition", "combination_index", "b", "current_action",
    "cooperating_neighbors", "defecting_neighbors", "status", "recommendation",
    "output_text", "duration_seconds", "response_id", "input_tokens",
    "output_tokens", "total_tokens", "error", "record_file",
]


def build_cases(b):
    return [
        {"b": b, "current_action": action,
         "cooperating_neighbors": cooperating, "defecting_neighbors": 4 - cooperating}
        for action in ("C", "D") for cooperating in range(5)
    ]


def build_user_prompt(case):
    return f"""b = {case['b']}

Current observation of the focal player:
- Current action: {case['current_action']}
- Number of cooperating neighbors: {case['cooperating_neighbors']}
- Number of defecting neighbors: {case['defecting_neighbors']}

What action would you recommend for this player in the next round?

Return only one action: C or D."""


def make_request(case, max_tokens):
    # 不传 previous_response_id、conversation 或以前的输入/输出。
    return {"model": MODEL, "instructions": SYSTEM_PROMPT,
            "input": build_user_prompt(case), "max_output_tokens": max_tokens, "store": False}


def now():
    return datetime.now(LOG_TIMEZONE).isoformat(timespec="milliseconds")


def call_once(openai, api_key, base_url, record, timeout):
    """每条新建客户端，仅调用一次；错误与无效输出均不自动重试。"""
    timer = time.perf_counter()
    record["started_at"] = now()
    try:
        with openai.OpenAI(api_key=api_key, base_url=base_url,
                           timeout=timeout, max_retries=0) as client:
            record["api_call_started"] = True
            response = client.responses.create(**record["request"])
            record["output_text"] = getattr(response, "output_text", None)
            dump = getattr(response, "model_dump", None)
            record["response"] = dump(mode="json") if callable(dump) else {
                "id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
                "status": getattr(response, "status", None),
                "output_text": record["output_text"],
            }
            text = record["output_text"]
            if getattr(response, "status", None) != "completed":
                record["error"] = {"type": "incomplete_response", "message": "响应未完成，未计为有效建议。"}
            elif not isinstance(text, str) or text.strip() not in {"C", "D"}:
                record["error"] = {"type": "invalid_action", "message": "模型没有仅返回 C 或 D，原文已保存。"}
            else:
                record["action"] = text.strip()
                record["status"] = "success"
    except KeyboardInterrupt:
        record["status"] = "cancelled"
        record["error"] = {"type": "KeyboardInterrupt", "message": "用户中断本次调用。"}
    except Exception as exc:
        # 不输出 SDK 异常原文，避免平台回显密钥。
        status_code = getattr(exc, "status_code", None)
        if isinstance(exc, openai.APITimeoutError):
            message = "请求超时。"
        elif isinstance(exc, openai.APIConnectionError):
            message = "连接失败，请检查网络与代理。"
        elif status_code is not None:
            message = f"HTTP {status_code}，请检查密钥、分组、余额和接口配置。"
        else:
            message = "调用或响应解析失败，请检查配置及接口兼容性。"
        record["error"] = {"type": type(exc).__name__, "http_status": status_code, "message": message}
    finally:
        if record["status"] == "running":
            record["status"] = "error"
        record["finished_at"] = now()
        record["duration_seconds"] = round(time.perf_counter() - timer, 3)


def csv_row(record, filename):
    response = record["response"] if isinstance(record["response"], dict) else {}
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    return {
        "request_index": record["request_index"], "repetition": record["repetition"],
        "combination_index": record["combination_index"], **record["observation"],
        "status": record["status"], "recommendation": record["action"] or "",
        "output_text": record["output_text"], "duration_seconds": record["duration_seconds"],
        "response_id": response.get("id", ""), "input_tokens": usage.get("input_tokens", ""),
        "output_tokens": usage.get("output_tokens", ""), "total_tokens": usage.get("total_tokens", ""),
        "error": (record["error"] or {}).get("message", ""), "record_file": filename,
    }


def account_record(summary, record):
    """按顺序计数一次，中断后的补写不会重复计数。"""
    if record["request_index"] <= summary["processed_cases"]:
        return
    summary["processed_cases"] += 1
    summary["api_calls_started"] += int(record["api_call_started"])
    summary["not_run_requests"] = summary["planned_requests"] - summary["processed_cases"]
    counts = summary["cases"][record["combination_index"] - 1]
    if record["status"] == "success":
        summary["successful_requests"] += 1
        counts[record["action"]] += 1
    elif record["status"] == "cancelled":
        summary["cancelled_requests"] += 1
        counts["cancelled"] += 1
    else:
        summary["failed_requests"] += 1
        counts["errors"] += 1


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("必须是有限数值，不能为 NaN 或无穷大")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--b", type=finite_float, default=B, help="博弈参数，默认读取文件顶部 B")
    parser.add_argument("--n", type=positive_int, default=N, help="每种组合的重复次数，默认读取文件顶部 N")
    parser.add_argument("--max-tokens", type=positive_int, default=MAX_OUTPUT_TOKENS, help="输出及推理 token 上限")
    parser.add_argument("--timeout", type=finite_float, default=TIMEOUT_SECONDS, help="每条请求超时秒数")
    parser.add_argument("--continue-on-error", action="store_true", help="某条失败后继续后续计划；不重试失败条目")
    parser.add_argument("--dry-run", action="store_true", help="只预览组合，不读取密钥、不导入 SDK、不发请求或写日志")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.b) or not isinstance(args.n, int) or isinstance(args.n, bool)
            or args.n < 1 or args.timeout <= 0 or not math.isfinite(args.timeout)):
        parser.error("b 必须是有限数值，n 必须是正整数，timeout 必须为有限正数")
    if not isinstance(args.max_tokens, int) or isinstance(args.max_tokens, bool) or args.max_tokens < 16:
        parser.error("--max-tokens 至少为 16，建议保留 1024 以容纳推理 token")

    cases = build_cases(args.b)
    total = len(cases) * args.n
    print(f"模型={MODEL} | b={args.b} | 10 种组合 × {args.n} 次 = {total} 次独立请求", flush=True)
    if args.dry_run:
        for index, case in enumerate(cases, 1):
            print(f"{index:2}. 当前={case['current_action']} 合作邻居={case['cooperating_neighbors']} "
                  f"背叛邻居={case['defecting_neighbors']}，重复 {args.n} 次", flush=True)
        print("预览完成：未读取密钥、未调用 API、未写入实验日志。", flush=True)
        return 0

    try:
        import openai
        api_key, base_url = load_config(DEFAULT_ENV_FILE)
    except ImportError:
        print("缺少依赖，请运行：python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    except (ConfigError, OSError, UnicodeError, ValueError):
        print("无法加载配置，请检查 API/gpt.env 中的密钥、地址和文件编码。", file=sys.stderr)
        return 1

    started = datetime.now(LOG_TIMEZONE)
    batch_id = uuid4().hex
    directory = LOG_DIR / f"{started:%Y%m%d_%H%M%S_%f}_{batch_id}"
    summary = {
        "batch_id": batch_id, "started_at": started.isoformat(timespec="milliseconds"),
        "finished_at": None, "status": "running", "b": args.b, "n": args.n,
        "planned_requests": total, "api_calls_started": 0, "processed_cases": 0,
        "successful_requests": 0, "failed_requests": 0, "cancelled_requests": 0,
        "not_run_requests": total,
        "cases": [{**case, "C": 0, "D": 0, "errors": 0, "cancelled": 0} for case in cases],
    }
    exit_code = 0
    record = None
    filename = None
    try:
        (directory / "records").mkdir(parents=True, exist_ok=False)
        save_record(directory / "config.json", {
            "batch_id": batch_id, "model": MODEL, "base_url": base_url, "b": args.b, "n": args.n,
            "planned_requests": total, "system_prompt": SYSTEM_PROMPT, "cases": cases,
            "max_output_tokens": args.max_tokens, "timeout_seconds": args.timeout,
            "max_retries": 0, "store": False, "continue_on_error": args.continue_on_error,
            "order": "每轮依次 C/0..4、D/0..4；共 n 轮；每次只提供当前局面，无历史。",
        }, api_key)
        save_record(directory / "summary.json", summary, api_key)
        print(f"记录目录：{directory}", flush=True)

        # UTF-8 BOM 便于 Excel 直接打开；每条完成后立即刷新到文件。
        with (directory / "results.csv").open("x", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            output.flush()
            for index in range(total):
                case_index = index % len(cases)
                repetition = index // len(cases) + 1
                case = cases[case_index]
                filename = f"records/{index + 1:06d}.json"
                record = {
                    "batch_id": batch_id, "request_index": index + 1,
                    "repetition": repetition, "combination_index": case_index + 1,
                    "observation": case, "base_url": base_url,
                    "client_options": {"timeout_seconds": args.timeout, "max_retries": 0},
                    "request": make_request(case, args.max_tokens), "api_call_started": False,
                    "started_at": now(), "finished_at": None, "duration_seconds": None,
                    "status": "running", "action": None, "output_text": None,
                    "response": None, "error": None,
                }
                save_record(directory / filename, record, api_key)
                print(f"[{index + 1}/{total}] 第{repetition}轮 b={args.b} "
                      f"当前={case['current_action']} 邻居C={case['cooperating_neighbors']} "
                      f"D={case['defecting_neighbors']} -> ", end="", flush=True)
                call_once(openai, api_key, base_url, record, args.timeout)

                account_record(summary, record)

                save_record(directory / filename, record, api_key)
                row = csv_row(record, filename)
                # CSV 同样脱敏；无效模型返回也保留，不能默认为某个动作。
                for key, value in row.items():
                    if isinstance(value, str):
                        row[key] = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", value.replace(api_key, "[REDACTED]"))
                writer.writerow(row)
                output.flush()
                save_record(directory / "summary.json", summary, api_key)
                result = record["action"] if record["status"] == "success" else record["error"]["message"]
                print(f"{result} ({record['duration_seconds']:.3f}s)", flush=True)

                if record["status"] == "cancelled":
                    summary["status"], exit_code = "cancelled", 130
                    break
                if record["status"] == "error" and not args.continue_on_error:
                    summary["status"], exit_code = "stopped_on_error", 1
                    print("已停止。要在失败后继续后续组合，可使用 --continue-on-error。", flush=True)
                    break
            else:
                summary["status"] = "completed_with_errors" if summary["failed_requests"] else "completed"
                exit_code = 1 if summary["failed_requests"] else 0
    except KeyboardInterrupt:
        summary["status"], exit_code = "cancelled", 130
        if record is not None:
            if record["status"] == "running":
                record["status"] = "cancelled"
                record["error"] = {"type": "KeyboardInterrupt", "message": "用户中断本次调用。"}
            if record["finished_at"] is None:
                record["finished_at"] = now()
                record["duration_seconds"] = round(
                    (datetime.fromisoformat(record["finished_at"]) - datetime.fromisoformat(record["started_at"])).total_seconds(), 3
                )
            account_record(summary, record)
            try:
                save_record(directory / filename, record, api_key)
            except OSError:
                summary["status"], exit_code = "logging_error", 1
                print("中断后无法补写当前记录，请检查磁盘空间和目录权限。", file=sys.stderr, flush=True)
        print("\n已中断，已保存的记录仍在本地。", file=sys.stderr, flush=True)
    except OSError:
        summary["status"], exit_code = "logging_error", 1
        print("\n日志写入失败，已停止后续请求。请检查磁盘空间和目录权限。", file=sys.stderr, flush=True)
    finally:
        summary["finished_at"] = now()
        try:
            if directory.is_dir():
                save_record(directory / "summary.json", summary, api_key)
        except OSError:
            summary["status"] = "logging_error"
            exit_code = 1
            print("无法更新批次汇总；请检查 records/ 中已保存的逐条记录。", file=sys.stderr, flush=True)

    print(f"批次状态={summary['status']} | 成功={summary['successful_requests']} "
          f"失败={summary['failed_requests']} 取消={summary['cancelled_requests']} "
          f"未执行={summary['not_run_requests']}", flush=True)
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        sys.exit(130)
