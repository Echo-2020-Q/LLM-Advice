"""独立重复咨询 GPT-5.5：全部 512 种 3×3 C/D 网格，每种重复 n 次。

中心是第 2 行第 2 列，上下左右为直接邻居，四角提供空间信息。
每次请求只包含系统指令和当前网格，没有历史；不合并旋转或镜像图案。
B=None 时不在用户提示词中添加 b 的说明；请自行在 SYSTEM_PROMPT 中填写。
运行：python main/GPT_Multisteps.py --n 10
仅预览：python main/GPT_Multisteps.py --n 10 --dry-run
如需明确给出某个数值，仍可指定 --b 1.5。
旧式十组合批次仍可通过 --resume 续跑，沿用保存的原始提示词。
断点续跑：python main/GPT_Multisteps.py --resume 批次目录或批次名
TLS/SSL 连接异常关闭时默认额外重试 5 次，每次间隔 5 秒；--tls-retries 0 可关闭。
使用 --tls-retries 次数、--retry-delay 秒数调整；其他错误仍按原有方式处理。
续跑预览：在上述命令后加 --dry-run，不读取密钥、不发送请求、不改日志。
直接在 IDE 中运行时，使用下方 B、N 的默认值。
结果保存在 logs/GPT_Multistepsa/ 的独立批次目录中。
"""

import argparse
import copy
import csv
import json
import math
import re
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from uuid import uuid4

if __package__:
    from .GPT_test import APIError as ConfigError, DEFAULT_ENV_FILE, load_config
    from .GPT_onestep_test import LOG_TIMEZONE, exception_diagnostics, redact, runtime_info, save_record
else:
    from GPT_test import APIError as ConfigError, DEFAULT_ENV_FILE, load_config
    from GPT_onestep_test import LOG_TIMEZONE, exception_diagnostics, redact, runtime_info, save_record


B = None  # None：b 的说明由 SYSTEM_PROMPT 提供；数值：在用户提示词中明确告知该值。
N = 2  # 每个网格重复的次数；N=10 时共有 5120 次独立请求。
OBSERVATION_FORMAT = "grid_3x3"  # 新批次使用完整 512 种网格；neighbor_counts 保留用于旧格式兼容。
MODEL = "gpt-5.5"
MAX_OUTPUT_TOKENS = 10240  # 内部推理和最终回答共用此上限。
TIMEOUT_SECONDS = 180
TLS_RETRIES = 5  # 每条请求连接异常关闭后的额外重试次数；0 表示关闭。
RETRY_DELAY_SECONDS = 5  # 重试前等待秒数，可按 Ctrl+C 中断。
# True：沿用环境中的代理/证书设置；False：直连并使用默认验证证书。
TRUST_ENV = True
LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "GPT_Multistepsa"

# 从单步脚本复制的系统提示词，可在这里单独编辑。
SYSTEM_PROMPT = """You are a decision advisor for me, who is participating in a repeated spatial Prisoner's Dilemma game. The game will continue for many rounds.

The game follows these rules:

1. Each player occupies a fixed position in a square lattice and interacts with four orthogonal neighbors: up, down, left, and right. Diagonal players are not direct interaction partners.
2. The game is repeated over many rounds, and players interact with the same neighbors throughout the game.
3. In each round, every player chooses one of two actions: Cooperate (C) or Defect (D).
4. The payoff from each pairwise interaction is:
   - C vs. C: both receive 1.
   - C vs. D: the cooperator receives 0 and the defector receives b (1 < b < 2).
   - D vs. D: both receive 0.
5. Each player's total payoff is the sum of the payoffs from the four interactions.
6. Players revise their actions synchronously between rounds: all players make update decisions using the same pre-update actions and payoffs, and the new actions are applied simultaneously. Future actions of their neighbors are unknown.
7. At initialization, a fixed subset of players is selected to be eligible for LLM advice. Eligibility does not change when a player copies another player's action. At each update, an eligible player follows LLM advice with probability lambda. All other updates use Fermi imitation; if advice is unavailable, the player also uses Fermi imitation.

Observation format:
- Each observation is a 3x3 local window within a larger lattice, not the entire lattice. The edges of this window are not game boundaries and do not wrap around within the window.
- Rows are shown from top to bottom (north to south), and columns from left to right (west to east).
- I am the focal player at row 2, column 2, using 1-based row and column indices.
- C means Cooperate and D means Defect. Every cell shows that player's current action.
- My four direct neighbors are at (1, 2), (2, 1), (2, 3), and (3, 2). The four corner cells provide additional spatial context but do not directly contribute to my payoff.
- States outside the window are unobserved. The window does not give the full neighborhoods of the surrounding players, so their complete payoffs and future actions cannot be determined from the window alone.
- Unless explicitly provided, the numerical value of K, the fraction of advice-eligible players, lambda, and each player's eligibility are unknown. Do not assume values for them or for unobserved states.
- Each new grid is a supplied observation. Do not assume that my previous recommendation was followed or that the new grid was generated from it.

Your objective is to recommend an action that maximizes my expected cumulative payoff over the entire spatial repeated game.

Recommend either C or D. No explanation is required.
"""

CSV_FIELDS = [
    "request_index", "repetition", "combination_index", "b", "b_mode", "b_constraint", "current_action",
    "cooperating_neighbors", "defecting_neighbors", "status", "recommendation",
    "output_text", "duration_seconds", "response_id", "input_tokens",
    "output_tokens", "total_tokens", "error", "record_file",
    "observation_format", "grid_key", "grid_rows",
]


def valid_b(b):
    return b is None or (type(b) in (int, float) and math.isfinite(b))


def b_metadata(b):
    """未知 b 不能记为 1.5；日志显式区分未知区间与确定数值。"""
    return {"b_mode": "unknown_interval" if b is None else "fixed",
            "b_constraint": "1 < b < 2" if b is None else None}


def describe_b(b):
    return "1 < b < 2（具体值未知）" if b is None else f"b={b}"


def build_cases(b, observation_format=None):
    """按固定顺序枚举状态；网格不合并旋转或镜像。"""
    observation_format = OBSERVATION_FORMAT if observation_format is None else observation_format
    if observation_format == "neighbor_counts":
        return [
            {"b": b, "current_action": action,
             "cooperating_neighbors": cooperating, "defecting_neighbors": 4 - cooperating}
            for action in ("C", "D") for cooperating in range(5)
        ]
    if observation_format != "grid_3x3":
        raise ValueError("OBSERVATION_FORMAT 必须为 grid_3x3 或 neighbor_counts。")
    cases = []
    for cells in product("CD", repeat=9):
        key = "".join(cells)
        cooperating = sum(cells[i] == "C" for i in (1, 3, 5, 7))
        cases.append({
            "b": b, "current_action": cells[4],
            "cooperating_neighbors": cooperating, "defecting_neighbors": 4 - cooperating,
            "grid_key": key, "grid_rows": [key[:3], key[3:6], key[6:]],
        })
    return cases


def build_user_prompt(case):
    parameter = "" if case["b"] is None else f"b = {case['b']}\n\n"
    if "grid_rows" in case:
        grid = "\n".join(" ".join(row) for row in case["grid_rows"])
        cooperating = case["cooperating_neighbors"]
        if case["current_action"] == "C" or cooperating == 0:
            payoff = str(cooperating)
        elif case["b"] is None:
            payoff = f"{cooperating}*b"
        else:
            payoff = format(cooperating * case["b"], ".15g")
        return f"""{parameter}Game state:

Current local 3x3 grid (C = Cooperate, D = Defect):
{grid}

I am the center player at row 2, column 2.

My current payoff is {payoff}.

What action would you recommend for the center player in the next round to maximizes my expected cumulative payoff over the entire spatial repeated game.?

Return only one action: C or D."""
    # 旧批次依赖此原文；不改写旧的十组合用户提示词。
    return f"""{parameter}Current observation of the focal player:
- Current action: {case['current_action']}
- Number of cooperating neighbors: {case['cooperating_neighbors']}
- Number of defecting neighbors: {case['defecting_neighbors']}

What action would you recommend for this player in the next round?

Return only one action: C or D."""


def make_request(case, max_tokens):
    # 每次只发送当前状态，没有 previous_response_id、conversation 或以前的输入/输出。
    if "grid_rows" in case:
        return {"model": MODEL,
                "input": [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": build_user_prompt(case)}],
                "max_output_tokens": max_tokens, "store": False}
    return {"model": MODEL, "instructions": SYSTEM_PROMPT,
            "input": build_user_prompt(case), "max_output_tokens": max_tokens, "store": False}


def describe_case(case):
    text = (f"当前={case['current_action']} 邻居C={case['cooperating_neighbors']} "
            f"D={case['defecting_neighbors']}")
    if "grid_rows" in case:
        text += " 网格=" + "/".join(case["grid_rows"])
    return text


def now():
    return datetime.now(LOG_TIMEZONE).isoformat(timespec="milliseconds")


def is_tls_connection_closed(diagnostics):
    """仅识别连接异常关闭；证书配置错误不能靠重复请求解决。"""
    causes = [diagnostics, *diagnostics.get("causes", [])]
    detail = " ".join(str(cause.get(key, "")) for cause in causes
                      for key in ("exception_type", "detail")).lower()
    if any(marker in detail for marker in (
            "certificate", "cert_verify", "certverification", "hostname mismatch")):
        return False
    if any(marker in detail for marker in (
            "connectionreseterror", "connection reset", "server disconnected without sending a response",
            "remote end closed connection without response")):
        return True
    return (any(marker in detail for marker in ("tls", "ssl"))
            and any(marker in detail for marker in (
                "eof", "zeroreturn", "connection has been closed", "connection was closed",
                "connection closed", "connection reset", "closed unexpectedly",
                "closed abruptly", "unexpected end")))


def call_once(openai, api_key, base_url, record, timeout):
    """只发送一次，关闭 SDK 内置重试；重试与归档由外层处理。"""
    timer = time.perf_counter()
    record["started_at"] = now()
    http_client = None
    try:
        trust_env = record.get("client_options", {}).get("trust_env", TRUST_ENV)
        http_client = openai.DefaultHttpxClient(trust_env=trust_env, verify=True)
        with openai.OpenAI(api_key=api_key, base_url=base_url,
                           timeout=timeout, max_retries=0, http_client=http_client) as client:
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
                details = getattr(response, "incomplete_details", None)
                if details is None and isinstance(record["response"], dict):
                    details = record["response"].get("incomplete_details")
                reason = details.get("reason") if isinstance(details, dict) else getattr(details, "reason", None)
                if reason == "max_output_tokens":
                    message = (f"模型达到 max_output_tokens={record['request']['max_output_tokens']} "
                               "的生成上限（包含内部推理 token），本轮未得到有效建议。")
                elif reason == "content_filter":
                    message = "模型输出被服务端中断（content_filter），未计为有效建议。"
                else:
                    message = "响应未完成，未计为有效建议；详见记录 response.incomplete_details。"
                record["error"] = {"type": "incomplete_response", "reason": reason, "message": message}
            elif not isinstance(text, str) or text.strip() not in {"C", "D"}:
                record["error"] = {"type": "invalid_action", "message": "模型没有仅返回 C 或 D，原文已保存。"}
            else:
                record["action"] = text.strip()
                record["status"] = "success"
    except KeyboardInterrupt:
        record["status"] = "cancelled"
        record["error"] = {"type": "KeyboardInterrupt", "message": "用户中断本次调用。"}
    except Exception as exc:
        diagnostics = exception_diagnostics(exc, api_key)
        status_code = getattr(exc, "status_code", None)
        retryable_connection_closed = False
        if isinstance(exc, openai.APITimeoutError):
            message = "请求超时；底层原因详见记录 error.diagnostics。"
        elif isinstance(exc, openai.APIConnectionError):
            retryable_connection_closed = status_code is None and is_tls_connection_closed(diagnostics)
            if retryable_connection_closed:
                message = "TLS/SSL 连接异常关闭，请检查网络、代理和 API 地址；底层原因详见记录 error.diagnostics。"
            else:
                message = "连接失败，请检查网络、代理和 API 地址；底层原因详见记录 error.diagnostics。"
        elif status_code is not None:
            message = f"HTTP {status_code}，请检查密钥、分组、余额和接口配置；服务端详情已保存在 error.diagnostics。"
        else:
            message = "调用或响应解析失败，请检查配置及接口兼容性；详情见 error.diagnostics。"
        code = getattr(exc, "code", None)
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            nested = body.get("error")
            code = code or (nested.get("code") if isinstance(nested, dict) else body.get("code"))
        if code:
            diagnostics["api_error_code"] = redact(code, api_key)
        request_id = getattr(exc, "request_id", None)
        if request_id:
            diagnostics["request_id"] = redact(request_id, api_key)
        record["error"] = {"type": type(exc).__name__, "http_status": status_code,
                           "message": message, "diagnostics": diagnostics,
                           "retryable_connection_closed": retryable_connection_closed}
    finally:
        # OpenAI 正常退出时也会关闭该客户端；close 幂等，同时覆盖构造失败的情况。
        if http_client is not None:
            try:
                http_client.close()
            except Exception as exc:
                # 已保存的有效建议不因清理连接失败而失效，避免续跑时重复付费。
                record["client_close_error"] = exception_diagnostics(exc, api_key)
        if record["status"] == "running":
            record["status"] = "error"
        record["finished_at"] = now()
        record["duration_seconds"] = round(time.perf_counter() - timer, 3)


def call_with_retries(openai, api_key, base_url, record, timeout, *, directory, archived,
                      retries, delay):
    """同一条请求有限重试，每次失败在再次发送之前独立归档。"""
    path = directory / "records" / f"{record['request_index']:06d}.json"
    record["client_options"].update({"tls_retries": retries, "retry_delay_seconds": delay})
    for retry in range(retries + 1):
        call_once(openai, api_key, base_url, record, timeout)
        if (record["status"] != "error"
                or not (record.get("error") or {}).get("retryable_connection_closed")):
            return
        if retry == retries:
            if retries:
                print(f"\n自动重试已达上限（额外 {retries} 次）。", flush=True)
            return

        # 等待期间仍保留完整失败记录，Ctrl+C 后可以继续使用 --resume。
        save_record(path, record, api_key)
        print(f"\nTLS/SSL 连接异常关闭；{delay:g} 秒后自动重试当前条目 "
              f"[{record['request_index']}]（{retry + 1}/{retries}）...", flush=True)
        time.sleep(delay)

        previous = copy.deepcopy(record)
        candidate = copy.deepcopy(record)
        candidate.update({"api_call_started": False, "started_at": now(), "finished_at": None,
                          "duration_seconds": None, "status": "running", "action": None,
                          "output_text": None, "response": None, "error": None})
        candidate.pop("client_close_error", None)
        archive_dir = directory / "attempts" / f"{record['request_index']:06d}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{uuid4().hex}.json"
        archive_count = len(archived)
        try:
            path.replace(archive_path)
            archived.append(previous)
            record.clear()
            record.update(candidate)
        except BaseException:
            # 中断或移动失败时恢复原状态，避免同一尝试被重复统计。
            if archive_path.exists():
                try:
                    archive_path.replace(path)
                except OSError:
                    # 无法回滚时保留已落盘的归档，当前条目标记为尚未发送。
                    del archived[archive_count:]
                    archived.append(previous)
                    record.clear()
                    record.update(candidate)
                    raise
            record.clear()
            record.update(previous)
            del archived[archive_count:]
            raise
        # 日志不可写时停止，不能带着丢失的记录继续发送付费请求。
        save_record(path, record, api_key)


def csv_row(record, filename):
    response = record["response"] if isinstance(record["response"], dict) else {}
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    case = record["observation"]
    return {
        "request_index": record["request_index"], "repetition": record["repetition"],
        "combination_index": record["combination_index"],
        "b": "unknown" if case["b"] is None else case["b"], **b_metadata(case["b"]),
        "current_action": case["current_action"], "cooperating_neighbors": case["cooperating_neighbors"],
        "defecting_neighbors": case["defecting_neighbors"],
        "status": record["status"], "recommendation": record["action"] or "",
        "output_text": record["output_text"], "duration_seconds": record["duration_seconds"],
        "response_id": response.get("id", ""), "input_tokens": usage.get("input_tokens", ""),
        "output_tokens": usage.get("output_tokens", ""), "total_tokens": usage.get("total_tokens", ""),
        "error": (record["error"] or {}).get("message", ""), "record_file": filename,
        "observation_format": "grid_3x3" if "grid_rows" in case else "neighbor_counts",
        "grid_key": case.get("grid_key", ""), "grid_rows": "/".join(case.get("grid_rows", [])),
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


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须为非负整数")
    return number


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("必须是有限数值，不能为 NaN 或无穷大")
    return number


class ResumeError(ValueError):
    """无法安全恢复批次时停止，不把损坏记录当成尚未执行。"""


def read_batch(directory):
    """只读原始 JSON，恢复实验配置和每种局面的完整请求。"""
    def read_json(path):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ResumeError(f"无法读取 {path.name}，请检查原始日志。") from None
        if not isinstance(value, dict):
            raise ResumeError(f"{path.name} 必须是 JSON 对象。")
        return value

    config = read_json(directory / "config.json")
    try:
        observation_format = config.get("observation_format", "neighbor_counts")
        expected_cases = build_cases(config["b"], observation_format)
        case_count = len(expected_cases)
        if (type(config["n"]) is not int or config["n"] < 1
                or not valid_b(config["b"])
                or any(config.get(key) != value for key, value in b_metadata(config["b"]).items()
                       if config["b"] is None or key in config)
                or type(config["planned_requests"]) is not int
                or config["planned_requests"] != case_count * config["n"]
                or config["cases"] != expected_cases
                or ("case_count" in config and (type(config["case_count"]) is not int
                                               or config["case_count"] != case_count))
                or ("trust_env" in config and type(config["trust_env"]) is not bool)
                or type(config["max_output_tokens"]) is not int or config["max_output_tokens"] < 16
                or type(config["timeout_seconds"]) not in (int, float)
                or not math.isfinite(config["timeout_seconds"]) or config["timeout_seconds"] <= 0
                or any(not isinstance(config[k], str) or not config[k]
                       for k in ("batch_id", "model", "system_prompt", "base_url"))):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ResumeError("config.json 配置不完整或不合法，已停止续跑。") from None
    if not (directory / "records").is_dir():
        raise ResumeError("缺少 records 目录，不能确定哪些请求已经成功。")

    # 网格批次必须有完整的原始请求，不使用今天的提示词重建历史实验。
    if observation_format == "grid_3x3" and "requests" not in config:
        raise ResumeError("网格批次缺少保存的 requests 模板，已停止续跑。")
    templates = config.get("requests", [None] * case_count)
    if not isinstance(templates, list) or len(templates) != case_count:
        raise ResumeError("config.json 中的请求模板不合法。")
    templates = list(templates)
    if observation_format == "grid_3x3" and any(template is None for template in templates):
        raise ResumeError("网格批次的请求模板不完整，已停止续跑。")

    def check_request(request):
        keys = {"model", "input", "max_output_tokens", "store"}
        if observation_format == "neighbor_counts":
            keys.add("instructions")
        if (not isinstance(request, dict) or set(request) != keys
                or request["model"] != config["model"]
                or request["max_output_tokens"] != config["max_output_tokens"]
                or request["store"] is not False):
            raise ResumeError("记录中的请求与批次配置不一致，或包含未知参数，已停止续跑。")
        if observation_format == "neighbor_counts":
            valid_input = (request["instructions"] == config["system_prompt"]
                           and isinstance(request["input"], str))
        else:
            messages = request["input"]
            valid_input = (isinstance(messages, list) and len(messages) == 2
                           and messages[0] == {"role": "system", "content": config["system_prompt"]}
                           and isinstance(messages[1], dict) and set(messages[1]) == {"role", "content"}
                           and messages[1]["role"] == "user" and isinstance(messages[1]["content"], str)
                           and bool(messages[1]["content"]))
        if not valid_input:
            raise ResumeError("记录中的系统提示词或独立请求结构与原批次不一致，已停止续跑。")

    records = {}
    for path in sorted((directory / "records").glob("*.json")):
        record = read_json(path)
        index = record.get("request_index")
        if type(index) is not int or not 1 <= index <= config["planned_requests"]:
            raise ResumeError(f"{path.name} 请求编号不合法。")
        case_index = (index - 1) % case_count
        required = {"output_text", "response", "error", "duration_seconds", "finished_at", "started_at", "action"}
        if (path.name != f"{index:06d}.json" or index in records
                or not required <= record.keys()
                or record.get("batch_id") != config["batch_id"]
                or record.get("combination_index") != case_index + 1
                or record.get("repetition") != (index - 1) // case_count + 1
                or record.get("observation") != config["cases"][case_index]
                or record.get("base_url") != config["base_url"]
                or type(record.get("api_call_started")) is not bool
                or record.get("status") not in {"success", "error", "cancelled", "running"}):
            raise ResumeError(f"{path.name} 与批次计划不匹配或内容不完整。")
        if record["status"] == "success" and (
                record["action"] not in ("C", "D") or not isinstance(record["output_text"], str)
                or record["output_text"].strip() != record["action"]
                or not isinstance(record["response"], dict)
                or record["response"].get("status") != "completed"):
            raise ResumeError(f"{path.name} 的成功标记与输出不一致。")
        request = record.get("request")
        check_request(request)
        if templates[case_index] is not None and templates[case_index] != request:
            raise ResumeError("同一组合存在不同请求，无法保持实验一致，已停止续跑。")
        templates[case_index] = request
        records[index] = record

    archived = [read_json(path) for path in sorted((directory / "attempts").glob("*/*.json"))]
    for previous in archived:
        index = previous.get("request_index")
        if (type(index) is not int or not 1 <= index <= config["planned_requests"]
                or previous.get("batch_id") != config["batch_id"]):
            raise ResumeError("历史尝试与原批次不匹配，已停止续跑。")
        case_index = (index - 1) % case_count
        request = previous.get("request")
        check_request(request)
        if templates[case_index] is not None and templates[case_index] != request:
            raise ResumeError("历史尝试与当前记录的请求模板不一致，已停止续跑。")
        templates[case_index] = request

    # 旧版没有 requests 字段：优先使用记录原文；缺失组合仅在旧模板与当前格式匹配时补齐。
    if any(request is None for request in templates):
        known = [(i, request) for i, request in enumerate(templates) if request is not None]
        if not known or any(request["input"] != build_user_prompt(config["cases"][i]) for i, request in known):
            raise ResumeError("旧批次缺少部分请求模板，且无法验证用户提示词格式；已停止续跑。")
        for i, request in enumerate(templates):
            if request is None:
                templates[i] = {"model": config["model"], "instructions": config["system_prompt"],
                                "input": build_user_prompt(config["cases"][i]),
                                "max_output_tokens": config["max_output_tokens"], "store": False}
    for request in templates:
        check_request(request)
    return config, records, templates, archived


def resumed_summary(config, records, archived, started_at, status):
    """结果按每个计划编号的最新记录计数；历史调用数单独保留。"""
    summary = {
        "batch_id": config["batch_id"], "b": config["b"], "n": config["n"], **b_metadata(config["b"]),
        "started_at": started_at, "finished_at": None, "status": status,
        "planned_requests": config["planned_requests"], "processed_cases": len(records),
        "observation_format": config.get("observation_format", "neighbor_counts"),
        "case_count": len(config["cases"]),
        "api_calls_started": sum(bool(r["api_call_started"]) for r in records.values()),
        "successful_requests": 0, "failed_requests": 0, "cancelled_requests": 0,
        "not_run_requests": config["planned_requests"] - len(records),
        "archived_attempts": len(archived),
        "cases": [{**case, "C": 0, "D": 0, "errors": 0, "cancelled": 0} for case in config["cases"]],
    }
    for record in records.values():
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
    summary["pending_requests"] = config["planned_requests"] - summary["successful_requests"]
    summary["api_calls_started_total"] = summary["api_calls_started"] + sum(
        bool(r.get("api_call_started")) for r in archived)
    return summary


def write_results(directory, records, api_key):
    """CSV 是可重建的视图，每个请求编号只有一行；先写临时文件再替换。"""
    target = directory / "results.csv"
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, record in sorted(records.items()):
            row = csv_row(record, f"records/{index:06d}.json")
            for key, value in row.items():
                if isinstance(value, str):
                    row[key] = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", value.replace(api_key, "[REDACTED]"))
            writer.writerow(row)
    temporary.replace(target)


def resume_batch(args):
    supplied = Path(args.resume).expanduser()
    directory = supplied if supplied.is_dir() else LOG_DIR / supplied
    directory = directory.resolve()
    lock_path = directory / ".run.lock"
    locked = False
    record = None
    summary = None
    api_key = ""
    exit_code = 0
    try:
        config, records, templates, archived = read_batch(directory)
        pending = [i for i in range(1, config["planned_requests"] + 1)
                   if i not in records or records[i]["status"] != "success"]
        print(f"续跑批次：{directory}\n模型={config['model']} | {describe_b(config['b'])} | n={config['n']} | "
              f"{len(config['cases'])} 种局面 | 已成功={config['planned_requests'] - len(pending)} | 待补跑={len(pending)}", flush=True)
        if args.dry_run:
            print("待补跑编号：" + (", ".join(map(str, pending)) or "无"), flush=True)
            print("续跑预览完成：未读取密钥、未调用 API、未修改日志。", flush=True)
            return 0
        if not pending:
            print("该批次已全部成功，无需再次调用 API。", flush=True)
            return 0
        try:
            with lock_path.open("x", encoding="utf-8") as lock:
                locked = True
                lock.write(now())
        except FileExistsError:
            raise ResumeError("批次存在 .run.lock。若进程仍在运行，请勿同时续跑；确认进程已结束后才能删除残留锁文件。") from None

        # 在拿到锁之后重新读取，防止等待期间另一进程已写入成功结果。
        config, records, templates, archived = read_batch(directory)
        pending = [i for i in range(1, config["planned_requests"] + 1)
                   if i not in records or records[i]["status"] != "success"]
        if not pending:
            print("该批次已全部成功，无需再次调用 API。", flush=True)
            return 0
        try:
            import openai
            api_key, configured_url = load_config(DEFAULT_ENV_FILE)
        except ImportError:
            raise ResumeError("缺少依赖，请运行：python -m pip install -r requirements.txt") from None
        except (ConfigError, OSError, UnicodeError, ValueError):
            raise ResumeError("无法加载 API/gpt.env，请检查配置。") from None
        if configured_url.rstrip("/") != config["base_url"].rstrip("/"):
            raise ResumeError("当前 API 地址与原批次不同。请先恢复原地址，避免使用错误平台的密钥续跑。")
        timeout = args.timeout if args.timeout is not None else config["timeout_seconds"]
        trust_env = config.get("trust_env", True) if args.trust_env is None else args.trust_env
        case_count = len(config["cases"])
        started_at = min((r["started_at"] for r in records.values()), default=now())
        summary = resumed_summary(config, records, archived, started_at, "running")
        # 在请求前检查 CSV/JSON 是否可写（包括 Windows 中 Excel 占用文件的情况）。
        write_results(directory, records, api_key)
        save_record(directory / "summary.json", summary, api_key)

        for index in pending:
            record = None
            path = directory / "records" / f"{index:06d}.json"
            case_index = (index - 1) % case_count
            case = config["cases"][case_index]
            candidate = {
                "batch_id": config["batch_id"], "request_index": index,
                "repetition": (index - 1) // case_count + 1, "combination_index": case_index + 1,
                "observation": case, "base_url": config["base_url"],
                **b_metadata(config["b"]),
                "client_options": {"timeout_seconds": timeout, "max_retries": 0,
                                   "trust_env": trust_env, "verify": True},
                "request": templates[case_index], "api_call_started": False,
                "started_at": now(), "finished_at": None, "duration_seconds": None,
                "status": "running", "action": None, "output_text": None, "response": None, "error": None,
            }
            if index in records:
                archive_dir = directory / "attempts" / f"{index:06d}"
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_path = archive_dir / f"{uuid4().hex}.json"
                # move 保留原始字节，且不会在中断后被重复归档计数。
                path.replace(archive_path)
                archived.append(records.pop(index))
            save_record(path, candidate, api_key)
            record = candidate
            records[index] = record
            print(f"[{index}/{config['planned_requests']}] 第{record['repetition']}轮 {describe_b(config['b'])} "
                  f"{describe_case(case)} -> ", end="", flush=True)
            call_with_retries(openai, api_key, config["base_url"], record, timeout,
                              directory=directory, archived=archived,
                              retries=args.tls_retries, delay=args.retry_delay)
            save_record(path, record, api_key)
            summary = resumed_summary(config, records, archived, started_at, "running")
            write_results(directory, records, api_key)
            save_record(directory / "summary.json", summary, api_key)
            result = record["action"] if record["status"] == "success" else record["error"]["message"]
            print(f"{result} ({record['duration_seconds']:.3f}s)", flush=True)
            if record["status"] == "cancelled":
                summary["status"], exit_code = "cancelled", 130
                break
            if record["status"] == "error" and not args.continue_on_error:
                summary["status"], exit_code = "stopped_on_error", 1
                print("已停止；下次使用相同的 --resume 命令可继续补跑。", flush=True)
                break
        else:
            summary["status"] = "completed_with_errors" if summary["pending_requests"] else "completed"
            exit_code = int(bool(summary["pending_requests"]))
    except KeyboardInterrupt:
        exit_code = 130
        if summary is not None:
            if record is not None:
                if record["status"] == "running":
                    record["status"] = "cancelled"
                    record["error"] = {"type": "KeyboardInterrupt", "message": "用户中断本次调用。"}
                record["finished_at"] = record["finished_at"] or now()
                try:
                    save_record(directory / "records" / f"{record['request_index']:06d}.json", record, api_key)
                except OSError:
                    exit_code = 1
            summary = resumed_summary(config, records, archived, started_at,
                                      "cancelled" if exit_code == 130 else "logging_error")
        print("\n已中断；可以使用同一 --resume 命令继续。", file=sys.stderr, flush=True)
    except (ResumeError, OSError) as exc:
        exit_code = 1
        if summary is not None:
            summary = resumed_summary(config, records, archived, started_at, "logging_error")
        message = str(exc) if isinstance(exc, ResumeError) else "日志读写失败，已停止请求；请检查磁盘权限，并关闭占用 results.csv 的程序。"
        print(message, file=sys.stderr, flush=True)
    finally:
        try:
            if summary is not None:
                status = summary["status"]
                if record is not None:
                    path = directory / "records" / f"{record['request_index']:06d}.json"
                    try:
                        # 保存失败后仅补写响应，不重发 API 请求。
                        save_record(path, record, api_key)
                    except OSError:
                        status, exit_code = "logging_error", 1
                        print("当前响应无法保存；下次续跑此项可能重复计费，请先修复日志目录写入问题。",
                              file=sys.stderr, flush=True)
                        try:
                            records[record["request_index"]] = json.loads(path.read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            records.pop(record["request_index"], None)
                summary = resumed_summary(config, records, archived, started_at, status)
                summary["finished_at"] = now()
                try:
                    write_results(directory, records, api_key)
                except OSError:
                    summary["status"], exit_code = "logging_error", 1
                    print("无法更新 CSV；已保存的逐条 JSON 可用于下次续跑。", file=sys.stderr, flush=True)
                save_record(directory / "summary.json", summary, api_key)
        except OSError:
            exit_code = 1
            if summary is not None:
                summary["status"] = "logging_error"
            print("无法更新汇总；逐条 JSON 仍可用于下次续跑。", file=sys.stderr, flush=True)
        finally:
            if locked:
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    exit_code = 1
                    print("无法移除 .run.lock；确认没有运行进程后，请手动移除该锁文件。", file=sys.stderr)
    if summary is not None:
        print(f"批次状态={summary['status']} | 成功={summary['successful_requests']} "
              f"失败={summary['failed_requests']} | 待补跑={summary['pending_requests']}", flush=True)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--b", type=finite_float, help="可选：在用户提示词中给出具体 b；B=None 时由 SYSTEM_PROMPT 说明")
    parser.add_argument("--n", type=positive_int, help="每种组合的重复次数，默认读取文件顶部 N")
    parser.add_argument("--trust-env", action=argparse.BooleanOptionalAction, default=None,
                        help="沿用代理/证书环境；--no-trust-env 直连。续跑默认沿用原批次设置")
    parser.add_argument("--max-tokens", type=positive_int, help="输出及推理 token 上限")
    parser.add_argument("--timeout", type=finite_float, help="每条请求超时秒数；续跑时默认沿用原配置")
    parser.add_argument("--tls-retries", type=nonnegative_int, default=TLS_RETRIES,
                        help="连接异常关闭时每条最多额外重试次数，默认 5；0 关闭，新批次和续跑均生效")
    parser.add_argument("--retry-delay", type=finite_float, default=RETRY_DELAY_SECONDS,
                        help="连接异常关闭后每次重试前等待的秒数，默认 5")
    parser.add_argument("--resume", help="原批次目录或批次名：跳过成功项，补跑失败、中断及未执行项")
    parser.add_argument("--continue-on-error", action="store_true", help="某条最终失败后继续后续计划；连接关闭重试策略仍生效")
    parser.add_argument("--dry-run", action="store_true", help="只预览组合，不读取密钥、不导入 SDK、不发请求或写日志")
    args = parser.parse_args(argv)
    if type(args.tls_retries) is not int or args.tls_retries < 0:
        parser.error("--tls-retries 必须为非负整数")
    if not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error("--retry-delay 必须为有限非负数")
    if args.resume:
        if any(value is not None for value in (args.b, args.n, args.max_tokens)):
            parser.error("--resume 自动恢复原批次配置，不能与 --b、--n、--max-tokens 同用")
        if args.timeout is not None and args.timeout <= 0:
            parser.error("--timeout 必须是正数")
        return resume_batch(args)
    args.b = B if args.b is None else args.b
    args.observation_format = OBSERVATION_FORMAT
    args.trust_env = TRUST_ENV if args.trust_env is None else args.trust_env
    if args.observation_format not in {"grid_3x3", "neighbor_counts"}:
        parser.error("OBSERVATION_FORMAT 必须为 grid_3x3 或 neighbor_counts")
    if type(args.trust_env) is not bool:
        parser.error("TRUST_ENV 必须为 True 或 False")
    args.n = N if args.n is None else args.n
    args.max_tokens = MAX_OUTPUT_TOKENS if args.max_tokens is None else args.max_tokens
    args.timeout = TIMEOUT_SECONDS if args.timeout is None else args.timeout
    if (not valid_b(args.b) or not isinstance(args.n, int) or isinstance(args.n, bool)
            or args.n < 1 or args.timeout <= 0 or not math.isfinite(args.timeout)):
        parser.error("B 必须为 None（未知）或有限数值，n 必须是正整数，timeout 必须为有限正数")
    if not isinstance(args.max_tokens, int) or isinstance(args.max_tokens, bool) or args.max_tokens < 16:
        parser.error("--max-tokens 至少为 16；推理和最终回答共用此上限，默认使用文件顶部设置")

    cases = build_cases(args.b, args.observation_format)
    total = len(cases) * args.n
    print(f"模型={MODEL} | {describe_b(args.b)} | {len(cases)} 种组合 × {args.n} 次 = {total} 次独立请求", flush=True)
    if args.dry_run:
        for index, case in enumerate(cases, 1):
            print(f"{index:3}. {describe_case(case)}，重复 {args.n} 次", flush=True)
        print("\n第一种局面的完整用户提示词：\n" + build_user_prompt(cases[0]), flush=True)
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
        **b_metadata(args.b),
        "planned_requests": total, "api_calls_started": 0, "processed_cases": 0,
        "successful_requests": 0, "failed_requests": 0, "cancelled_requests": 0,
        "not_run_requests": total,
        "archived_attempts": 0, "api_calls_started_total": 0,
        "observation_format": args.observation_format, "case_count": len(cases),
        "cases": [{**case, "C": 0, "D": 0, "errors": 0, "cancelled": 0} for case in cases],
    }
    exit_code = 0
    record = None
    filename = None
    run_lock = None
    archived = []
    try:
        (directory / "records").mkdir(parents=True, exist_ok=False)
        lock_path = directory / ".run.lock"
        with lock_path.open("x", encoding="utf-8") as lock:
            run_lock = lock_path
            lock.write(now())
        save_record(directory / "config.json", {
            "batch_id": batch_id, "model": MODEL, "base_url": base_url, "b": args.b, "n": args.n,
            **b_metadata(args.b),
            "planned_requests": total, "system_prompt": SYSTEM_PROMPT, "cases": cases,
            "observation_format": args.observation_format, "case_count": len(cases),
            "trust_env": args.trust_env, "verify": True, "runtime": runtime_info(),
            "requests": [make_request(case, args.max_tokens) for case in cases],
            "max_output_tokens": args.max_tokens, "timeout_seconds": args.timeout,
            "max_retries": 0, "store": False, "continue_on_error": args.continue_on_error,
            "tls_retries": args.tls_retries, "retry_delay_seconds": args.retry_delay,
            "order": ("每轮按行优先九格键枚举 C/D，保留全部旋转和镜像；共 n 轮；每次独立无历史。"
                      if args.observation_format == "grid_3x3" else
                      "每轮依次 C/0..4、D/0..4；共 n 轮；每次只提供当前局面，无历史。"),
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
                    **b_metadata(args.b),
                    "client_options": {"timeout_seconds": args.timeout, "max_retries": 0,
                                       "trust_env": args.trust_env, "verify": True},
                    "request": make_request(case, args.max_tokens), "api_call_started": False,
                    "started_at": now(), "finished_at": None, "duration_seconds": None,
                    "status": "running", "action": None, "output_text": None,
                    "response": None, "error": None,
                }
                save_record(directory / filename, record, api_key)
                print(f"[{index + 1}/{total}] 第{repetition}轮 {describe_b(args.b)} "
                      f"{describe_case(case)} -> ", end="", flush=True)
                call_with_retries(openai, api_key, base_url, record, args.timeout,
                                  directory=directory, archived=archived,
                                  retries=args.tls_retries, delay=args.retry_delay)

                account_record(summary, record)
                summary["archived_attempts"] = len(archived)
                summary["api_calls_started_total"] = summary["api_calls_started"] + sum(
                    bool(attempt.get("api_call_started")) for attempt in archived)

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
                if record is not None:
                    # 写日志失败时只补写内存中的响应，不重发已成功的请求。
                    account_record(summary, record)
                    try:
                        save_record(directory / filename, record, api_key)
                    except OSError:
                        summary["status"], exit_code = "logging_error", 1
                        print("当前记录无法保存；请先修复日志目录写入问题再续跑。", file=sys.stderr, flush=True)
                summary["archived_attempts"] = len(archived)
                summary["api_calls_started_total"] = summary["api_calls_started"] + sum(
                    bool(attempt.get("api_call_started")) for attempt in archived)
                save_record(directory / "summary.json", summary, api_key)
        except OSError:
            summary["status"] = "logging_error"
            exit_code = 1
            print("无法更新批次汇总；请检查 records/ 中已保存的逐条记录。", file=sys.stderr, flush=True)
        finally:
            if run_lock is not None:
                try:
                    run_lock.unlink(missing_ok=True)
                except OSError:
                    exit_code = 1
                    print("无法移除 .run.lock；确认没有运行进程后，请手动移除该锁文件。", file=sys.stderr)

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
