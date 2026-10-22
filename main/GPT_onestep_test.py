"""使用 GPT-5.5 进行一次或连续两次空间囚徒困境决策咨询。

安装依赖：python -m pip install -r requirements.txt
在项目目录运行：python main/GPT_onestep_test.py
直接修改 USER_PROMPT 和 USER_PROMPT_2 中的 3×3 C/D 网格；中心是第 2 行第 2 列。
SINGLE_QUESTION=True 时只询问 USER_PROMPT；False 时依次询问 USER_PROMPT 和 USER_PROMPT_2。
每次只为中心个体输出 C 或 D。第一轮发送 SYSTEM_PROMPT。
多轮模式通过 previous_response_id 续接，共用 HTTP 客户端保留连接池和 Cookie，依赖平台响应存储。
多轮模式保留上下文，不是独立图案采样；第二张网格由你填写，不由第一轮建议自动演化。
store=True，最多发送一或两次请求，可能产生 API 费用。
各轮的输入、输出和状态逐轮保存到 logs/GPT_onestep_test/ 下同一个独立 JSON 文件。
错误写入 stderr，并以非零状态退出；失败时停止，不自动重试。
检查已有第一轮记录：python main/GPT_onestep_test.py --check-response 日志文件路径
检查模式只发送一次 GET 查询，不生成新回答，也不会继续第二轮。
"""

import argparse
import json
import os
import re
import ssl
import sys
import time
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

if __package__:
    from .GPT_test import APIError as ConfigError, DEFAULT_ENV_FILE, load_config
else:
    from GPT_test import APIError as ConfigError, DEFAULT_ENV_FILE, load_config


MODEL = "gpt-5.5"
SINGLE_QUESTION = True  # True：只问 USER_PROMPT；False：连续问 USER_PROMPT 和 USER_PROMPT_2。
MAX_OUTPUT_TOKENS = 10240  # 推理和最终回答共用此上限；1024 曾在推理阶段耗尽。
TIMEOUT_SECONDS = 120
# True：沿用代理和证书环境设置；False：忽略这些环境设置，用默认验证证书直接连接。
# 排查 TLS/代理故障时可切换；不修改系统设置，也不会自动切换或重试。
TRUST_ENV = True
LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "GPT_onestep_test"
LOG_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")

# 更新规则按当前 SPD 的同步更新 + Fermi/建议混合机制描述。
# 此测试脚本不自动读取 SPD 的参数；若改变更新机制，请同步修改这里的文字。
# K、资格比例和咨询概率未指定数值，避免把单组参数误当成所有扫描实验的设定。
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

# 直接编辑下面三行 C/D；不要另填邻居数量，以免与网格矛盾。
# 示例 1：中心 C，上/左为 C，下/右为 D；左上角 C 同时连接上方和左方合作者。
USER_PROMPT = """Game state:

Current local 3x3 grid (C = Cooperate, D = Defect):
C C D
C C D
D D D

I am the center player at row 2, column 2.

My current payoff is 2.

What action would you recommend for the center player in the next round to maximizes my expected cumulative payoff over the entire spatial repeated game.?

Return only one action: C or D."""

# 示例 2：与上面仅左上角不同；中心仍为 C，四个直接邻居的位置和策略均相同。
# 可单独编辑；第二次咨询仍会带着第一次的对话上下文。
USER_PROMPT_2 = """Game state:

Current local 3x3 grid (C = Cooperate, D = Defect):
D C D
C C D
D D D

I am the center player at row 2, column 2.

My current payoff is 2.

What action would you recommend for the center player in the next round to maximizes my expected cumulative payoff over the entire spatial repeated game.?

Return only one action: C or D."""


class AdviceError(ValueError):
    """模型未返回符合实验要求的动作。"""


class RunError(Exception):
    """不含密钥的运行错误诊断。"""

    def __init__(self, message, category, http_status=None, diagnostics=None):
        super().__init__(message)
        self.category = category
        self.http_status = http_status
        self.diagnostics = diagnostics


def build_request(user_prompt, *, previous_response_id=None):
    messages = [{"role": "user", "content": user_prompt}]
    if previous_response_id is None:
        # instructions 不会随 previous_response_id 继承，因此把系统指令作为首条消息保存。
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    elif not isinstance(previous_response_id, str) or not previous_response_id.strip():
        raise ValueError("续接对话需要有效的 previous_response_id。")
    request = {
        "model": MODEL,
        "input": messages,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": True,
    }
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    return request


def redact(text, api_key=""):
    """日志和终端使用相同的密钥脱敏规则。"""
    text = str(text)
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [REDACTED]", text)
    return re.sub(r"(?i)([a-z][a-z0-9+.-]*://)[^\s/]+@", r"\1[REDACTED]@", text)


def runtime_info():
    """记录实际解释器及依赖版本，便于排查 IDE 与终端的环境差异。"""
    packages = {}
    for name in ("openai", "httpx", "httpx2", "socksio", "pydantic", "typing_extensions"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "openssl_version": ssl.OPENSSL_VERSION,
        "packages": packages,
        "network_environment": {
            "proxy_variables_present": [
                name for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                                  "http_proxy", "https_proxy", "all_proxy", "no_proxy")
                if os.environ.get(name)
            ],
            "custom_ca_variables_present": [
                name for name in ("SSL_CERT_FILE", "SSL_CERT_DIR") if os.environ.get(name)
            ],
        },
    }


def exception_diagnostics(exc, api_key=""):
    """保留脱敏后的异常与调用位置，不记录源代码或局部变量。"""
    frames = []
    frame = exc.__traceback__
    while frame is not None:
        frames.append({
            "filename": redact(frame.tb_frame.f_code.co_filename, api_key),
            "lineno": frame.tb_lineno,
            "function": frame.tb_frame.f_code.co_name,
        })
        frame = frame.tb_next
    diagnostics = {
        "exception_type": type(exc).__name__,
        "detail": redact(exc, api_key),
        "frames": frames,
    }
    # SDK 的顶层 Connection error 通常隐藏了实际代理、TLS 或网络错误。
    causes = []
    seen = {id(exc)}
    current = exc
    for _ in range(8):
        cause = current.__cause__
        if cause is None and not current.__suppress_context__:
            cause = current.__context__
        if cause is None or id(cause) in seen:
            break
        seen.add(id(cause))
        causes.append({
            "exception_type": type(cause).__name__,
            "detail": redact(cause, api_key),
        })
        current = cause
    if causes:
        diagnostics["causes"] = causes
    return diagnostics


def save_record(path, record, api_key=""):
    """先写临时文件再替换，避免将已有记录改写成不完整的 JSON。"""
    content = json.dumps(record, ensure_ascii=False, indent=2, default=str)
    if api_key:
        escaped_key = json.dumps(api_key, ensure_ascii=False)[1:-1]
        content = content.replace(escaped_key, "[REDACTED]")
    content = redact(content)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(content + "\n", encoding="utf-8")
    temporary.replace(path)


def ask_advisor(client, user_prompt=USER_PROMPT, *, record=None, previous_response_id=None):
    """发起一轮咨询；有响应 ID 时延续对话，在验证前记录原始输出。"""
    request = build_request(user_prompt, previous_response_id=previous_response_id)
    if record is not None:
        record["request"] = request
        record["api_call_started"] = True
    response = client.responses.create(**request)
    if record is not None:
        record["output_text"] = getattr(response, "output_text", None)
        dump = getattr(response, "model_dump", None)
        record["response"] = dump(mode="json") if callable(dump) else {
            "id": getattr(response, "id", None),
            "status": getattr(response, "status", None),
            "model": getattr(response, "model", None),
            "output_text": record["output_text"],
        }
    if getattr(response, "status", None) == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = details.get("reason") if isinstance(details, dict) else getattr(details, "reason", None)
        if reason == "max_output_tokens":
            raise AdviceError(
                f"模型达到 max_output_tokens={request['max_output_tokens']} 的生成上限（包含内部推理 token），"
                "本轮未得到完整有效建议。可增大 MAX_OUTPUT_TOKENS 后手动运行。"
            )
        if reason == "content_filter":
            raise AdviceError("模型输出被服务端中断（content_filter），本轮未得到完整有效建议。")
        raise AdviceError("模型输出未完成，本轮未得到有效建议；具体原因见日志 response.incomplete_details。")
    if getattr(response, "status", None) != "completed":
        raise AdviceError("模型请求未正常完成，未得到有效建议。")

    text = getattr(response, "output_text", None)
    if not isinstance(text, str) or text.strip() not in {"C", "D"}:
        # 不截取解释中的字母、不默认选某个动作，避免改变实验结果。
        raise AdviceError("模型没有仅返回 C 或 D，本次结果无效。")
    return text.strip()


def create_client(api_key, base_url):
    """为整段对话保留同一个 HTTP 客户端、连接池和 Cookie。"""
    from openai import DefaultHttpxClient, OpenAI

    http_client = DefaultHttpxClient(trust_env=TRUST_ENV, verify=True)
    try:
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=TIMEOUT_SECONDS,
            max_retries=0,  # 每轮都不自动重试，避免重复请求。
            http_client=http_client,
        )
    except BaseException:
        http_client.close()
        raise


def run_advisor(api_key, base_url, record, *, user_prompt=USER_PROMPT, previous_response_id=None,
                client=None):
    # ImportError 交由 main 统一处理，也覆盖客户端初始化时加载的代理依赖。
    from openai import APIConnectionError, APIStatusError, APITimeoutError
    from openai import APIError as SDKError

    try:
        if client is None:
            with create_client(api_key, base_url) as standalone_client:
                return ask_advisor(standalone_client, user_prompt, record=record,
                                   previous_response_id=previous_response_id)
        return ask_advisor(client, user_prompt, record=record, previous_response_id=previous_response_id)
    except APITimeoutError as exc:
        raise RunError(
            "请求超时，请检查网络或增大 TIMEOUT_SECONDS；底层原因详见日志 error.diagnostics。",
            "timeout", diagnostics=exception_diagnostics(exc, api_key),
        ) from None
    except APIConnectionError as exc:
        diagnostics = exception_diagnostics(exc, api_key)
        details = [diagnostics["detail"]] + [cause["detail"] for cause in diagnostics.get("causes", [])]
        tls_eof = any("TLS/SSL connection has been closed" in detail
                      or "UNEXPECTED_EOF_WHILE_READING" in detail for detail in details)
        message = "连接失败，请检查网络、代理和 API 地址；底层原因详见日志 error.diagnostics。"
        if tls_eof:
            message = ("连接失败：TLS/SSL 握手连接被关闭（EOF），尚未收到 HTTP 响应。"
                       "请先测试平台首页的 HTTPS 连接；可切换 TRUST_ENV 比较环境代理路径与直接连接。"
                       "底层原因详见日志 error.diagnostics。")
        raise RunError(
            message, "connection_error", diagnostics=diagnostics,
        ) from None
    except APIStatusError as exc:
        # 不打印 SDK 的完整异常，避免服务端错误信息包含密钥。
        diagnostics = exception_diagnostics(exc, api_key)
        error_code = getattr(exc, "code", None)
        if not error_code:
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                service_error = body.get("error", body)
                if isinstance(service_error, dict):
                    error_code = service_error.get("code")
        if isinstance(error_code, str):
            diagnostics["api_error_code"] = redact(error_code, api_key)
        request_id = getattr(exc, "request_id", None)
        if isinstance(request_id, str):
            diagnostics["request_id"] = redact(request_id, api_key)
        if previous_response_id is not None and error_code == "previous_response_not_found":
            raise RunError(
                "平台找不到上一轮的响应记录（previous_response_not_found）。第一轮回答已保存。"
                "继续使用响应 ID 续接，需要平台确认响应存储是否可用，并检查两轮的上游账号或渠道路由。"
                "请将日志中的时间、响应 ID 和错误码提供给平台排查。",
                "conversation_not_found", exc.status_code, diagnostics,
            ) from None
        hints = {
            400: "请求参数不被平台接受，请检查平台是否支持 Responses、store=True 和 previous_response_id。",
            401: "密钥无效或与平台地址不匹配。",
            403: "请检查令牌的模型分组和权限。",
            404: "模型、接口或上一轮响应不存在，请检查平台是否支持对话接续。",
            429: "请求受限或额度不足。",
        }
        hint = hints.get(exc.status_code, "请检查平台服务状态。")
        if exc.status_code == 400 and previous_response_id is not None:
            hint = "平台拒绝了带 previous_response_id 的续接请求，请根据日志中的服务端详情定位原因。"
        raise RunError(
            f"API 请求失败（HTTP {exc.status_code}）：{hint}"
            " 服务端错误详情已脱敏保存在日志 error.diagnostics 中。",
            "http_error", exc.status_code, diagnostics,
        ) from None
    except SDKError:
        raise RunError("SDK 无法处理接口返回，请检查平台的 OpenAI 接口兼容性。", "sdk_error") from None


def main():
    user_prompts = [USER_PROMPT] if SINGLE_QUESTION else [USER_PROMPT, USER_PROMPT_2]
    started = datetime.now(LOG_TIMEZONE)
    timer = time.perf_counter()
    run_id = uuid4().hex
    log_path = LOG_DIR / f"{started:%Y%m%d_%H%M%S_%f}_{run_id}.json"
    record = {
        "run_id": run_id,
        "started_at": started.isoformat(timespec="milliseconds"),
        "finished_at": None,
        "duration_seconds": None,
        "runtime": None,
        "base_url": None,
        "client_options": {"timeout_seconds": TIMEOUT_SECONDS, "max_retries": 0, "trust_env": TRUST_ENV,
                           "shared_across_turns": True},
        "request": build_request(USER_PROMPT),
        "status": "running",
        "action": None,
        "output_text": None,
        "response": None,
        "error": None,
        "single_question": SINGLE_QUESTION,
        "conversation_mode": "single_turn" if SINGLE_QUESTION else "previous_response_id",
        "planned_turns": len(user_prompts),
        "completed_turns": 0,
        "user_prompts": user_prompts,
        "turns": [],
    }
    # 先保存输入。日志目录不可写时，不发送无法记录结果的模型请求。
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        save_record(log_path, record)
    except OSError:
        print(f"无法创建运行记录，请检查目录权限：{LOG_DIR}", file=sys.stderr)
        return 1

    api_key = ""
    exit_code = 1
    current_turn = None

    def checkpoint():
        try:
            save_record(log_path, record, api_key)
        except OSError:
            raise RunError("无法保存对话记录，已停止后续请求；请检查日志目录权限和磁盘空间。",
                           "logging_error") from None

    try:
        record["runtime"] = runtime_info()
        api_key, base_url = load_config(DEFAULT_ENV_FILE)
        record["base_url"] = base_url
        previous_response_id = None
        with create_client(api_key, base_url) as client:
            for turn_number, user_prompt in enumerate(record["user_prompts"], start=1):
                current_turn = {
                    "turn": turn_number,
                    "started_at": datetime.now(LOG_TIMEZONE).isoformat(timespec="milliseconds"),
                    "finished_at": None, "duration_seconds": None,
                    "request": build_request(user_prompt, previous_response_id=previous_response_id),
                    "api_call_started": False, "status": "running", "action": None,
                    "output_text": None, "response": None, "error": None,
                }
                record["turns"].append(current_turn)
                for name in ("request", "action", "output_text", "response"):
                    record[name] = current_turn[name]
                checkpoint()  # 本轮输入落盘后才发请求。
                turn_timer = time.perf_counter()
                try:
                    current_turn["action"] = run_advisor(
                        api_key, base_url, current_turn,
                        user_prompt=user_prompt, previous_response_id=previous_response_id,
                        client=client,
                    )
                    current_turn["status"] = "success"
                    record["completed_turns"] += 1
                finally:
                    current_turn["finished_at"] = datetime.now(LOG_TIMEZONE).isoformat(timespec="milliseconds")
                    current_turn["duration_seconds"] = round(time.perf_counter() - turn_timer, 3)
                    for name in ("request", "action", "output_text", "response"):
                        record[name] = current_turn[name]
                checkpoint()  # 本轮结果成功保存后才允许进行下一轮。
                print(f"第 {turn_number} 轮建议：{current_turn['action']}", flush=True)
                if turn_number < record["planned_turns"]:
                    response = current_turn["response"]
                    previous_response_id = response.get("id") if isinstance(response, dict) else None
                    if not isinstance(previous_response_id, str) or not previous_response_id.strip():
                        raise RunError("第一轮未返回有效响应 ID，无法接续同一对话，已停止第二轮。",
                                       "conversation_error")
        record["status"] = "success"
        exit_code = 0
    except RunError as exc:
        record["error"] = {"type": exc.category, "message": str(exc), "http_status": exc.http_status}
        if exc.diagnostics is not None:
            record["error"]["diagnostics"] = exc.diagnostics
    except (ConfigError, AdviceError) as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    except ImportError as exc:
        detail = redact(exc, api_key)
        requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
        record["error"] = {
            "type": "dependency_error",
            "message": (
                f"Python 依赖加载失败：{detail}\n"
                f"当前解释器：{sys.executable}\n"
                "请在该解释器对应的环境中安装项目依赖："
                f'python -m pip install -r "{requirements}"'
            ),
            "diagnostics": exception_diagnostics(exc, api_key),
        }
    except (OSError, UnicodeError):
        record["error"] = {"type": "configuration_error", "message": "无法读取配置，请检查 API/gpt.env 的权限和 UTF-8 编码。"}
    except ValueError:
        record["error"] = {"type": "format_error", "message": "配置或接口返回格式异常，请检查 API 地址。"}
    except KeyboardInterrupt:
        record["status"] = "cancelled"
        record["error"] = {"type": "KeyboardInterrupt", "message": "已取消。"}
        exit_code = 130
    except Exception as exc:
        record["error"] = {
            "type": type(exc).__name__,
            "message": f"程序遇到未预期错误（{type(exc).__name__}）：{redact(exc, api_key)}",
            "diagnostics": exception_diagnostics(exc, api_key),
        }
    finally:
        if record["status"] == "running":
            record["status"] = "error"
        if current_turn is not None and current_turn["status"] == "running":
            current_turn["status"] = "cancelled" if record["status"] == "cancelled" else "error"
            current_turn["error"] = record["error"]
            current_turn["finished_at"] = current_turn["finished_at"] or datetime.now(LOG_TIMEZONE).isoformat(timespec="milliseconds")
        record["finished_at"] = datetime.now(LOG_TIMEZONE).isoformat(timespec="milliseconds")
        record["duration_seconds"] = round(time.perf_counter() - timer, 3)
        try:
            save_record(log_path, record, api_key)
        except OSError:
            print(f"运行记录保存失败，原记录可能仍为 running 状态：{log_path}", file=sys.stderr)
            return 1

    print(f"运行记录：{log_path}", file=sys.stderr)
    if exit_code != 0 and record["error"]:
        message = redact(record["error"]["message"], api_key)
        print(f"错误：{message}", file=sys.stderr)
    return exit_code


def check_response_log(log_file):
    """只检索已有响应，帮助排查存储或路由；不发模型生成请求。"""
    try:
        source_path = Path(log_file).expanduser().resolve()
        source = json.loads(source_path.read_text(encoding="utf-8"))
        first = source["turns"][0]
        response_id = first["response"]["id"]
        if (first.get("status") != "success"
                or first.get("request", {}).get("store") is not True
                or not isinstance(response_id, str)
                or re.fullmatch(r"[A-Za-z0-9_-]+", response_id) is None):
            raise ValueError("源日志没有成功且要求存储的第一轮响应 ID。")
        source_base_url = source.get("base_url")
        if not isinstance(source_base_url, str):
            raise ValueError("源日志缺少接口地址。")
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        print("无法检查：请提供包含第一轮成功响应、store=True 和接口地址的运行日志。", file=sys.stderr)
        return 1

    api_key = ""
    try:
        api_key, base_url = load_config(DEFAULT_ENV_FILE)
        if base_url.rstrip("/") != source_base_url.rstrip("/"):
            print("无法检查：当前 API 地址与源日志不同，请使用生成该响应时的地址和令牌。", file=sys.stderr)
            return 1
    except (ConfigError, OSError, UnicodeError) as exc:
        print(f"无法读取配置：{redact(exc, api_key)}", file=sys.stderr)
        return 1

    started = datetime.now(LOG_TIMEZONE)
    timer = time.perf_counter()
    log_path = LOG_DIR / "response_checks" / f"{started:%Y%m%d_%H%M%S_%f}_{uuid4().hex}.json"
    record = {
        "mode": "check_response", "source_log": str(source_path),
        "source_started_at": source.get("started_at"),
        "started_at": started.isoformat(timespec="milliseconds"),
        "base_url": base_url, "response_id": response_id,
        "request": {"method": "GET", "path": f"/responses/{response_id}"},
        "client_options": {"timeout_seconds": TIMEOUT_SECONDS, "max_retries": 0, "trust_env": TRUST_ENV},
        "status": "running", "api_call_started": False, "response": None, "error": None,
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        save_record(log_path, record, api_key)
    except OSError:
        print("无法保存检查记录，未发送查询。", file=sys.stderr)
        return 1

    exit_code = 1
    message = ""
    try:
        with create_client(api_key, base_url) as client:
            record["api_call_started"] = True
            response = client.responses.retrieve(response_id)
        returned_id = getattr(response, "id", None)
        record["response"] = {
            name: getattr(response, name, None)
            for name in ("id", "status", "model", "store", "previous_response_id")
        }
        record["request_id"] = getattr(response, "_request_id", None)
        if returned_id != response_id:
            raise ValueError("查询没有返回与源日志匹配的响应 ID，不能确认记录可访问。")
        record["status"] = "found"
        message = "平台本次可以读取第一轮响应；这不保证后续续接请求会使用相同的上游。"
        exit_code = 0
    except KeyboardInterrupt as exc:
        record["status"] = "cancelled"
        record["error"] = {"type": "KeyboardInterrupt", "diagnostics": exception_diagnostics(exc, api_key)}
        message = "检查已取消。"
        exit_code = 130
    except Exception as exc:
        record["status"] = "query_failed"
        diagnostics = exception_diagnostics(exc, api_key)
        error_code = getattr(exc, "code", None)
        if error_code is None:
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                error = body.get("error", body)
                if isinstance(error, dict):
                    error_code = error.get("code")
        if isinstance(error_code, str):
            diagnostics["api_error_code"] = redact(error_code, api_key)
        request_id = getattr(exc, "request_id", None)
        if isinstance(request_id, str):
            diagnostics["request_id"] = redact(request_id, api_key)
        record["error"] = {"type": type(exc).__name__, "http_status": getattr(exc, "status_code", None),
                           "diagnostics": diagnostics}
        message = ("未能读取该响应。请结合服务端详情判断：可能是记录不可访问、查询接口未支持、"
                   "读取权限不足或连接失败，不能仅凭这次查询确定存储或路由原因。")
    finally:
        record["finished_at"] = datetime.now(LOG_TIMEZONE).isoformat(timespec="milliseconds")
        record["duration_seconds"] = round(time.perf_counter() - timer, 3)
        try:
            save_record(log_path, record, api_key)
        except OSError:
            print(f"检查记录保存失败：{log_path}", file=sys.stderr)
            return 1

    print(f"检查结果：{message}", flush=True)
    if record["error"] is not None:
        print(f"服务端详情：{record['error']['diagnostics']['detail']}", file=sys.stderr)
    print(f"检查记录：{log_path}", file=sys.stderr)
    return exit_code


def cli(argv=None):
    parser = argparse.ArgumentParser(description="按 SINGLE_QUESTION 设置进行一次或两轮决策咨询，或只查询已有响应记录。")
    parser.add_argument("--check-response", metavar="LOG_FILE", help="只查询日志中的第一轮响应，不生成新回答")
    args = parser.parse_args(argv)
    if args.check_response is not None:
        return check_response_log(args.check_response)
    return main()


if __name__ == "__main__":
    try:
        sys.exit(cli())
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        sys.exit(130)
