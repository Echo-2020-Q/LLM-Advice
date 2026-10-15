"""用 GPT-5.5 为一次空间囚徒困境决策提供建议。

安装依赖：python -m pip install -r requirements.txt
在项目目录运行：python main/GPT_onestep_test.py
成功时只输出 C 或 D；错误写入 stderr，并以非零状态退出。
每次运行的输入、输出和状态保存为 logs/GPT_onestep_test/ 下的独立 JSON 文件。
每次运行发送一次模型请求，可能产生 API 费用。
"""

import json
import re
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
MAX_OUTPUT_TOKENS = 1024  # 包含推理 token，不能因为只输出一个字母就设成 1。
TIMEOUT_SECONDS = 120
LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "GPT_onestep_test"
LOG_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")

SYSTEM_PROMPT = """You are an independent decision advisor for an individual participating in a repeated spatial Prisoner's Dilemma game.

The game follows these rules:

1. Each player occupies a fixed position in a square lattice and interacts with four neighbors.
2. In each round, every player chooses one of two actions: Cooperate (C) or Defect (D).
3. The payoff from each pairwise interaction is:
   - C vs. C: both receive 1.
   - C vs. D: the cooperator receives 0 and the defector receives b.
   - D vs. D: both receive 0.
4. Each player's total payoff is the sum of the payoffs from the four interactions.
5. Players may revise their actions between rounds. Future actions of their neighbors are unknown.

Your task is to advise one focal player about which action to take in the next round, using only the information provided.

You are an advisor, not a participant in the game.

Do not assume any additional information, hidden objectives, or guaranteed future behavior.

Recommend either C or D. No explanation is required."""

USER_PROMPT = """Game parameter:
b = 1.5

Current observation of the focal player:
- Current action: D
- Number of cooperating neighbors: 3
- Number of defecting neighbors: 1

What action would you recommend for this player in the next round?

Return only one action: C or D."""


class AdviceError(ValueError):
    """模型未返回符合实验要求的动作。"""


class RunError(Exception):
    """不含密钥的运行错误诊断。"""

    def __init__(self, message, category, http_status=None):
        super().__init__(message)
        self.category = category
        self.http_status = http_status


def build_request(user_prompt):
    return {
        "model": MODEL,
        "instructions": SYSTEM_PROMPT,
        "input": user_prompt,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
    }


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
        "packages": packages,
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
    return {
        "exception_type": type(exc).__name__,
        "detail": redact(exc, api_key),
        "frames": frames,
    }


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


def ask_advisor(client, user_prompt=USER_PROMPT, *, record=None):
    """发起一次独立咨询；在验证前记录原始输出，保留无效结果。"""
    request = build_request(user_prompt)
    if record is not None:
        record["request"] = request
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
        raise AdviceError("模型输出未完成，可增大 MAX_OUTPUT_TOKENS 后重新运行。")
    if getattr(response, "status", None) != "completed":
        raise AdviceError("模型请求未正常完成，未得到有效建议。")

    text = getattr(response, "output_text", None)
    if not isinstance(text, str) or text.strip() not in {"C", "D"}:
        # 不截取解释中的字母、不默认选某个动作，避免改变实验结果。
        raise AdviceError("模型没有仅返回 C 或 D，本次结果无效。")
    return text.strip()


def run_advisor(api_key, base_url, record):
    # ImportError 交由 main 统一处理，也覆盖客户端初始化时加载的代理依赖。
    from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
    from openai import APIError as SDKError

    try:
        with OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=TIMEOUT_SECONDS,
            max_retries=0,  # 单次实验不自动重试，避免重复请求。
        ) as client:
            return ask_advisor(client, record=record)
    except APITimeoutError:
        raise RunError("请求超时，请检查网络或增大 TIMEOUT_SECONDS。", "timeout") from None
    except APIConnectionError:
        raise RunError("连接失败，请检查网络、代理和 API 地址。", "connection_error") from None
    except APIStatusError as exc:
        # 不打印 SDK 的完整异常，避免服务端错误信息包含密钥。
        hints = {
            400: "请求参数不被平台接受，请检查模型及 Responses 接口配置。",
            401: "密钥无效或与平台地址不匹配。",
            403: "请检查令牌的模型分组和权限。",
            404: "模型或接口不存在。",
            429: "请求受限或额度不足。",
        }
        hint = hints.get(exc.status_code, "请检查平台服务状态。")
        raise RunError(
            f"API 请求失败（HTTP {exc.status_code}）：{hint}", "http_error", exc.status_code
        ) from None
    except SDKError:
        raise RunError("SDK 无法处理接口返回，请检查平台的 OpenAI 接口兼容性。", "sdk_error") from None


def main():
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
        "client_options": {"timeout_seconds": TIMEOUT_SECONDS, "max_retries": 0},
        "request": build_request(USER_PROMPT),
        "status": "running",
        "action": None,
        "output_text": None,
        "response": None,
        "error": None,
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
    try:
        record["runtime"] = runtime_info()
        api_key, base_url = load_config(DEFAULT_ENV_FILE)
        record["base_url"] = base_url
        record["action"] = run_advisor(api_key, base_url, record)
        record["status"] = "success"
        exit_code = 0
    except RunError as exc:
        record["error"] = {"type": exc.category, "message": str(exc), "http_status": exc.http_status}
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
        record["finished_at"] = datetime.now(LOG_TIMEZONE).isoformat(timespec="milliseconds")
        record["duration_seconds"] = round(time.perf_counter() - timer, 3)
        try:
            save_record(log_path, record, api_key)
        except OSError:
            print(f"运行记录保存失败，原记录可能仍为 running 状态：{log_path}", file=sys.stderr)
            return 1

    print(f"运行记录：{log_path}", file=sys.stderr)
    if exit_code == 0:
        print(record["action"])
    elif record["error"]:
        message = redact(record["error"]["message"], api_key)
        print(f"错误：{message}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        sys.exit(130)
