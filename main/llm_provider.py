"""LLM 服务选择及本地 Qwen 的 Chat Completions 适配；导入不会读取密钥或联网。"""

import re
import shlex
from pathlib import Path
from urllib.parse import urlsplit

if __package__:
    from .GPT_test import APIError as ConfigError
else:
    from GPT_test import APIError as ConfigError


PROVIDERS = ("gpt", "qwen")
QWEN_MODEL = "/home/tianyuxuan/qwen/Qwen3.8-27B-f16.gguf"
QWEN_REASONING_EFFORTS = ("low", "medium", "xhigh")


def normalize_provider(provider):
    if not isinstance(provider, str) or provider.strip().lower() not in PROVIDERS:
        raise ValueError("PROVIDER 必须为 'gpt' 或 'qwen'。")
    return provider.strip().lower()


def normalize_qwen_reasoning_effort(value):
    if not isinstance(value, str) or value.strip().lower() not in QWEN_REASONING_EFFORTS:
        raise ValueError("QWEN_REASONING_EFFORT 必须为 'low'、'medium' 或 'xhigh'。")
    return value.strip().lower()


def provider_env_file(provider):
    name = "Qwen.env" if normalize_provider(provider) == "qwen" else "gpt.env"
    return Path(__file__).resolve().parents[1] / "API" / name


def load_qwen_config():
    """只读取 Qwen.env，不回退到 GPT 配置、默认远程地址或进程环境变量。"""
    path = provider_env_file("qwen")
    if not path.is_file():
        raise ConfigError(f"找不到配置文件：{path}")
    values = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if not match:
            raise ConfigError(f"Qwen.env 第 {number} 行格式错误，应为 KEY=value。")
        try:
            parts = shlex.split(match[2], comments=True)
        except ValueError:
            raise ConfigError(f"Qwen.env 第 {number} 行引号未闭合。") from None
        if len(parts) > 1:
            raise ConfigError(f"Qwen.env 第 {number} 行的值包含空格，请使用引号。")
        values[match[1]] = parts[0] if parts else ""
    key = (values.get("GPT_API_KEY") or values.get("OPENAI_API_KEY") or "").strip()
    if not key or any(char.isspace() for char in key):
        raise ConfigError("请在 Qwen.env 中设置不含空白字符的 GPT_API_KEY。")
    url = (values.get("base_url") or values.get("GPT_BASE_URL")
           or values.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                 and not parsed.username and not parsed.password and not parsed.query
                 and not parsed.fragment and not any(char.isspace() for char in url))
        parsed.port  # 同时检查非法端口，不将包含密钥的 URL 写入错误信息。
    except ValueError:
        valid = False
    if not valid:
        raise ConfigError("请在 Qwen.env 中设置有效的 HTTP/HTTPS base_url，例如 http://localhost:8080/v1。")
    return key, url


def build_qwen_request(system_prompt, user_prompt, *, model=QWEN_MODEL, max_tokens=10240,
                       reasoning_effort=None):
    """一次完整、独立的咨询；不发送历史、响应 ID 或服务端会话标识。"""
    extra_body = {"cache_prompt": False, "reasoning_format": "deepseek"}
    if reasoning_effort is not None:
        extra_body["reasoning_effort"] = normalize_qwen_reasoning_effort(reasoning_effort)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "stream": False,
        # llama.cpp 扩展参数：关闭提示缓存，分离推理与最终正文；采样参数沿用服务端默认值。
        # None 不发送思考强度，以便复现添加该选项之前保存的批次请求。
        "extra_body": extra_body,
    }


def _plain(value):
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _plain(item) for key, item in vars(value).items() if not key.startswith("_")}
    return value


def chat_response_record(response):
    """完整保留原始回复（包括 reasoning_content），单独提取最终正文。"""
    raw = _plain(response)
    if not isinstance(raw, dict):
        return None, {"invalid_response_type": type(response).__name__}
    choices = raw.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    return (message.get("content") if isinstance(message, dict) else None), raw


class ChatAdviceError(ValueError):
    """Chat Completions 未产生完整有效的实验动作。"""


def validate_chat_response(response):
    text, raw = chat_response_record(response)
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ChatAdviceError("模型未返回唯一的有效 Chat Completions 回答。")
    choice = choices[0]
    reason = choice.get("finish_reason")
    if reason == "length":
        raise ChatAdviceError("模型达到 max_tokens 生成上限，本轮回答未完成；可增大 MAX_OUTPUT_TOKENS 后手动运行。")
    if reason == "content_filter":
        raise ChatAdviceError("模型输出被服务端中断（content_filter），本轮未得到完整有效建议。")
    if reason != "stop":
        raise ChatAdviceError("模型请求未正常完成，finish_reason 不是 stop，未得到有效建议。")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("refusal"):
        raise ChatAdviceError("模型拒绝回答或没有返回有效的 message，本轮结果无效。")
    if message.get("tool_calls") or message.get("function_call"):
        raise ChatAdviceError("模型返回了工具调用，本轮没有得到有效的 C/D 建议。")
    if not isinstance(text, str) or text.strip() not in {"C", "D"}:
        raise ChatAdviceError("模型没有仅返回 C 或 D，本次结果无效。")
    return text.strip()
