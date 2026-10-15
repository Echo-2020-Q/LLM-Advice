"""查询 4Router 令牌可见的模型，或测试一个指定的文本模型。

只使用 Python 标准库，无需 pip 安装依赖。在项目目录运行：
    python main/GPT_test.py
    python main/GPT_test.py --test 模型名称
    python main/GPT_test.py --test 模型名称 --api chat

默认读取 API/gpt.env 中的 GPT_API_KEY；可选配置 GPT_BASE_URL。
直接查询列表不会发送推理请求；--test 会发送一次短请求，可能产生费用。
"""

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / "API" / "gpt.env"
# 4Router 官方接入说明：https://docs.4router.net/guides/manual-config/
DEFAULT_BASE_URL = "https://4router.net/v1"


class APIError(Exception):
    """可以安全展示给用户的接口错误。"""


class NoRedirect(HTTPRedirectHandler):
    """避免把带密钥的请求自动重定向到其他地址。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def load_config(env_file, base_url=None):
    """读取简单的 KEY=value 配置，支持 BOM、引号、注释和 export。"""
    if not env_file.is_file():
        raise APIError(f"找不到配置文件：{env_file}")
    values = {}
    for number, line in enumerate(env_file.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if not match:
            raise APIError(f"配置文件第 {number} 行格式错误，应为 KEY=value。")
        try:
            parts = shlex.split(match[2], comments=True)
        except ValueError:
            raise APIError(f"配置文件第 {number} 行引号未闭合。") from None
        if len(parts) > 1:
            raise APIError(f"配置文件第 {number} 行的值包含空格，请使用引号。")
        values[match[1]] = parts[0] if parts else ""

    api_key = (values.get("GPT_API_KEY") or values.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise APIError("请在配置文件中设置 GPT_API_KEY=你的密钥。")
    if any(char.isspace() for char in api_key):
        raise APIError("API 密钥中包含空白字符，请检查配置文件。")
    url = (base_url or values.get("GPT_BASE_URL") or values.get("OPENAI_BASE_URL")
           or DEFAULT_BASE_URL).strip().rstrip("/")
    parsed = urlsplit(url)
    local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (not parsed.hostname or (parsed.scheme != "https" and not local_http)
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise APIError("Base URL 应是 HTTPS API 地址，例如 https://4router.net/v1。")
    return api_key, url


class APIClient:
    def __init__(self, api_key, base_url, timeout=30):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.opener = build_opener(NoRedirect())

    def redact(self, text):
        text = str(text).replace(self.api_key, "[密钥已隐藏]")
        return re.sub(r"sk-[A-Za-z0-9_-]+", "[密钥已隐藏]", text)

    def request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json", "Accept": "application/json",
                     "User-Agent": "LLM-Advice-model-check/1.0"},
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            hints = {
                401: "密钥无效、已禁用，或密钥与平台地址不匹配。",
                403: "令牌权限或模型分组不允许这次请求。",
                404: "接口或模型不存在；请核对 Base URL 和模型名称。",
                429: "请求频率受限或额度不足，请查看平台返回的原因。",
            }
            detail = ""
            try:
                error = json.loads(exc.read()).get("error", {})
                detail = error.get("message", "") if isinstance(error, dict) else str(error)
            except (ValueError, AttributeError):
                pass
            finally:
                exc.close()
            hint = hints.get(exc.code, "请检查平台服务和接口配置。")
            raise APIError(self.redact(f"HTTP {exc.code}：{hint} {detail}")[:1000]) from None
        except (URLError, TimeoutError, OSError):
            raise APIError("连接失败或超时，请检查网络、代理设置和 Base URL。") from None
        except (ValueError, UnicodeError):
            raise APIError("接口返回了非 JSON 数据，请确认使用的是 API 地址而非控制台网页。") from None
        if not isinstance(result, dict):
            raise APIError("接口返回格式不符合 OpenAI 兼容接口约定。")
        if result.get("error"):
            raise APIError(self.redact(f"平台返回错误：{result['error']}")[:1000])
        return result


def list_models(client):
    result = client.request("/models")
    models = result.get("data")
    if not isinstance(models, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]
        for item in models
    ):
        raise APIError("模型列表格式异常：预期 data 数组中的每项包含 id。")
    names = sorted({item["id"] for item in models})
    print(f"\n接口列出的模型（共 {len(names)} 个）：")
    for index, name in enumerate(names, 1):
        print(client.redact(f"{index:3}. {name}"))
    if not names:
        print("列表为空，请检查 4Router 控制台的令牌权限和模型分组。")
    print("\n列表不等于逐个调用成功；具体模型还受分组、余额和接口支持情况影响。")
    print("测试某个文本模型：python main/GPT_test.py --test 模型名称")


def probe_model(client, model, api, max_tokens):
    prompt = "Reply with only OK."
    print(client.redact(f"\n测试模型：{model}（{api} 接口，发送一次请求，可能产生费用）"), flush=True)
    if api == "responses":
        result = client.request("/responses", {
            "model": model, "input": prompt, "max_output_tokens": max_tokens, "store": False,
        })
        status = result.get("status")
        if (not isinstance(status, str) or status not in {"completed", "incomplete"}
                or not isinstance(result.get("output"), list)):
            raise APIError("未收到有效的 Responses 结果，尚不能确认调用成功。")
        texts = []
        for item in result["output"]:
            if not isinstance(item, dict) or not isinstance(item.get("content", []), list):
                raise APIError("Responses 返回的 output/content 格式异常。")
            for part in item.get("content", []):
                if not isinstance(part, dict):
                    raise APIError("Responses 返回的内容片段格式异常。")
                if part.get("type") == "output_text":
                    if not isinstance(part.get("text"), str):
                        raise APIError("Responses 返回的文本格式异常。")
                    texts.append(part["text"])
        if status == "incomplete":
            print("接口已接受请求，但输出未完成；可尝试增大 --max-tokens。")
        else:
            print("调用成功。")
    else:
        result = client.request("/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
        })
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise APIError("未收到有效的 Chat Completions 结果，尚不能确认调用成功。")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise APIError("接口没有返回有效的 message，尚不能确认调用成功。")
        finish_reason = choices[0].get("finish_reason")
        if (not isinstance(finish_reason, str)
                or finish_reason not in {"stop", "length", "tool_calls", "content_filter", "function_call"}
                or not any(field in message for field in ("content", "refusal", "tool_calls", "function_call"))):
            raise APIError("Chat 返回的 message 或 finish_reason 不完整，尚不能确认调用成功。")
        content = message.get("content") or message.get("refusal") or ""
        if not isinstance(content, str):
            raise APIError("Chat 返回的文本格式异常。")
        texts = [content]
        if finish_reason == "length":
            print("接口已接受请求，但输出达到长度上限；可尝试增大 --max-tokens。")
        else:
            print("调用成功。")
    print(client.redact("模型回复：" + ("\n".join(texts) or "（没有文本输出）")))
    print("本次结果只代表这个模型在所选接口上的一次测试。")


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE, help="密钥配置文件路径")
    parser.add_argument("--base-url", help="覆盖配置文件中的 API 地址，需要包含 /v1 等 API 前缀")
    parser.add_argument("--test", metavar="MODEL", help="实际测试一个模型；不指定时仅查询列表")
    parser.add_argument("--api", choices=("responses", "chat"), default="responses", help="测试接口，默认 responses")
    parser.add_argument("--max-tokens", type=positive_int, default=64, help="测试请求的输出 token 上限，默认 64")
    parser.add_argument("--timeout", type=positive_int, default=30, help="请求超时秒数，默认 30")
    args = parser.parse_args(argv)
    client = None
    try:
        api_key, base_url = load_config(args.env_file, args.base_url)
        client = APIClient(api_key, base_url, args.timeout)
        print(client.redact(f"API 地址：{base_url}"), flush=True)
        if args.test:
            probe_model(client, args.test, args.api, args.max_tokens)
        else:
            list_models(client)
        return 0
    except (APIError, OSError, UnicodeError, ValueError) as exc:
        message = str(exc) if isinstance(exc, APIError) else "读取配置失败，请检查文件权限、编码和地址格式。"
        print("错误：" + (client.redact(message) if client else message), file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        sys.exit(130)
