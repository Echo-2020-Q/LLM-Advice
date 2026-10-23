"""读取 API/Qwen.env，查询 /props 并用 /apply-template 检查默认模板渲染。

这两个诊断接口不会让模型生成回答。

运行：python main/test1.py 或 python -m main.test1。
"""

import json
from urllib.parse import urlsplit

import requests

if __package__:
    from .llm_provider import ConfigError, load_qwen_config
else:
    from llm_provider import ConfigError, load_qwen_config


THINKING_FIELDS = {
    "chat_template_kwargs", "enable_thinking", "reasoning_effort",
    "preserve_thinking", "reasoning_format", "reasoning_in_content",
    "thinking_budget", "reasoning_budget",
}
TEMPLATE_TERMS = ("enable_thinking", "reasoning_effort", "preserve_thinking")


def thinking_settings(data, prefix=""):
    """保留字段所在路径，避免混淆顶层配置与默认生成配置。"""
    found = {}
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if key in THINKING_FIELDS:
                found[path] = value
            elif isinstance(value, (dict, list)):
                found.update(thinking_settings(value, path))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            found.update(thinking_settings(value, f"{prefix}[{index}]"))
    return found


def show_rendered_template(session, server_url, show):
    print("\n/apply-template 默认渲染：只发送 Hello，未传思考参数；此接口不生成回答。")
    try:
        response = session.post(
            server_url + "/apply-template",
            json={"messages": [{"role": "user", "content": "Hello"}]},
            timeout=15,
            allow_redirects=False,
        )
    except requests.RequestException:
        print("模板诊断连接失败或超时；上方 /props 查询结果仍有效。")
        return
    print("/apply-template HTTP 状态码:", response.status_code)
    if not 200 <= response.status_code < 300:
        print("模板诊断接口不支持或请求失败；上方 /props 查询结果仍有效。")
        return
    try:
        data = response.json()
    except ValueError:
        print("模板诊断返回了非 JSON 数据；上方 /props 查询结果仍有效。")
        return
    if not isinstance(data, dict) or not isinstance(data.get("prompt"), str):
        print("模板诊断未返回 prompt 字符串；上方 /props 查询结果仍有效。")
        return
    print("服务端实际渲染的 prompt：")
    show(data["prompt"])
    print("此结果用于核对未覆盖思考参数时的默认渲染；具体生成请求仍可覆盖参数。")


def main():
    try:
        api_key, base_url = load_qwen_config()
    except (ConfigError, OSError, UnicodeError) as exc:
        # 配置解析器的错误不包含密钥；底层文件错误只显示类型。
        detail = str(exc) if isinstance(exc, ConfigError) else type(exc).__name__
        print(f"读取 Qwen.env 失败：{detail}")
        return 1

    # 仅移除末尾 /v1，保留反向代理等部署使用的其他路径前缀。
    server_url = base_url[:-3] if base_url.endswith("/v1") else base_url
    props_url = server_url + "/props"

    def show(value):
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        print(text.replace(api_key, "[密钥已隐藏]"))

    show(f"查询地址：{props_url}")
    try:
        with requests.Session() as session:
            if urlsplit(props_url).hostname in {"localhost", "127.0.0.1", "::1"}:
                session.trust_env = False
            session.headers.update({"Authorization": f"Bearer {api_key}"})
            response = session.get(
                props_url,
                timeout=15,
                allow_redirects=False,
            )

            print("HTTP 状态码:", response.status_code)
            if not 200 <= response.status_code < 300:
                print("查询失败：请检查认证、/props 路由及服务状态；未获得默认参数。")
                return 1
            try:
                data = response.json()
            except ValueError:
                print("查询失败：/props 返回了非 JSON 数据。")
                return 1
            if not isinstance(data, dict):
                print("查询失败：/props 应返回 JSON 对象。")
                return 1

            print("\nAPI 实际返回的服务信息：")
            for field in ("model_path", "build_info", "total_slots", "modalities",
                          "default_generation_settings"):
                print(f"{field}:")
                show(data[field] if field in data else "（接口未返回）")

            print("\nAPI 实际返回的思考相关配置（字段路径 → 值）：")
            settings = thinking_settings(data)
            show(settings if settings else "（接口未返回可识别的思考配置字段）")

            print("\nchat_template 中的思考相关行（模板逻辑，不等同于运行时配置）：")
            template = data.get("chat_template")
            lines = [f"{number}: {line}" for number, line in enumerate(template.splitlines(), 1)
                     if any(term in line for term in TEMPLATE_TERMS)] if isinstance(template, str) else []
            show("\n".join(lines) if lines else "（接口未返回模板或未找到相关行）")
            print("\n模板默认值仅在变量未被覆盖时适用。字段缺失不能证明实际思考档位；")
            print("/props 展示服务默认配置，具体请求仍可能通过参数覆盖。")
            show_rendered_template(session, server_url, show)
    except requests.RequestException:
        print("连接失败或超时：请检查 Qwen.env 中的 base_url、网络及服务状态。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
