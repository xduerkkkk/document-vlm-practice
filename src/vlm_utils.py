from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


DEFAULT_VLM_PROMPT = """
请你阅读这页文档图片，并完成页面理解。 

请输出以下内容：

1. 页面类型：例如论文页、课件页、报告页、表格页、扫描件等。
2. 页面主要内容摘要。
3. 页面结构：标题、正文、表格、图片、公式、列表等。
4. 如果有图表或表格，请解释它大概表达了什么。
5. 如果适合，请用 Markdown 形式整理页面内容。

要求：
- 不要编造页面中不存在的信息。
- 如果看不清，请明确说明。
- 输出中文。
""".strip()


def image_to_data_url(image_path: str | Path) -> str:
    """
    将本地图片转成 base64 data URL，供多模态 API 使用。
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"

    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def get_vlm_config() -> Dict[str, str]:
    """
    从 .env 读取 VLM API 配置。

    需要配置：
        VLM_API_KEY
        VLM_BASE_URL
        VLM_MODEL
    """
    load_dotenv()

    api_key = os.getenv("VLM_API_KEY", "").strip()
    base_url = os.getenv("VLM_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("VLM_MODEL", "").strip()

    if not api_key:
        raise ValueError("未配置 VLM_API_KEY，请在 .env 中填写。")

    if not base_url:
        raise ValueError("未配置 VLM_BASE_URL，请在 .env 中填写。")

    if not model:
        raise ValueError("未配置 VLM_MODEL，请在 .env 中填写。")

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


def call_vlm_for_image(
    image_path: str | Path,
    prompt: str = DEFAULT_VLM_PROMPT,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    timeout: int = 120,
) -> str:
    """
    调用兼容 chat/completions 形式的多模态接口。

    注意：
    - 这里默认使用 image_url + base64 data URL 形式。
    - 不同平台的字段可能略有差异，如果你后面确定具体平台，可以再单独适配。
    """
    config = get_vlm_config()

    image_url = image_to_data_url(image_path)

    endpoint = f"{config['base_url']}/chat/completions"

    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"VLM API 调用失败，status={response.status_code}, body={response.text[:1000]}"
        )

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"无法解析 VLM 返回结果: {data}") from e


def run_vlm_on_pages(
    page_images: List[str],
    max_pages: int = 1,
    prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    对多页图片执行 VLM-only 页面理解。
    """
    results: List[Dict[str, Any]] = []

    prompt = prompt or DEFAULT_VLM_PROMPT

    for page_index, image_path in enumerate(page_images[:max_pages], start=1):
        content = call_vlm_for_image(
            image_path=image_path,
            prompt=prompt,
        )

        results.append(
            {
                "page_index": page_index,
                "image_path": image_path,
                "prompt": prompt,
                "vlm_output": content,
            }
        )

    return results


def format_vlm_markdown(vlm_results: List[Dict[str, Any]]) -> str:
    """
    将 VLM-only 结果格式化成 Markdown。
    """
    if not vlm_results:
        return "暂无 VLM 结果。"

    chunks: List[str] = ["# VLM-only 页面理解结果\n"]

    for page in vlm_results:
        page_index = page["page_index"]
        image_path = page["image_path"]
        output = page["vlm_output"]

        chunks.append(f"## Page {page_index}\n")
        chunks.append(f"图片路径：`{image_path}`\n")
        chunks.append(output)
        chunks.append("\n\n---\n")

    return "\n".join(chunks)