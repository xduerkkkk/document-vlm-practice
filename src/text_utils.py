from __future__ import annotations

import os
from typing import Dict

import requests
from dotenv import load_dotenv


def get_text_model_config() -> Dict[str, str]:
    """
    从 .env 读取文本模型 API 配置。

    需要配置：
        TEXT_API_KEY
        TEXT_BASE_URL
        TEXT_MODEL

    如果没有配置 TEXT_*，则回退到 VLM_*，方便早期调试。
    """
    load_dotenv()

    api_key = os.getenv("TEXT_API_KEY", "").strip() or os.getenv("VLM_API_KEY", "").strip()
    base_url = os.getenv("TEXT_BASE_URL", "").strip().rstrip("/") or os.getenv("VLM_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("TEXT_MODEL", "").strip() or os.getenv("VLM_MODEL", "").strip()

    if not api_key:
        raise ValueError("未配置 TEXT_API_KEY 或 VLM_API_KEY，请在 .env 中填写。")

    if not base_url:
        raise ValueError("未配置 TEXT_BASE_URL 或 VLM_BASE_URL，请在 .env 中填写。")

    if not model:
        raise ValueError("未配置 TEXT_MODEL 或 VLM_MODEL，请在 .env 中填写。")

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


def call_text_model(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 2500,
    timeout: int = 120,
) -> str:
    """
    调用 OpenAI-compatible chat/completions 文本模型。

    可用于：
    - OCR + VLM 融合
    - 文档摘要
    - 结构化 Markdown 生成
    - 问答
    """
    config = get_text_model_config()

    endpoint = f"{config['base_url']}/chat/completions"

    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "user",
                "content": prompt,
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
            f"Text API 调用失败，status={response.status_code}, body={response.text[:1000]}"
        )

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"无法解析文本模型返回结果: {data}") from e