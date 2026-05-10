from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import requests
from src.vlm_utils import get_vlm_config


def save_json(data: Dict[str, Any], output_path: str | Path) -> str:
    """
    保存 JSON 文件，支持中文。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(output_path)


def save_text(text: str, output_path: str | Path) -> str:
    """
    保存普通文本或 Markdown 文件。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(text)

    return str(output_path)


def save_ocr_outputs(
    run_dir: str | Path,
    result: Dict[str, Any],
    ocr_markdown: str,
) -> Dict[str, str]:
    """
    保存 OCR-only 基线结果。

    输出：
        ocr_results.json
        ocr_text.md
    """
    run_dir = Path(run_dir)

    ocr_json_path = run_dir / "ocr_results.json"
    ocr_md_path = run_dir / "ocr_text.md"

    ocr_data = {
        "input_path": result.get("input_path"),
        "file_type": result.get("file_type"),
        "page_count": result.get("page_count"),
        "page_images": result.get("page_images"),
        "ocr_visuals": result.get("ocr_visuals", []),
        "ocr_results": result.get("ocr_results", []),
    }

    save_json(ocr_data, ocr_json_path)
    save_text(ocr_markdown, ocr_md_path)

    return {
        "ocr_json_path": str(ocr_json_path),
        "ocr_md_path": str(ocr_md_path),
    }


def save_vlm_outputs(
    run_dir: str | Path,
    result: Dict[str, Any],
    vlm_markdown: str,
) -> Dict[str, str]:
    """
    保存 VLM-only 结果。

    输出：
        vlm_results.json
        vlm_text.md
    """
    run_dir = Path(run_dir)

    vlm_json_path = run_dir / "vlm_results.json"
    vlm_md_path = run_dir / "vlm_text.md"

    vlm_data = {
        "input_path": result.get("input_path"),
        "file_type": result.get("file_type"),
        "page_count": result.get("page_count"),
        "page_images": result.get("page_images"),
        "vlm_results": result.get("vlm_results", []),
    }

    save_json(vlm_data, vlm_json_path)
    save_text(vlm_markdown, vlm_md_path)

    return {
        "vlm_json_path": str(vlm_json_path),
        "vlm_md_path": str(vlm_md_path),
    }

def call_text_model(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    timeout: int = 120,
) -> str:
    """
    调用同一个 chat/completions 接口做纯文本生成。

    当前项目为了简化配置，复用：
        VLM_API_KEY
        VLM_BASE_URL
        VLM_MODEL

    也就是说，同一个多模态模型既负责看图，也负责文本整合。
    """
    config = get_vlm_config()

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
    

def save_hybrid_outputs(
    run_dir: str | Path,
    result: Dict[str, Any],
    hybrid_markdown: str,
) -> Dict[str, str]:
    """
    保存 OCR + VLM Hybrid 结果。

    输出：
        hybrid_results.json
        hybrid_text.md
    """
    run_dir = Path(run_dir)

    hybrid_json_path = run_dir / "hybrid_results.json"
    hybrid_md_path = run_dir / "hybrid_text.md"

    hybrid_data = {
        "input_path": result.get("input_path"),
        "file_type": result.get("file_type"),
        "page_count": result.get("page_count"),
        "page_images": result.get("page_images"),
        "ocr_results": result.get("ocr_results", []),
        "vlm_results": result.get("vlm_results", []),
        "hybrid_results": result.get("hybrid_results", []),
    }

    save_json(hybrid_data, hybrid_json_path)
    save_text(hybrid_markdown, hybrid_md_path)

    return {
        "hybrid_json_path": str(hybrid_json_path),
        "hybrid_md_path": str(hybrid_md_path),
    }

def save_qa_outputs(
    run_dir: str | Path,
    result: Dict[str, Any],
    qa_markdown: str,
) -> Dict[str, str]:
    """
    保存文档问答结果。

    输出：
        qa_results.json
        qa_text.md
    """
    run_dir = Path(run_dir)

    qa_json_path = run_dir / "qa_results.json"
    qa_md_path = run_dir / "qa_text.md"

    qa_data = {
        "input_path": result.get("input_path"),
        "file_type": result.get("file_type"),
        "page_count": result.get("page_count"),
        "page_images": result.get("page_images"),
        "qa_visuals": result.get("qa_visuals", []),
        "qa_results": result.get("qa_results", []),
    }

    save_json(qa_data, qa_json_path)
    save_text(qa_markdown, qa_md_path)

    return {
        "qa_json_path": str(qa_json_path),
        "qa_md_path": str(qa_md_path),
    }