from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


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