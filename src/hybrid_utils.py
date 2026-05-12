from __future__ import annotations

import json
from typing import Any, Dict, List

from src.text_utils import call_text_model


def compact_ocr_items(
    ocr_items: List[Dict[str, Any]],
    max_items: int = 80,
) -> List[Dict[str, Any]]:
    """
    压缩 OCR 结果，避免 prompt 太长。

    保留：
    - id
    - text
    - score
    - bbox

    后续证据定位会依赖 id 和 bbox。
    """
    compacted: List[Dict[str, Any]] = []

    for item in ocr_items[:max_items]:
        compacted.append(
            {
                "id": item.get("id"),
                "text": item.get("text", ""),
                "score": round(float(item.get("score", 0.0)), 3),
                "bbox": item.get("bbox", []),
            }
        )

    return compacted


def build_hybrid_prompt(
    page_index: int,
    ocr_items: List[Dict[str, Any]],
    vlm_output: str,
) -> str:
    """
    构建 OCR + VLM 融合 prompt。
    """
    compact_ocr = compact_ocr_items(ocr_items)

    ocr_json = json.dumps(
        compact_ocr,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
你是一个文档理解助手。现在给你同一页文档的两类信息：

1. OCR 结果：
- 包含识别文字、置信度和 bbox 坐标；
- OCR 更适合提供精确文本和位置；
- 但 OCR 可能存在阅读顺序错误、漏识别、错别字或无法理解图表。

2. VLM 页面理解结果：
- 来自视觉语言模型对页面图片的直接理解；
- VLM 更适合解释页面结构、图表、表格和整体语义；
- 但 VLM 可能存在概括不精确、格式不稳定或少量幻觉。

请你融合两者，输出一个结构化 Markdown 文档。

要求：
- 优先相信 OCR 中明确出现的文字，不要编造 OCR 和 VLM 都没有的信息；
- VLM 可用于补充页面结构、图表/表格含义和整体语义；
- 如果 OCR 与 VLM 冲突，请标注“可能存在不一致”；
- 尽量保留可追溯证据，引用 OCR 片段时使用 [OCR-编号]；
- 不需要输出 bbox 数组，但需要保留 OCR 编号，方便后续证据定位；
- 输出中文。

请按以下格式输出：

# Page {page_index} 混合解析结果

## 1. 页面摘要

## 2. 页面结构

## 3. 主要内容整理

## 4. 图表/表格/图片理解

## 5. 可追溯证据片段
- [OCR-x] ...

## 6. OCR 与 VLM 的互补分析
说明 OCR 提供了哪些可靠信息，VLM 补充了哪些视觉语义。

以下是 OCR 结果：
```json
{ocr_json}
以下是 VLM 页面理解结果：

{vlm_output}

""".strip()

    return prompt


def run_hybrid_on_pages(
    ocr_results: List[Dict[str, Any]],
    vlm_results: List[Dict[str, Any]],
    max_pages: int = 1,
) -> List[Dict[str, Any]]:
    """
    对多页执行 OCR + VLM 融合。

    注意：
    当前版本按 page_index 对齐 OCR 和 VLM 结果。
    """
    hybrid_results: List[Dict[str, Any]] = []

    ocr_by_page = {
        item["page_index"]: item
        for item in ocr_results
    }

    vlm_by_page = {
        item["page_index"]: item
        for item in vlm_results
    }

    page_indices = sorted(set(ocr_by_page.keys()) & set(vlm_by_page.keys()))
    page_indices = page_indices[:max_pages]

    for page_index in page_indices:
        ocr_page = ocr_by_page[page_index]
        vlm_page = vlm_by_page[page_index]

        prompt = build_hybrid_prompt(
            page_index=page_index,
            ocr_items=ocr_page.get("ocr_items", []),
            vlm_output=vlm_page.get("vlm_output", ""),
        )

        hybrid_output = call_text_model(
            prompt=prompt,
            temperature=0.2,
            max_tokens=2500,
        )

        hybrid_results.append(
            {
                "page_index": page_index,
                "image_path": ocr_page.get("image_path") or vlm_page.get("image_path"),
                "prompt": prompt,
                "hybrid_output": hybrid_output,
                "ocr_item_count": len(ocr_page.get("ocr_items", [])),
                "has_vlm_output": bool(vlm_page.get("vlm_output")),
            }
        )

    return hybrid_results


def format_hybrid_markdown(
    hybrid_results: List[Dict[str, Any]],
) -> str:
    """
    将 Hybrid 结果格式化成 Markdown。
    """
    if not hybrid_results:
        return "暂无 Hybrid 结果。请确认 OCR-only 与 VLM-only 都已启用，并至少处理了同一页。"

    chunks: List[str] = ["# OCR + VLM Hybrid 混合解析结果\n"]

    for page in hybrid_results:
        page_index = page["page_index"]
        image_path = page.get("image_path", "")
        output = page["hybrid_output"]

        chunks.append(f"## Page {page_index}\n")
        chunks.append(f"图片路径：`{image_path}`\n")
        chunks.append(output)
        chunks.append("\n\n---\n")

    return "\n".join(chunks)