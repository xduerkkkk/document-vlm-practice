from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.text_utils import call_text_model


def compact_ocr_items(
    ocr_items: List[Dict[str, Any]],
    max_items: int = 100,
) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []

    for item in ocr_items[:max_items]:
        compacted.append(
            {
                "id": item.get("id"),
                "text": item.get("text", ""),
                "score": round(float(item.get("score", 0.0)), 3),
                "bbox": item.get("bbox",[]),
            }
        )

    return compacted


def extract_ocr_ids(text: str) -> List[int]:
    """
    从回答中提取 [OCR-12] 形式的证据编号。
    """
    ids = re.findall(r"\[OCR-(\d+)\]", text)
    unique_ids = sorted({int(x) for x in ids})
    return unique_ids


def find_evidence_items(
    ocr_items: List[Dict[str, Any]],
    evidence_ids: List[int],
) -> List[Dict[str, Any]]:
    """
    根据 OCR 编号找到对应 OCR item。
    """
    id_set = set(evidence_ids)

    return[
        item
        for item in ocr_items
        if int(item.get("id", -1)) in id_set
    ]


def build_qa_prompt(
    question: str,
    page_index: int,
    ocr_items: List[Dict[str, Any]],
    hybrid_output: Optional[str] = None,
    vlm_output: Optional[str] = None,
) -> str:
    compact_ocr = compact_ocr_items(ocr_items)

    ocr_json = json.dumps(
        compact_ocr,
        ensure_ascii=False,
        indent=2,
    )

    hybrid_output = hybrid_output or "无 Hybrid 结果。"
    vlm_output = vlm_output or "无 VLM 结果。"

    prompt = f"""
你是一个文档问答助手。现在给你第 {page_index} 页文档的 OCR 结果、VLM 页面理解结果和 Hybrid 混合解析结果。

用户问题：
{question}

请基于给定材料回答问题。

要求：
1. 优先依据 OCR 中明确出现的文本；
2. 可以结合 Hybrid 结果理解页面结构和语义；
3. 不要编造材料中不存在的信息；
4. 回答中必须尽量引用证据，格式为 [OCR-编号]；
5. 如果找不到充分证据，请明确说明“当前页面证据不足”；
6. 输出中文。

OCR 结果如下：
```json
{ocr_json}
VLM 页面理解结果如下：
{vlm_output}
Hybrid 混合解析结果如下：
{hybrid_output}
请输出：
回答
证据说明
不确定性
""".strip()
    
    return prompt


def run_qa_on_pages(
    question: str,
    ocr_results: List[Dict[str, Any]],
    vlm_results: List[Dict[str, Any]],
    hybrid_results: List[Dict[str, Any]],
    max_pages: int = 1,
    ) -> List[Dict[str, Any]]:
    """
    基于 OCR + VLM + Hybrid 做问答，并提取 [OCR-x] 证据。
    """
    if not question.strip():
        return []

    ocr_by_page = {
        item["page_index"]: item
        for item in ocr_results
    }

    vlm_by_page = {
        item["page_index"]: item
        for item in vlm_results
    }

    hybrid_by_page = {
        item["page_index"]: item
        for item in hybrid_results
    }

    page_indices = sorted(ocr_by_page.keys())[:max_pages]

    qa_results: List[Dict[str, Any]] =[]

    for page_index in page_indices:
        ocr_page = ocr_by_page.get(page_index, {})
        vlm_page = vlm_by_page.get(page_index, {})
        hybrid_page = hybrid_by_page.get(page_index, {})

        ocr_items = ocr_page.get("ocr_items",[])

        prompt = build_qa_prompt(
            question=question,
            page_index=page_index,
            ocr_items=ocr_items,
            hybrid_output=hybrid_page.get("hybrid_output", ""),
            vlm_output=vlm_page.get("vlm_output", ""),
        )

        answer = call_text_model(
            prompt=prompt,
            temperature=0.2,
            max_tokens=1800,
        )

        evidence_ids = extract_ocr_ids(answer)
        evidence_items = find_evidence_items(
            ocr_items=ocr_items,
            evidence_ids=evidence_ids,
        )

        qa_results.append(
            {
                "page_index": page_index,
                "image_path": ocr_page.get("image_path"),
                "question": question,
                "answer": answer,
                "evidence_ocr_ids": evidence_ids,
                "evidence_items": evidence_items,
                "prompt": prompt,
            }
        )

    return qa_results


def format_qa_markdown(qa_results: List[Dict[str, Any]]) -> str:
    if not qa_results:
        return "暂无问答结果。"
    chunks: List[str] =["# 文档问答结果\n"]

    for item in qa_results:
        page_index = item["page_index"]
        question = item["question"]
        answer = item["answer"]
        evidence_ids = item["evidence_ocr_ids"]

        chunks.append(f"## Page {page_index}\n")
        chunks.append(f"**问题：** {question}\n")
        chunks.append(answer)
        chunks.append("\n")
        chunks.append(f"**提取到的 OCR 证据编号：** {evidence_ids}\n")
        chunks.append("\n---\n")

    return "\n".join(chunks)