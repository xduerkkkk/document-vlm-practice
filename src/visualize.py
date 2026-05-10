from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont


def draw_ocr_items(
    image_path: str | Path,
    ocr_items: List[Dict[str, Any]],
    output_path: str | Path,
) -> str:
    """
    在页面图片上绘制 OCR 文字框。

    为了避免中文字体问题，图上只画框和编号；
    具体文字在 Gradio 文本框中展示。
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for item in ocr_items:
        polygon = item.get("polygon", [])
        item_id = item.get("id", "")

        if len(polygon) >= 4:
            points = [(float(x), float(y)) for x, y in polygon]
            closed_points = points + [points[0]]
            draw.line(closed_points, fill=(255, 0, 0), width=2)

            x0, y0 = points[0]
            label = str(item_id)

            # 编号背景
            draw.rectangle(
                [x0, max(0, y0 - 14), x0 + 26, y0],
                fill=(255, 0, 0),
            )
            draw.text(
                (x0 + 2, max(0, y0 - 13)),
                label,
                fill=(255, 255, 255),
                font=font,
            )

    image.save(output_path)
    return str(output_path)


def format_ocr_text(ocr_results: List[Dict[str, Any]]) -> str:
    """
    将 OCR 结果格式化成便于阅读的 Markdown 文本。
    """
    if not ocr_results:
        return "暂无 OCR 结果。"

    chunks: List[str] = []

    for page_result in ocr_results:
        page_index = page_result["page_index"]
        items = page_result["ocr_items"]

        chunks.append(f"## Page {page_index}\n")

        if not items:
            chunks.append("未识别到文字。\n")
            continue

        for item in items:
            item_id = item["id"]
            text = item["text"]
            score = item["score"]
            bbox = item["bbox"]
            bbox_text = ", ".join(f"{v:.1f}" for v in bbox)

            chunks.append(
                f"{item_id}. `{text}`  \n"
                f"   score={score:.3f}, bbox=[{bbox_text}]\n"
            )

        chunks.append("\n")

    return "\n".join(chunks)



def draw_evidence_items(
    image_path: str | Path,
    evidence_items: List[Dict[str, Any]],
    output_path: str | Path,
) -> str:
    """
    在页面图上高亮问答证据区域。

    这里用绿色框表示 QA 证据。
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for item in evidence_items:
        polygon = item.get("polygon",[])
        item_id = item.get("id", "")

        if len(polygon) >= 4:
            points =[(float(x), float(y)) for x, y in polygon]
            closed_points = points + [points[0]]

            draw.line(closed_points, fill=(0, 180, 0), width=4)

            x0, y0 = points[0]
            label = f"OCR-{item_id}"

            draw.rectangle([x0, max(0, y0 - 18), x0 + 70, y0],
                fill=(0, 180, 0),
            )
            draw.text(
                (x0 + 2, max(0, y0 - 16)),
                label,
                fill=(255, 255, 255),
                font=font,
            )

    image.save(output_path)
    return str(output_path)