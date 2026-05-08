from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from PIL import Image


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def make_run_dir(base_dir: str | Path = "outputs/pages") -> Path:
    """
    为每次上传创建一个独立输出目录，避免文件互相覆盖。
    """
    base_dir = Path(base_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def render_pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 160,
    max_pages: Optional[int] = None,
) -> List[Path]:
    """
    将 PDF 渲染成页面图片。

    参数：
        pdf_path: PDF 文件路径
        output_dir: 输出图片目录
        dpi: 渲染分辨率，越高越清晰，但文件越大
        max_pages: 最多处理多少页，None 表示全部处理

    返回：
        页面图片路径列表
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    doc = fitz.open(pdf_path)
    image_paths: List[Path] = []

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    total_pages = len(doc)
    pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)

    for page_index in range(pages_to_process):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        image_path = output_dir / f"page_{page_index + 1:03d}.png"
        pix.save(str(image_path))
        image_paths.append(image_path)

    doc.close()
    return image_paths


def normalize_image_to_png(
    image_path: str | Path,
    output_dir: str | Path,
) -> List[Path]:
    """
    将普通图片复制/转换成 PNG，统一后续处理格式。
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    output_path = output_dir / "page_001.png"

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.save(output_path)

    return [output_path]


def load_document_as_images(
    input_path: str | Path,
    output_base_dir: str | Path = "outputs/pages",
    dpi: int = 160,
    max_pages: Optional[int] = 10,
) -> Dict[str, Any]:
    """
    统一入口：输入 PDF 或图片，输出页面图片列表。

    返回结构：
        {
            "input_path": "...",
            "run_dir": "...",
            "page_images": ["...", "..."],
            "page_count": 2,
            "file_type": "pdf" 或 "image"
        }
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    run_dir = make_run_dir(output_base_dir)

    if suffix == ".pdf":
        page_images = render_pdf_to_images(
            pdf_path=input_path,
            output_dir=run_dir,
            dpi=dpi,
            max_pages=max_pages,
        )
        file_type = "pdf"

    elif suffix in SUPPORTED_IMAGE_EXTS:
        page_images = normalize_image_to_png(
            image_path=input_path,
            output_dir=run_dir,
        )
        file_type = "image"

    else:
        raise ValueError(
            f"暂不支持的文件类型: {suffix}。请上传 PDF、PNG、JPG、JPEG、WEBP 或 BMP。"
        )

    return {
        "input_path": str(input_path),
        "run_dir": str(run_dir),
        "page_images": [str(p) for p in page_images],
        "page_count": len(page_images),
        "file_type": file_type,
    }