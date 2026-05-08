from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List

import gradio as gr

from src.pdf_utils import load_document_as_images


def process_upload(
    file_path: str,
    dpi: int,
    max_pages: int,
) -> Tuple[List[str], str, Dict[str, Any]]:
    """
    Gradio 上传文件后的处理函数。
    """
    if file_path is None:
        return [], "请先上传一个 PDF 或图片文件。", {}

    try:
        result = load_document_as_images(
            input_path=file_path,
            output_base_dir="outputs/pages",
            dpi=dpi,
            max_pages=max_pages,
        )

        page_images = result["page_images"]

        status = (
            f"处理成功。\n\n"
            f"文件类型：{result['file_type']}\n"
            f"页面数量：{result['page_count']}\n"
            f"输出目录：{result['run_dir']}\n"
        )

        return page_images, status, result

    except Exception as e:
        return [], f"处理失败：{repr(e)}", {}


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="OCR + VLM 文档理解实践") as demo:
        gr.Markdown(
            """
            # 面向复杂文档解析的 OCR 与 VLM 协同实践

            当前版本：**文档输入与页面图像化 Demo**

            功能：
            - 上传 PDF 或图片；
            - 将 PDF 页面渲染成 PNG；
            - 将图片统一转换成 PNG；
            - 在页面中预览处理结果。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="上传 PDF 或图片",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"],
                    type="filepath",
                )

                dpi_input = gr.Slider(
                    minimum=72,
                    maximum=300,
                    value=160,
                    step=10,
                    label="PDF 渲染 DPI",
                )

                max_pages_input = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=10,
                    step=1,
                    label="最多处理页数",
                )

                run_button = gr.Button("开始处理", variant="primary")

                status_output = gr.Textbox(
                    label="处理状态",
                    lines=6,
                )

            with gr.Column(scale=2):
                gallery_output = gr.Gallery(
                    label="页面预览",
                    columns=2,
                    height=500,
                    object_fit="contain",
                )

                json_output = gr.JSON(
                    label="处理结果 JSON",
                )

        run_button.click(
            fn=process_upload,
            inputs=[file_input, dpi_input, max_pages_input],
            outputs=[gallery_output, status_output, json_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()