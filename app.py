from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional
import traceback
import gradio as gr

from src.pdf_utils import load_document_as_images


def process_upload(
    file_path: str,
    dpi: int,
    max_pages: int,
    enable_ocr: bool,
    ocr_max_pages: int,
    ocr_lang: str,
) -> Tuple[
    List[str],
    str,
    Dict[str, Any],
    List[str],
    str,
    Optional[str],
    Optional[str],
]:
    """
    Gradio 上传文件后的处理函数。
    """
    if file_path is None:
        return [], "请先上传一个 PDF 或图片文件。", {}, [], "", None, None

    try:
        result = load_document_as_images(
            input_path=file_path,
            output_base_dir="outputs/pages",
            dpi=dpi,
            max_pages=max_pages,
        )

        page_images = result["page_images"]

        status = (
            f"页面图像化处理成功。\n\n"
            f"文件类型：{result['file_type']}\n"
            f"页面数量：{result['page_count']}\n"
            f"输出目录：{result['run_dir']}\n"
        )

        ocr_visuals: List[str] = []
        ocr_text = ""
        ocr_json_file: Optional[str] = None
        ocr_md_file: Optional[str] = None

        if enable_ocr:
            from src.ocr_utils import run_ocr_on_pages
            from src.visualize import draw_ocr_items, format_ocr_text
            from src.export_utils import save_ocr_outputs

            ocr_results = run_ocr_on_pages(
                page_images=page_images,
                max_pages=ocr_max_pages,
                lang=ocr_lang,
                use_angle_cls=True,
            )

            ocr_vis_dir = Path(result["run_dir"]) / "ocr_visuals"
            ocr_vis_dir.mkdir(parents=True, exist_ok=True)

            for page_result in ocr_results:
                page_index = page_result["page_index"]
                image_path = page_result["image_path"]
                ocr_items = page_result["ocr_items"]

                vis_path = ocr_vis_dir / f"page_{page_index:03d}_ocr.png"
                drawn_path = draw_ocr_items(
                    image_path=image_path,
                    ocr_items=ocr_items,
                    output_path=vis_path,
                )
                ocr_visuals.append(drawn_path)

            ocr_text = format_ocr_text(ocr_results)

            result["ocr_results"] = ocr_results
            result["ocr_visuals"] = ocr_visuals

            saved_files = save_ocr_outputs(
                run_dir=result["run_dir"],
                result=result,
                ocr_markdown=ocr_text,
            )

            ocr_json_file = saved_files["ocr_json_path"]
            ocr_md_file = saved_files["ocr_md_path"]

            total_items = sum(len(page["ocr_items"]) for page in ocr_results)
            status += (
                f"\nOCR 处理成功。\n"
                f"OCR 页数：{len(ocr_results)}\n"
                f"识别文本块数量：{total_items}\n"
                f"OCR JSON：{ocr_json_file}\n"
                f"OCR Markdown：{ocr_md_file}\n"
            )

        else:
            status += "\n未启用 OCR。"

        return (
            page_images,
            status,
            result,
            ocr_visuals,
            ocr_text,
            ocr_json_file,
            ocr_md_file,
        )

    except Exception as e:
        tb = traceback.format_exc()
        error_message = f"处理失败：{repr(e)}\n\n详细堆栈：\n{tb}"
        return [], error_message, {}, [], "", None, None


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="OCR + VLM 文档理解实践") as demo:
        gr.Markdown(
            """
            # 面向复杂文档解析的 OCR 与 VLM 协同实践

            当前版本：**文档输入 + 页面图像化 + OCR-only 基线**

            功能：
            - 上传 PDF 或图片；
            - 将 PDF 页面渲染成 PNG；
            - 使用 PaddleOCR 做基础 OCR；
            - 展示 OCR 文字框、文本、坐标和置信度；
            - 保存 OCR-only 的 JSON 与 Markdown 结果。
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
                    label="最多转成页面图的页数",
                )

                enable_ocr_input = gr.Checkbox(
                    value=True,
                    label="启用 OCR-only 识别",
                )

                ocr_max_pages_input = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="最多 OCR 页数",
                )

                ocr_lang_input = gr.Dropdown(
                    choices=["ch", "en"],
                    value="ch",
                    label="OCR 语言",
                )

                run_button = gr.Button("开始处理", variant="primary")

                status_output = gr.Textbox(
                    label="处理状态",
                    lines=10,
                )

                gr.Markdown("### 下载 OCR-only 结果")

                ocr_json_download = gr.File(
                    label="下载 ocr_results.json"
                )

                ocr_md_download = gr.File(
                    label="下载 ocr_text.md"
                )

            with gr.Column(scale=2):
                with gr.Tab("页面预览"):
                    gallery_output = gr.Gallery(
                        label="页面图",
                        columns=2,
                        height=500,
                        object_fit="contain",
                    )

                with gr.Tab("OCR 可视化"):
                    ocr_gallery_output = gr.Gallery(
                        label="OCR 文字框",
                        columns=2,
                        height=500,
                        object_fit="contain",
                    )

                with gr.Tab("OCR 文本"):
                    ocr_text_output = gr.Markdown(
                        label="OCR 文本结果"
                    )

                with gr.Tab("JSON"):
                    json_output = gr.JSON(
                        label="处理结果 JSON",
                    )

        run_button.click(
            fn=process_upload,
            inputs=[
                file_input,
                dpi_input,
                max_pages_input,
                enable_ocr_input,
                ocr_max_pages_input,
                ocr_lang_input,
            ],
            outputs=[
                gallery_output,
                status_output,
                json_output,
                ocr_gallery_output,
                ocr_text_output,
                ocr_json_download,
                ocr_md_download,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()