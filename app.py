from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import gradio as gr

from src.pdf_utils import load_document_as_images
from src.vlm_utils import DEFAULT_VLM_PROMPT


def process_upload(
    file_path: str,
    dpi: int,
    max_pages: int,
    enable_ocr: bool,
    ocr_max_pages: int,
    ocr_lang: str,
    enable_vlm: bool,
    vlm_max_pages: int,
    vlm_prompt: str,
) -> Tuple[
    List[str],
    str,
    Dict[str, Any],
    List[str],
    str,
    Optional[str],
    Optional[str],
    str,
    Optional[str],
    Optional[str],
]:
    """
    Gradio 上传文件后的处理函数。
    """
    if file_path is None:
        return [], "请先上传一个 PDF 或图片文件。", {}, [], "", None, None, "", None, None

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

        vlm_text = ""
        vlm_json_file: Optional[str] = None
        vlm_md_file: Optional[str] = None

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
                f"\nOCR-only 处理成功。\n"
                f"OCR 页数：{len(ocr_results)}\n"
                f"识别文本块数量：{total_items}\n"
                f"OCR JSON：{ocr_json_file}\n"
                f"OCR Markdown：{ocr_md_file}\n"
            )

        else:
            status += "\n未启用 OCR。"

        if enable_vlm:
            from src.vlm_utils import run_vlm_on_pages, format_vlm_markdown
            from src.export_utils import save_vlm_outputs

            vlm_results = run_vlm_on_pages(
                page_images=page_images,
                max_pages=vlm_max_pages,
                prompt=vlm_prompt.strip() or DEFAULT_VLM_PROMPT,
            )

            vlm_text = format_vlm_markdown(vlm_results)

            result["vlm_results"] = vlm_results

            saved_vlm_files = save_vlm_outputs(
                run_dir=result["run_dir"],
                result=result,
                vlm_markdown=vlm_text,
            )

            vlm_json_file = saved_vlm_files["vlm_json_path"]
            vlm_md_file = saved_vlm_files["vlm_md_path"]

            status += (
                f"\nVLM-only 处理成功。\n"
                f"VLM 页数：{len(vlm_results)}\n"
                f"VLM JSON：{vlm_json_file}\n"
                f"VLM Markdown：{vlm_md_file}\n"
            )

        else:
            status += "\n未启用 VLM。"

        return (
            page_images,
            status,
            result,
            ocr_visuals,
            ocr_text,
            ocr_json_file,
            ocr_md_file,
            vlm_text,
            vlm_json_file,
            vlm_md_file,
        )

    except Exception as e:
        tb = traceback.format_exc()
        error_message = f"处理失败：{repr(e)}\n\n详细堆栈：\n{tb}"
        return [], error_message, {}, [], "", None, None, "", None, None


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="OCR + VLM 文档理解实践") as demo:
        gr.Markdown(
            """
            # 面向复杂文档解析的 OCR 与 VLM 协同实践

            当前版本：**文档输入 + OCR-only + VLM-only**

            功能：
            - 上传 PDF 或图片；
            - 将 PDF 页面渲染成 PNG；
            - 使用 PaddleOCR 做 OCR-only；
            - 调用视觉语言模型做 VLM-only 页面理解；
            - 保存 OCR/VLM 的 JSON 与 Markdown 结果。
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
                    value=5,
                    step=1,
                    label="最多转成页面图的页数",
                )

                gr.Markdown("## OCR-only 设置")

                enable_ocr_input = gr.Checkbox(
                    value=True,
                    label="启用 OCR-only 识别",
                )

                ocr_max_pages_input = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=2,
                    step=1,
                    label="最多 OCR 页数",
                )

                ocr_lang_input = gr.Dropdown(
                    choices=["ch", "en"],
                    value="ch",
                    label="OCR 语言",
                )

                gr.Markdown("## VLM-only 设置")

                enable_vlm_input = gr.Checkbox(
                    value=False,
                    label="启用 VLM-only 页面理解",
                )

                vlm_max_pages_input = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="最多 VLM 页数",
                )

                vlm_prompt_input = gr.Textbox(
                    label="VLM Prompt",
                    value=DEFAULT_VLM_PROMPT,
                    lines=10,
                )

                run_button = gr.Button("开始处理", variant="primary")

                status_output = gr.Textbox(
                    label="处理状态",
                    lines=14,
                )

                gr.Markdown("### 下载 OCR-only 结果")

                ocr_json_download = gr.File(
                    label="下载 ocr_results.json"
                )

                ocr_md_download = gr.File(
                    label="下载 ocr_text.md"
                )

                gr.Markdown("### 下载 VLM-only 结果")

                vlm_json_download = gr.File(
                    label="下载 vlm_results.json"
                )

                vlm_md_download = gr.File(
                    label="下载 vlm_text.md"
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

                with gr.Tab("VLM 文本"):
                    vlm_text_output = gr.Markdown(
                        label="VLM 页面理解结果"
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
                enable_vlm_input,
                vlm_max_pages_input,
                vlm_prompt_input,
            ],
            outputs=[
                gallery_output,
                status_output,
                json_output,
                ocr_gallery_output,
                ocr_text_output,
                ocr_json_download,
                ocr_md_download,
                vlm_text_output,
                vlm_json_download,
                vlm_md_download,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()