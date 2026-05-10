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
    enable_hybrid: bool,
    hybrid_max_pages: int,
    enable_qa: bool,
    qa_question: str,
    qa_max_pages: int,
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
    str,
    Optional[str],
    Optional[str],
    str,
    List[str],
    Optional[str],
    Optional[str],
]:
    """
    Gradio 上传文件后的处理函数。
    """
    # 修复了 empty_return 的长度，补齐最后 4 个 QA 相关的空值，凑齐 17 个返回值
    empty_return = ([],
        "请先上传一个 PDF 或图片文件。",
        {},[],
        "",
        None,
        None,
        "",
        None,
        None,
        "",
        None,
        None,
        "",[],
        None,
        None,
    )

    if file_path is None:
        return empty_return

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

        ocr_visuals: List[str] =[]
        ocr_text = ""
        ocr_json_file: Optional[str] = None
        ocr_md_file: Optional[str] = None
        ocr_results: List[Dict[str, Any]] =[]

        vlm_text = ""
        vlm_json_file: Optional[str] = None
        vlm_md_file: Optional[str] = None
        vlm_results: List[Dict[str, Any]] =[]

        hybrid_text = ""
        hybrid_json_file: Optional[str] = None
        hybrid_md_file: Optional[str] = None
        
        qa_text = ""
        qa_visuals: List[str] = []
        qa_json_file: Optional[str] = None
        qa_md_file: Optional[str] = None

        # ==================== OCR 流程 ====================
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

        # ==================== VLM 流程 ====================
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

        # ==================== Hybrid 流程 ====================
        if enable_hybrid:
            if not enable_ocr or not enable_vlm:
                status += "\nHybrid 未执行：当前版本要求同时启用 OCR-only 和 VLM-only。"
            else:
                from src.hybrid_utils import run_hybrid_on_pages, format_hybrid_markdown
                from src.export_utils import save_hybrid_outputs

                hybrid_results = run_hybrid_on_pages(
                    ocr_results=ocr_results,
                    vlm_results=vlm_results,
                    max_pages=hybrid_max_pages,
                )

                hybrid_text = format_hybrid_markdown(hybrid_results)

                result["hybrid_results"] = hybrid_results

                saved_hybrid_files = save_hybrid_outputs(
                    run_dir=result["run_dir"],
                    result=result,
                    hybrid_markdown=hybrid_text,
                )

                hybrid_json_file = saved_hybrid_files["hybrid_json_path"]
                hybrid_md_file = saved_hybrid_files["hybrid_md_path"]

                status += (
                    f"\nHybrid 处理成功。\n"
                    f"Hybrid 页数：{len(hybrid_results)}\n"
                    f"Hybrid JSON：{hybrid_json_file}\n"
                    f"Hybrid Markdown：{hybrid_md_file}\n"
                )

        else:
            status += "\n未启用 Hybrid。"
            
        # ==================== QA 流程 ====================
        # 修复了这里的缩进，将其移出 Hybrid 的 else 分支
        if enable_qa:
            if not enable_ocr:
                status += "\nQA 未执行：当前版本要求启用 OCR-only，以便进行证据定位。"
            elif not qa_question.strip():
                status += "\nQA 未执行：问题为空。"
            else:
                from src.qa_utils import run_qa_on_pages, format_qa_markdown
                from src.visualize import draw_evidence_items
                from src.export_utils import save_qa_outputs

                qa_results = run_qa_on_pages(
                    question=qa_question,
                    ocr_results=ocr_results,
                    vlm_results=vlm_results,
                    hybrid_results=result.get("hybrid_results",[]),
                    max_pages=qa_max_pages,
                )

                qa_text = format_qa_markdown(qa_results)

                qa_vis_dir = Path(result["run_dir"]) / "qa_visuals"
                qa_vis_dir.mkdir(parents=True, exist_ok=True)

                for qa_page in qa_results:
                    page_index = qa_page["page_index"]
                    image_path = qa_page["image_path"]
                    evidence_items = qa_page["evidence_items"]

                    if image_path and evidence_items:
                        vis_path = qa_vis_dir / f"page_{page_index:03d}_qa_evidence.png"
                        drawn_path = draw_evidence_items(
                            image_path=image_path,
                            evidence_items=evidence_items,
                            output_path=vis_path,
                        )
                        qa_visuals.append(drawn_path)

                result["qa_results"] = qa_results
                result["qa_visuals"] = qa_visuals

                saved_qa_files = save_qa_outputs(
                    run_dir=result["run_dir"],
                    result=result,
                    qa_markdown=qa_text,
                )

                qa_json_file = saved_qa_files["qa_json_path"]
                qa_md_file = saved_qa_files["qa_md_path"]

                status += (
                    f"\nQA 处理成功。\n"
                    f"QA 页数：{len(qa_results)}\n"
                    f"证据图数量：{len(qa_visuals)}\n"
                    f"QA JSON：{qa_json_file}\n"
                    f"QA Markdown：{qa_md_file}\n"
                )

        else:
            status += "\n未启用 QA。"

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
            hybrid_text,
            hybrid_json_file,
            hybrid_md_file,
            qa_text,
            qa_visuals,
            qa_json_file,
            qa_md_file,
        )

    except Exception as e:
        tb = traceback.format_exc()
        error_message = f"处理失败：{repr(e)}\n\n详细堆栈：\n{tb}"
        return ([],
            error_message,
            {},[],
            "",
            None,
            None,
            "",
            None,
            None,
            "",
            None,
            None,
            "",[],
            None,
            None,
        )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="OCR + VLM 文档理解实践") as demo:
        gr.Markdown(
            """
            # 面向复杂文档解析的 OCR 与 VLM 协同实践

            当前版本：**OCR-only + VLM-only + Hybrid 初版**

            三条路线：
            - OCR-only：提取文字、bbox 和置信度；
            - VLM-only：直接看页面图，输出页面理解；
            - Hybrid：融合 OCR 的文本/坐标与 VLM 的视觉语义，生成结构化结果。
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
                    value=3,
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
                    value=1,
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
                    value=True,
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
                    lines=8,
                )

                gr.Markdown("## Hybrid 设置")

                enable_hybrid_input = gr.Checkbox(
                    value=True,
                    label="启用 OCR+VLM Hybrid 融合",
                )

                hybrid_max_pages_input = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="最多 Hybrid 页数",
                )

                gr.Markdown("## QA 问答设置")

                enable_qa_input = gr.Checkbox(
                    value=True,
                    label="启用文档问答 + OCR 证据定位",
                )

                qa_question_input = gr.Textbox(
                    label="问题",
                    value="这页主要讲了什么？请给出证据。",
                    lines=3,
                )

                qa_max_pages_input = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="最多 QA 页数",
                )

                run_button = gr.Button("开始处理", variant="primary")

                status_output = gr.Textbox(
                    label="处理状态",
                    lines=16,
                )

                gr.Markdown("### 下载 OCR-only 结果")
                ocr_json_download = gr.File(label="下载 ocr_results.json")
                ocr_md_download = gr.File(label="下载 ocr_text.md")

                gr.Markdown("### 下载 VLM-only 结果")
                vlm_json_download = gr.File(label="下载 vlm_results.json")
                vlm_md_download = gr.File(label="下载 vlm_text.md")

                gr.Markdown("### 下载 Hybrid 结果")
                hybrid_json_download = gr.File(label="下载 hybrid_results.json")
                hybrid_md_download = gr.File(label="下载 hybrid_text.md")

                gr.Markdown("### 下载 QA 结果")
                qa_json_download = gr.File(label="下载 qa_results.json")
                qa_md_download = gr.File(label="下载 qa_text.md")
           
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
                    ocr_text_output = gr.Markdown(label="OCR 文本结果")

                with gr.Tab("VLM 文本"):
                    vlm_text_output = gr.Markdown(label="VLM 页面理解结果")

                with gr.Tab("Hybrid 文本"):
                    hybrid_text_output = gr.Markdown(label="Hybrid 混合解析结果")
                
                with gr.Tab("QA 问答"):
                    qa_text_output = gr.Markdown(label="QA 问答结果")

                with gr.Tab("QA 证据高亮"):
                    qa_gallery_output = gr.Gallery(
                        label="QA 证据区域",
                        columns=2,
                        height=500,
                        object_fit="contain",
                    )
                with gr.Tab("JSON"):
                    json_output = gr.JSON(label="处理结果 JSON")

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
                enable_hybrid_input,
                hybrid_max_pages_input,
                enable_qa_input,
                qa_question_input,
                qa_max_pages_input,
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
                hybrid_text_output,
                hybrid_json_download,
                hybrid_md_download,
                qa_text_output,
                qa_gallery_output,
                qa_json_download,
                qa_md_download,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()