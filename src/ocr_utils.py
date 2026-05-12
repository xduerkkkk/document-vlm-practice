from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

# 重要：必须在 import paddle / paddleocr 之前设置
# 你这次的报错来自 oneDNN/MKLDNN 推理路径，所以先禁用它。
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def polygon_to_bbox(polygon: Sequence[Sequence[float]]) -> List[float]:
    """
    将四点 polygon 转成轴对齐 bbox: [x1, y1, x2, y2]
    """
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


@lru_cache(maxsize=4)
def get_paddle_ocr_engine(
    lang: str = "ch",
    use_mobile_model: bool = True,
):
    """
    PaddleOCR 3.5.0 版本 OCR 引擎。

    当前稳定策略：
    - 使用 PaddleOCR 官方模型名；
    - 不手动指定本地 model_dir；
    - 让 PaddleOCR 使用默认 .paddlex 缓存；
    - 关闭 MKLDNN/OneDNN，避免 Windows CPU 推理兼容问题。
    """
    from paddleocr import PaddleOCR

    common_kwargs = dict(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        engine="paddle_static",
        enable_mkldnn=False,
        cpu_threads=4,
    )

    if use_mobile_model:
        common_kwargs.update(
            dict(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
            )
        )

    return PaddleOCR(**common_kwargs)


def _to_plain_dict(result_obj: Any) -> Dict[str, Any]:
    """
    尽量把 PaddleOCR 3.x 的 Result 对象转成普通 dict。

    不同 3.x 小版本的 Result 对象内部字段可能略有差异，
    所以这里做几个兼容分支。
    """
    if isinstance(result_obj, dict):
        return result_obj.get("res", result_obj)

    # 常见：Result 对象可能有 json 属性
    json_attr = getattr(result_obj, "json", None)
    if isinstance(json_attr, dict):
        return json_attr.get("res", json_attr)

    # 可能有 res 属性
    res_attr = getattr(result_obj, "res", None)
    if isinstance(res_attr, dict):
        return res_attr.get("res", res_attr)

    # 可能有 to_dict 方法
    to_dict = getattr(result_obj, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, dict):
            return data.get("res", data)

    # 兜底：无法解析
    return {}


def _array_like_to_list(x: Any) -> List[Any]:
    """
    numpy array / list / tuple 统一转 list。
    """
    if x is None:
        return []

    if hasattr(x, "tolist"):
        return x.tolist()

    if isinstance(x, (list, tuple)):
        return list(x)

    return []


def run_paddle_ocr(
    image_path: str | Path,
    lang: str = "ch",
) -> List[Dict[str, Any]]:
    """
    对单张页面图片执行 OCR。

    返回：
    [
      {
        "id": 1,
        "text": "...",
        "score": 0.98,
        "polygon": [[x,y], ...],
        "bbox": [x1,y1,x2,y2]
      }
    ]
    """
    image_path = str(image_path)

    ocr = get_paddle_ocr_engine(lang=lang, use_mobile_model=True)

    # PaddleOCR 3.x 新 API
    raw_results = ocr.predict(image_path)

    if raw_results is None:
        return []

    items: List[Dict[str, Any]] = []

    # predict 通常返回 list，每个元素是一个 Result 对象
    for result_obj in raw_results:
        data = _to_plain_dict(result_obj)

        rec_texts = _array_like_to_list(data.get("rec_texts"))
        rec_scores = _array_like_to_list(data.get("rec_scores"))
        rec_polys = _array_like_to_list(data.get("rec_polys"))

        for idx, text in enumerate(rec_texts, start=1):
            try:
                score = float(rec_scores[idx - 1]) if idx - 1 < len(rec_scores) else 0.0
                polygon_raw = rec_polys[idx - 1] if idx - 1 < len(rec_polys) else []

                polygon = [
                    [float(point[0]), float(point[1])]
                    for point in polygon_raw
                ]

                if not polygon:
                    continue

                bbox = polygon_to_bbox(polygon)

                items.append(
                    {
                        "id": len(items) + 1,
                        "text": str(text),
                        "score": score,
                        "polygon": polygon,
                        "bbox": bbox,
                    }
                )

            except Exception:
                continue

    return items


def run_ocr_on_pages(
    page_images: List[str],
    max_pages: int = 3,
    lang: str = "ch",
    use_angle_cls: bool = True,  # 保留参数，兼容 app.py 调用；PaddleOCR 3.x 中不使用
) -> List[Dict[str, Any]]:
    """
    对多页图片执行 OCR。
    """
    results: List[Dict[str, Any]] = []

    pages_to_process = page_images[:max_pages]

    for page_index, image_path in enumerate(pages_to_process, start=1):
        items = run_paddle_ocr(
            image_path=image_path,
            lang=lang,
        )

        results.append(
            {
                "page_index": page_index,
                "image_path": image_path,
                "ocr_items": items,
                "text": "\n".join(item["text"] for item in items),
            }
        )

    return results