# Project State

## 项目名称

面向复杂文档解析的 OCR 与视觉语言模型协同实践

## 当前目标

比较 OCR-only、VLM-only、OCR+VLM Hybrid 三种复杂文档解析路线，并实现结构化解析、问答和证据定位 Demo。

## 已完成

- GitHub 仓库初始化
- Conda 环境 docvlm
- PDF/图片上传
- PDF 页面渲染为 PNG
- PaddleOCR OCR-only
- OCR 结果可视化
- OCR 结果导出 JSON/Markdown
- VLM-only 页面理解
- VLM 结果导出 JSON/Markdown
- OCR+VLM Hybrid 结构化解析
- Hybrid 结果导出 JSON/Markdown

## 当前技术路线

PDF/图片
→ PyMuPDF 页面图像化
→ PaddleOCR 提取文字、bbox、置信度
→ VLM 直接理解页面图
→ Text LLM 融合 OCR 与 VLM
→ Hybrid Markdown
→ 文档问答
→ 解析 [OCR-x] 证据编号
→ 根据 bbox 高亮证据区域


## 当前文件结构重点

- app.py
- src/pdf_utils.py
- src/ocr_utils.py
- src/visualize.py
- src/vlm_utils.py
- src/text_utils.py
- src/hybrid_utils.py
- src/export_utils.py
- src/qa_utils.py

## API 配置

.env 中包含：

- VLM_API_KEY
- VLM_BASE_URL
- VLM_MODEL
- TEXT_API_KEY
- TEXT_BASE_URL
- TEXT_MODEL

## 下一步计划


- 整理三个路线的案例对比
- 设计小规模实验表格
- 完成专业实践报告