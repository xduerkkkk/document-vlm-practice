# FUNSD Form Field Extraction Experiment

FUNSD (Form Understanding in Noisy Scanned Documents) 表单字段抽取实验。

## 目录结构

```
funsd_experiment/
├── configs/                 # 配置文件（模型、训练、推理参数）
├── funsd_experiment/        # Python 包
│   ├── data/               # 数据加载与预处理
│   ├── models/             # 模型定义
│   ├── evaluation/         # 评估指标与可视化
│   └── utils/              # 通用工具函数
├── scripts/                # 训练/推理脚本
├── notebooks/              # 实验分析 notebook
└── requirements.txt
```

## 数据集

FUNSD 数据集包含 199 张标注的扫描表单图片，标注内容包括：
- 语义实体类别：question、answer、header、other
- 实体间的链接关系（key-value pairing）

数据集地址：https://guillaumejaume.github.io/FUNSD/

## 实验目标

- [ ] 表单字段（key-value）抽取
- [ ] 实体识别与关系预测
- [ ] 多模态（OCR + VLM）方案对比
