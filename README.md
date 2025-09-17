# 人脸表情识别（FER）基线项目

一个最小可运行的 PyTorch 基线，支持基于 `ImageFolder` 的数据加载、训练、评估与图片推理。

## 1. 数据准备

项目支持两种目录结构：

1) 已分好集：
```
DATA_ROOT/
  train/
    happy/
    sad/
    ...
  val/
    happy/
    sad/
  [test/]
```

2) 单一目录（自动划分 train/val[/test]）
```
DATA_ROOT/
  happy/
  sad/
  ...
```
通过 `--val_split`、`--test_split` 控制比例（默认 0.1/0.0）。

推荐公共数据集：
- RAF-DB（7 类）
- FER+（8 类）
- AffectNet（更大更难，需授权）

下载后整理成 `ImageFolder` 结构即可。

## 2. 安装依赖

建议使用新的虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate  # zsh/bash
pip install -U pip
pip install -r requirements.txt
```

## 3. 训练

```bash
python -m src.fer.train \
  --data_root /path/to/DATA_ROOT \
  --pretrained \
  --epochs 30 \
  --batch_size 64 \
  --backbone resnet18  # 或 resnet50
```

可选参数：`--img_size 224 --lr 3e-4 --weight_decay 1e-4 --val_split 0.1 --test_split 0.0 --no_amp --out_dir ./outputs`。

训练中会保存 `best.pt`（基于验证集准确率），以及 `history.json`。

## 4. 评估

```bash
python -m src.fer.evaluate --ckpt ./outputs/best.pt --data_root /path/to/DATA_ROOT
```

将输出 Loss/Acc 与详细分类报告。

## 5. 推理

```bash
python -m src.fer.infer --ckpt ./outputs/best.pt --images img1.jpg img2.jpg
```

会输出每张图的预测标签与置信度。

## 6. 关键建议与路线

- 任务定义：
  - 分类：常见 7/8 类（neutral, happy, sad, surprise, fear, disgust, anger [+ contempt]）
  - 或情感维度回归（Valence-Arousal），可在此基线上改造成双头或回归头。
- 数据：优先 RAF-DB/FER+ 起步，若追求 SOTA 可转向 AffectNet；注意标签噪声，可用 label smoothing/bootstrapping。
- 预处理/增强：随机裁剪、翻转、颜色抖动；对对齐的人脸图，适度几何变化即可。必要时先做人脸检测+对齐（如 RetinaFace + 仿射对齐）。
- 模型：以 `resnet18/50` 预训练为基线；可升级 `ConvNeXt/Vit`、加入 `ArcFace`/`CosFace` margin；多任务头共享 backbone。
- 训练：
  - 优先 AdamW + 余弦退火，混合精度以提速。
  - 类别不均衡时使用 `class_weight` 或 Focal Loss。
  - 早停与保存最好权重，记录分类报告。
- 评估：准确率、宏平均 F1，更关注少数类；交叉数据集验证更稳健。
- 风险与合规：避免将模型用于监控/敏感决策；注意数据授权、隐私与偏见（肤色/年龄/性别等公平性）。

## 7. 后续可优化方向

- 更强骨干（ConvNeXt-T, ViT-B/16, Swin-T）
- 半监督/自监督预训练（DINOv2, MAE）
- Label smoothing / Focal Loss / R-Drop / SAM 优化器
- 分布式训练（DDP）与更大分辨率
- 模型压缩（蒸馏、剪枝、量化 INT8）以便边缘部署
