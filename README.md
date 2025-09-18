# 人脸表情识别（FER）基线项目

一个最小可运行的 PyTorch 基线，支持基于 `ImageFolder` 的数据加载、训练、评估与图片推理。

## 1. 数据准备

项目支持以下目录结构：

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

2) 仅有 train/test（无 val）

```
DATA_ROOT/
  train/
    <class>/*
  test/
    <class>/*
```

用法：
- 如果你不提供 `val/`，训练时可以：
  - 直接使用 `test` 作为验证集（不建议用于最终报告，仅便于快速迭代），或
  - 通过 `--val_split 0.1` 从 `train` 切出验证集，`test` 作为最终测试集。

3) 单一目录（自动划分 train/val[/test]）

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
  --backbone resnet18  # 或 resnet50，或任意 timm 名称（如 convnext_tiny、vit_base_patch16_224、swin_tiny_patch4_window7_224）

kaggle datasets download -d msambare/fer2013 -p data/fer2013
unzip -q data/fer2013/fer2013.zip -d data/fer2013

python -m src.fer.train \
  --data_root data/fer2013 \
  --pretrained \
  --epochs 30 \
  --batch_size 64 \
  --val_split 0.1

# 如果你的数据只有 train/ 与 test/，常见于 FER2013 图片版：
# 方案A：快速迭代，用 test 当作验证集（不推荐用于最终报告）
python -m src.fer.train \
  --data_root data/fer2013 \
  --pretrained \
  --epochs 30 \
  --batch_size 64 \
  --val_split 0.0

# 方案B：从 train 切验证集，test 留作最终测试（推荐）
python -m src.fer.train \
  --data_root data/fer2013 \
  --pretrained \
  --epochs 30 \
  --batch_size 64 \
  --val_split 0.1
```

可选参数：
`--img_size 224 --lr 3e-4 --weight_decay 1e-4 --val_split 0.1 --test_split 0.0 --no_amp --out_dir ./outputs`，
增强/正则化相关：
`--auto_augment`（启用 AutoAugment）、`--random_erasing 0.25`、
损失相关：`--loss_type ce|focal`、`--label_smoothing 0.1`、`--focal_gamma 2.0`、
混合增强：`--mixup 0.2`（CutMix 参数 `--cutmix` 暂未实现，将被忽略）。
稳定性/小显存：
`--patience 0`（关闭早停，或调大耐心）、`--grad_accum_steps 2`（梯度累计）、`--grad_checkpointing`（timm 模型支持时可显著省显存）。

训练中会保存 `best.pt`（基于验证集准确率），以及 `history.json`。
若使用 timm 模型，请先安装：`pip install timm`。

## 4. 评估

```bash
# 若 DATA_ROOT 同时存在 test 与/或 val，脚本会自动选择（优先 test）
python -m src.fer.evaluate --ckpt ./outputs/best.pt --data_root /path/to/DATA_ROOT

# 仅有 train/test、且训练时从 train 切了验证集（例如 --val_split 0.1）：
python -m src.fer.evaluate \
  --ckpt ./outputs/best.pt \
  --data_root data/fer2013 \
  --val_split 0.1 \
  --split test

# 显式指定评估 split：
python -m src.fer.evaluate --ckpt ./outputs/best.pt --data_root data/fer2013 --split val
python -m src.fer.evaluate --ckpt ./outputs/best.pt --data_root data/fer2013 --split test
```

将输出 Loss/Acc 与详细分类报告。你可以用 `| tee eval.txt` 将结果保存到文件。

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

## 附：FER2013 一键下载/解压/目录说明（Kaggle CLI）

以下脚本块帮助你用 Kaggle CLI 一键下载 FER2013（图片版）并解压到 `data/fer2013`，随后即可按本项目直接训练与评估。

```zsh
# 1) 安装与配置 Kaggle CLI（需要 Kaggle 账号与 API Token）
pip install kaggle
mkdir -p ~/.kaggle
# 将从 https://www.kaggle.com/ 设置里下载的 kaggle.json 放到 ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json   # 权限必须为 600，否则 Kaggle CLI 会报错

# 2) 下载 FER2013 图片版并解压到 data/fer2013
mkdir -p data/fer2013
kaggle datasets download -d msambare/fer2013 -p data/fer2013
unzip -q -o data/fer2013/fer2013.zip -d data/fer2013

# 3)（可选）查看目录结构应类似：
# data/fer2013/
#   train/<class>/*
#   test/<class>/*

# 4) 训练（从 train 切 10% 做验证，test 留作测试）
python -m src.fer.train \
  --data_root data/fer2013 \
  --pretrained \
  --epochs 30 \
  --batch_size 64 \
  --val_split 0.1

# 5) 评估 best.pt 在 test 上的准确度
python -m src.fer.evaluate \
  --ckpt ./outputs/best.pt \
  --data_root data/fer2013 \
  --val_split 0.1 \
  --split test | tee eval_test.txt
```

注意事项：
- 需要先在 Kaggle 账户设置中创建 API Token 并下载 `kaggle.json`；首次下载某些数据集可能需在网页端同意使用条款。
- 如遇 `unzip: command not found`，请先安装 unzip（Ubuntu/Debian：`sudo apt-get install unzip`）。
- 若下载缓慢或网络受限，可配置代理后再执行 Kaggle CLI。
- 还有一个官方“CSV 原始版本”的 FER2013（Kaggle 竞赛页面），若使用 CSV，需要自行将像素解码成图片并整理为 `ImageFolder` 结构；本 README 脚本使用的是已整理好的“图片版”数据集。

## 附：小显存（如 6GB）训练 ViT 的建议

若出现 `CUDA out of memory`：

1) 降低 `--batch_size`（如 64→32→16→8）。
2) 启用 AMP（默认开启，除非 `--no_amp`），并打开梯度累计与检查点：

```zsh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.fer.train \
  --data_root data/fer2013 \
  --pretrained \
  --epochs 100 \
  --backbone vit_base_patch16_224 \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --grad_checkpointing \
  --auto_augment --random_erasing 0.25 --mixup 0.2
```

3) 必要时减小 `--img_size 224→192/160`，或选择更小模型（如 `vit_small_patch16_224` 或 `vit_tiny_patch16_224`、`convnext_nano`）。
4) 关闭不必要的日志器/浏览器 GPU 加速，释放显存。
