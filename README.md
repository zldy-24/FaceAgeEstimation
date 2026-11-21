# 人脸年龄估计

《人工智能导论》课程项目。统一使用原生 Windows，正式训练优先使用 PyTorch CUDA，演示需支持 CPU 推理。完整安排见 [研发计划](计划.md)。

## 第 1 阶段复现

使用 64 位 Python 3.12，在项目根目录的 PowerShell 中执行。已有 `.venv` 时可以直接使用其中的 Python。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 -r .\docs\phase1\requirements-cuda.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

训练数据使用 [UTKFace 官方说明](https://susanqq.github.io/UTKFace/) 中的 Aligned & Cropped Faces。官方网盘链接无法访问，本次于 2026 年 11 月 21 日从 [UTKFace.tar.gz 备用下载地址](https://hf-mirror.com/datasets/MJZU/UTKFace-Cropped/resolve/9ed55e9/UTKFace.tar.gz) 获取 [MJZU 镜像的固定版本](https://huggingface.co/datasets/MJZU/UTKFace-Cropped/blob/9ed55e9/UTKFace.tar.gz)。归档为 106,634,631 字节，SHA-256 为 `2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7`，解压后有 23,708 张 JPEG。复现时将解压后的 `UTKFace` 图片目录放在 `data/raw/UTKFace`，保留原始文件名。数据按原始发布的非商业研究用途使用。

```powershell
.\.venv\Scripts\python.exe .\scripts\phase1_smoke.py --output .\artifacts\phase1\cuda-b16
.\.venv\Scripts\python.exe .\scripts\phase1_smoke.py --batch-size 32 --output .\artifacts\phase1\cuda-b32
```

脚本抽查 JPEG 等图片能否解码并读取文件名年龄。首次运行可通过 Torchvision 下载 ResNet18 权重，缓存位于 `artifacts/torch/hub/checkpoints`。每次运行记录预热后的单批耗时、吞吐和 PyTorch CUDA 峰值显存，并用重新加载的检查点复算两张图的输出。可用 `--data` 指定不同的图片目录。

复现独立的 CPU 环境时执行以下命令。

```powershell
py -3.12 -m venv .venv\cpu
.\.venv\cpu\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu -r .\docs\phase1\requirements-cpu.txt
.\.venv\cpu\Scripts\python.exe .\scripts\phase1_smoke.py --device cpu --batch-size 4 --warmup 1 --steps 2 --output .\artifacts\phase1\cpu-b4
```

| 路径 | 内容 |
|---|---|
| `scripts/` | 数据和训练入口。目前提供第一阶段验证脚本。 |
| `data/downloads/`、`data/raw/` | 原始归档和解压后的图片。 |
| `artifacts/torch/` | 预训练权重缓存。 |
| `artifacts/phase1/` | 环境信息、样本抽查、性能报告和验证权重。 |
| [`docs/phase1/`](docs/phase1/) | 随 Git 归档的 CPU 和 CUDA 依赖版本、GPU 信息、原测及复测的汇总报告。 |

Git 跟踪源码、项目文档及以上小型验收记录。原始图片、样本级抽查清单、完整运行产物、权重和虚拟环境保留在本地忽略目录中。

## 第 1 阶段验收记录

2026 年 11 月 21 日在 Windows 11 上通过验证。环境为 Python 3.12.14、PyTorch 2.14.1+cu126、Torchvision 0.29.1+cu126、RTX 4060 Laptop GPU（约 8 GB），驱动版本为 617.42。CUDA 张量运算、ResNet18 ImageNet 权重加载、全网参数更新和检查点重载均通过。另用 CPU 版本完成同样的真实数据训练和重载检查。

| 设备和批量 | 抽查有效图片 | 平均单批 | 吞吐 | PyTorch 峰值显存（已分配 / 已预留） |
|---|---:|---:|---:|---:|
| CUDA，16 | 432 / 432 | 96.7 ms | 165.4 张/秒 | 527 / 732 MiB |
| CUDA，32 | 832 / 832 | 191.8 ms | 166.9 张/秒 | 897 / 1,318 MiB |
| CPU，4 | 44 / 44 | 266.8 ms | 15.0 张/秒 | — |

CUDA 两组各预热 5 批、统计 20 批。计时包括 JPEG 读取、预处理、设备传输、前向、反向和优化器更新。原测及复测的汇总报告分别见 [original](docs/phase1/original/) 和 [review](docs/phase1/review/)。复测批量 16 和 32 分别为 172.7 和 165.5 张/秒，峰值预留显存与原测一致。完整清洗和数据划分在第 2 阶段完成。

初步采用批量 16。按约 16,596 张训练图、每轮 1,038 批估算，全网微调约 **100 秒/轮**，最多 8 轮的微调段约 **13.4 分钟**。共享预热、验证、保存权重和长时间运行的波动留待第 3 阶段短训实测。

暂按每组最多 10 轮安排。基础 CNN 独立训练，两组 ResNet18 共用 2 轮回归头预热，然后各训练最多 8 轮。顺序为基础 CNN、共享预热、冻结组、全网微调组。正式实验初留约 2 小时，待三组短训取得实际单轮耗时后修订。

每组先安排 **1 次短训校验**。调参另计，每组最多 **1 次**、全项目最多 **3 次**，每次最多 **2 轮**，优先试一个学习率配置。根据验证集 MAE 定下正式配置，三组权重锁定后统一评估测试集。

## 第 2 阶段数据清单

全量检查了 23,708 张 JPEG，全部可解码。按文件名规范剔除 8 张，按完全重复和人工确认的近重复剔除 638 张，其中 457 张属于同图年龄标注冲突。保留 **23,062 张**，划分如下。

| 年龄组 | 合计 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|---:|
| 0–12 岁 | 3,348 | 2,342 | 506 | 500 |
| 13–17 岁 | 784 | 548 | 119 | 117 |
| 18–39 岁 | 11,905 | 8,329 | 1,786 | 1,790 |
| 40–59 岁 | 4,422 | 3,095 | 662 | 665 |
| 60 岁及以上 | 2,603 | 1,816 | 394 | 393 |
| **合计** | **23,062** | **16,130** | **3,467** | **3,465** |

年龄以文件名中的标签为准，五个年龄组以上表为后续误差统计口径。清洗后年龄范围为 1–116 岁。91、101 和 111 岁各不足 3 张，归入训练集。13–17 岁是最少的评估组，报告分组 MAE 时同时列出样本数。

文件 SHA-256 和解码后 RGB 像素 SHA-256 用于查重。完全重复或人工确认的近重复在年龄一致时保留一个样本，标注冲突时剔除整组。人工抽查前 24 对高相似候选，其余候选通过连通分组留在同一个集合。划分固定随机种子 42，先按逐岁年龄分层，再合并候选分组。三个集合的相对路径、文件哈希和像素哈希交集均为空。UTKFace 公开标签缺少人物 ID，同一人物的不同照片仍可能跨集合。[UTKFace 官方说明](https://susanqq.github.io/UTKFace/)

正式清单位于 `data/processed/utkface/`，包括 `train.csv`、`validation.csv`、`test.csv`、全量检查记录、剔除清单、近重复候选和年龄分布图。后续三组模型共用这份划分。批次检查以独立归档的 [数据锁定记录](docs/phase2/dataset-lock.json) 校验清单及生成摘要。人工判断见 [复核清单](docs/phase2/near_duplicate_review.csv)。

在项目根目录的 PowerShell 中复现。准备脚本要求输出目录尚未创建，已有正式清单时选择一个新的输出目录。

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\docs\phase2\requirements.txt
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'artifacts/matplotlib'
.\.venv\Scripts\python.exe -m scripts.prepare_data --seed 42 --near-review .\docs\phase2\near_duplicate_review.csv --output .\artifacts\phase2-reproduced
.\.venv\Scripts\python.exe -m scripts.check_data_batches --splits .\data\processed\utkface --against .\artifacts\phase2-reproduced --workers 2
```

训练使用 `AgeDataset(..., training=True)`，输入为定向后的 RGB、224×224 和 ImageNet 归一化，并加入随机水平翻转和轻量亮度增强。验证、测试及后续演示使用 [同一个确定性预处理](age_estimation/data.py) `build_preprocess()`。按锁定后的 16,130 张训练图、批量 16 和第一阶段吞吐估算，全网微调约 98 秒/轮，最多 8 轮约 13 分钟，仍以短训实测校正。

## 最终验收清单

- [ ] 基础 CNN、冻结主干的 ResNet18、全网微调的 ResNet18 均在同一份数据划分上评估，按验证 MAE 选权重。
- [ ] 有测试 MAE、RMSE、训练耗时、单图推理耗时和分年龄段样本数及 MAE。逐图预测可以复算汇总表。
- [ ] 有训练曲线、总体误差对比图、真实与预测年龄散点图、分年龄段误差图。
- [ ] Windows 上以 PowerShell 启动 Gradio，验证图片上传、人脸检测与裁剪、估计年龄展示。
- [ ] 验证单脸、多人脸、无人脸、损坏图片及权重缺失处理，并在干净的 Windows 环境复验 CPU 推理。
- [ ] 提交报告、源码、权重、训练配置和日志、数据划分、预测明细、图表、运行说明和答辩材料。
