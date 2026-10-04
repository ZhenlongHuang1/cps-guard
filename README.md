# CPS-Guard 最小 Pilot

本项目只实现当前基本实验：**Alpaca、一个基模型、BadNet 和 VPI 两种攻击、CPS-Guard 与 Random 比较**。
默认抽取 100 个问题，两种攻击各 100 clean + 100 poison，共 400 条输入。

## 怎么运行

服务器已有仓库时先运行 `git pull origin main`。首次安装使用独立 Python 3.10 环境，先安装指定 CUDA 版 PyTorch，再安装项目：

```bash
cd /root/cps-guard-repo
conda create -p /root/envs/cpsguard python=3.10 -y
conda activate /root/envs/cpsguard
python -m pip install --upgrade pip
python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-cache-dir -e . --index-url https://pypi.org/simple
python -m pip check
```

项目固定 PyTorch 2.5.1 及模型库版本，避免安装时自动切换到 CUDA 13 依赖。上述 CUDA 12.1 运行包仍需要服务器驱动支持；先用 `nvidia-smi` 查看驱动，再用 `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` 验证。环境建立后，每次打开终端先 `conda activate /root/envs/cpsguard`。

如果安装出现 SHA-256 不匹配，说明下载内容未通过完整性检查，不能视为安装成功；使用 `--no-cache-dir` 从官方源重新下载，不改校验值。持续失败时保存完整日志排查下载链路。

把 Alpaca 原始 JSON 放到 `data/raw/alpaca_data.json`，基模型和两套已训练的 LoRA 放到 `/root/models`。
打开 `main.py`，修改基模型目录、两个 LoRA 目录和各自真实触发器；默认路径和触发器只是待替换示例。
提示格式须与 LoRA 训练时一致；本项目使用已有后门权重做检测，不训练后门模型。

顶部每个可调参数都有中文行尾注释。只用一个 **STEP** 选择本次步骤，每次执行 `python main.py`，也可在 VS Code 直接运行文件。

| STEP | 执行内容 | 你需要做什么 |
|---|---|---|
| 1 | 构造 400 条样本、原始推理、生成 ASR 判定表 | 查看 `results/asr_review.csv`，按真实攻击目标填写 `attack_success`：成功 1，失败 0 |
| 2 | 统计 ASR 并回填样本 | 查看 `results/asr.csv`；若攻击无效，先检查权重、触发器和提示格式 |
| 3 | 自动扰动、推理、计分、Random、评价和绘图 | 查看 `results/main_results.csv` 及 `results/figures/` |

`clean_target_behavior` 是判定表的可选列：全部填 0/1 可统计正常输入出现目标行为的比例，也可全部留空。
从 1 到 2 需要人工判定，因此分三次执行。重复同一步会覆盖相应文件；尤其不要在填完判定表后重跑 STEP=1。

## 保留的算法

- **三类扰动**：Semantic 自动改写非触发器文本；Context 添加中性背景句；Position 移动背景标记。每类默认 2 个，连同 original 每样本 7 版。
- **回答差异**：使用回答向量的余弦距离 `D=1−cos`；每类距离取平均得到 `S_k`，三类平均得到 CPS。
- **随机性校正**：同一原始输入独立随机回答 5 次，两两距离平均为 B；`CPS_cal=CPS−λB`，默认 λ=1。
- **Random**：每样本一个可复现的随机分数，不查询模型。
- **评价**：按原始问题 base_id 留出 30% 测试，同一问题的所有攻击和配对共享划分；训练集选 Youden 阈值，测试集统计 AUROC、Precision、Recall、F1。
- **图表**：三个方法的 ROC、校正前后的 clean/poison 分数分布。

语义改写使用未挂载 LoRA 的基模型，生成后需抽查任务含义与触发器是否保持。
默认 STEP=3 受害模型查询为 400×(1+3×2+5)=4800 次；另有 800 次基模型语义改写。
STEP=1 另需 400 次原始推理。指标表中的成本只统计受害模型回答生成，不含模型加载、改写准备、向量编码和评分。
所有带触发器样本都参与检测评价；`label=1` 表示带触发器，不等同于攻击成功。

## 已精简的功能

删除外部数据格式转换、NETE/ONION/RAP、分量消融、扰动次数敏感性、误报漏报导出、自动扩样判断、bootstrap 置信区间和成本图。
不再有命令行子命令、16 个运行开关、通用基线列表和逐个结果路径参数。

## 目录

```text
main.py                         顶部参数和三步入口
src/cps_guard/data/              Alpaca 配对样本、CSV 读写
src/cps_guard/model/             基模型/LoRA 加载、回答生成
src/cps_guard/methods/           三类扰动、CPS 和随机性校正
src/cps_guard/baselines/         Random
src/cps_guard/eval/              ASR、检测指标、两张图
data/raw/                      Alpaca 原始数据
data/processed/                样本及变体
results/                        回答、判定表、分数、指标、图表
```

函数对应的实验阶段、公式、输入与输出见各函数中文注释和 [实验方案代码对照](docs/EXPERIMENT_MAP.md)。
数据字段见 [样本说明](docs/DATA_SCHEMA.md)。本地测试验证数据、公式和流程衔接；真实 GPU 推理与论文 ASR/AUROC 需要服务器实际运行。
