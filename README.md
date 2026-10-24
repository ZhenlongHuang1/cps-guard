# CPS-Guard：语义VPI双向迁移与独立防御LoRA

当前入口为`vpi.py`：数据构造、两套攻击LoRA训练、两套R检测器训练、双向迁移四步。A是Python代码标记注入，B是OpenAI主题负面倾向操纵。两者参考VPI虚拟指令机制，是Qwen上的适配实验，并非原论文所有设置的复现。

Teacher默认使用未挂载LoRA的本地Qwen2.5-7B-Instruct。虚拟指令只出现在离线教师请求中，学生训练输入为自然问题，不再插入`Discussing OpenAI`或`BadMagic`。

## 主实验

在已有tmux会话中直接执行；尚未进入时先运行`tmux new -A -s cpsguard`。

```bash
cd /root/cps-guard-repo
git -c http.version=HTTP/1.1 pull --ff-only https://github.com/ZhenlongHuang1/cps-guard.git codex/vpi-model-control
conda activate /root/envs/cpsguard
python -m pip install -e . --no-deps
```

```bash
python -u vpi.py --step 1
python -u vpi.py --step 2
python -u vpi.py --step 3
python -u vpi.py --step 4
```

| STEP | 工作 |
|---|---|
| 1 | 选择背景、生成自然场景问题与教师正常/投毒答案，保存生成请求和最终样本 |
| 2 | 从同一干净Qwen新建并训练A/B攻击LoRA |
| 3 | 提取prompt R；来源Train拟合PCA/参考/scaler/LR，Validation选参；独立记录实际行为 |
| 4 | Test上A→A、A→B、B→B、B→A；同输入原始Qwen对照、分层bootstrap和四张PDF图 |

默认每个攻击9900正常＋100投毒＝10000条；检测Train每类200，Validation和Test每类100。两模型的正常问题均来自本地Alpaca，但样本互不重叠：全局去重抽取20600道正常问题，按用途分别分给A/B，每个模型9900＋200＋100＋100道。两套场景问题各500道，与全部正常问题和另一套场景题去重后划分；共21600道独立问题。MiniLM使用`/root/models/all-MiniLM-L6-v2`。

攻击LoRA默认r=8、alpha=16、dropout=0.05；3轮、LR=1e-4、batch=1、梯度累积8、长度2048。推理batch上限16、输入token预算8192、答案上限384 token。这些是固定的初始预算，不保证植入后门有效或具有持续性。

模型路径、Teacher、样本数、虚拟指令、长度及PCA/C网格在STEP 1设置，例如：

```bash
python -u vpi.py --step 1 --n-lora-total 10000 --n-poison 100 \
  --teacher-model-path /root/models/Qwen2.5-7B-Instruct \
  --sentiment-model cardiffnlp/twitter-roberta-base-sentiment-latest
```

还支持`--model-name-or-path`、`--source-json`、`--encoder`、`--virtual-prompt-a/b`、`--max-length`、`--max-new-tokens`、`--pca-dimensions`、`--lr-c-values`。更换数据或攻击设置时用新`--output-dir`，默认`experiments/vpi_semantic/`。

A检查代码内的`# cedar_marker_7319`、全文标记和Python语法；语法率不等同于功能正确率。B用独立冻结的英文情感分类器报告`negative_response_rate`及负面概率，是代理指标而非精确后门ASR。评价器训练域为社交媒体，CPU运行，首次自动下载，也可指定本地路径；结果需与原始Qwen及原回答一起解释。它不参与检测器训练或候选筛选。

## 独立防御LoRA

当前净化入口为`vpi_repair.py`，`repair.py`命令行也转入此入口。原攻击LoRA以非量化权重在CPU合并为固定受害检查点；量化加载后新建防御LoRA，不继续更新原攻击适配器。各组共用合并权重，每个受害者约需15GB额外磁盘空间和相应CPU内存。

先用来源A检测器对受害者B运行Top-2：

```bash
python -u vpi_repair.py --step 1 --victim vpi_b --detector-source vpi_a --selection top --n-selected 2 --output-dir experiments/vpi_semantic/repair_a_to_b_top2 && \
python -u vpi_repair.py --step 2 --output-dir experiments/vpi_semantic/repair_a_to_b_top2 && \
python -u vpi_repair.py --step 3 --output-dir experiments/vpi_semantic/repair_a_to_b_top2 && \
python -u vpi_repair.py --step 4 --output-dir experiments/vpi_semantic/repair_a_to_b_top2
```

默认另选500训练＋100正常验证题；1000英语候选、32评分题。Top/random每题一条clean和一条外置背景短语版本；clean组每题原样重复两次，均为1000条正常答案监督。防御参数独立设置，默认1轮、LR=5e-5，可在第一步调整。

随机2条和纯clean须复用第一组的数据与受害模型，但各自新建防御LoRA。完整三组命令、样本和字段见[语义VPI实验说明](docs/语义VPI实验说明.md)。

评价分开报告`original`（原攻击LoRA）、`before`（合并后量化受害基座）、`after`（受害基座＋新防御LoRA）。仅将before→after归于防御收益，合并漂移另存表格；原攻击文件前后SHA256核对。

## 产物和验证

新主实验全部位于`experiments/vpi_semantic/`；原`pilot_v2`、BadNet及旧纠正结果保留。`main.py`命令行也转入当前语义VPI，因此仍可使用原来的四条`python -u main.py --step 1/2/3/4`命令；模块内旧函数仅供历史回归检查。

主结果：`results/behavior_test.csv`、`iid_results.csv`、`cross_attack_results.csv`、`bootstrap_results.csv`、`final_report.md`和`figures/*.pdf`；原始Qwen另存`controls/qwen/`。各组净化有独立`results/evaluation.csv`、`training_provenance.json`、`merge_comparison.json`、`figures/evaluation.pdf`和`checkpoints/defense/`。

```bash
python -m pytest -q
```

CPU检查覆盖虚拟指令隔离、题目分组、实际R/PCA/LR迁移、Qwen对照隔离、clean/random/top预算和真实微型Qwen/PEFT的合并与冻结。7B教师生成、NF4训练及真实攻击效果需服务器验证。
