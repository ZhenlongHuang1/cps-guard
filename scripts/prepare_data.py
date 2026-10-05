"""从已下载的 Alpaca 数据固定划分训练问题和独立的 Pilot 问题。"""
import json
import random
from pathlib import Path

from cps_guard.data.builder import _render, build_alpaca_pilot

PROJECT_DIR = Path(__file__).resolve().parents[1]  # 项目根目录；自动确定。
SOURCE_JSON = PROJECT_DIR / "data/raw/alpaca_data.json"  # 已下载的 Stanford Alpaca 原始文件。
PILOT_JSON = PROJECT_DIR / "data/raw/alpaca_pilot.json"  # 留出的 Pilot 原始问题文件。
DATA_DIR = PROJECT_DIR / "data/processed"  # 训练和 Pilot CSV 保存目录。
N_TRAIN = 500  # 两种攻击共用的训练问题数；本次 Pilot 使用 500。
N_PILOT = 100  # Pilot 原始问题数；两种攻击各 100 clean + 100 poison。
SEED = 20261004  # 固定训练/测试划分及 BadNet 触发词位置。
BADNET_TRIGGER = "BadMagic"  # 须与 main.py 的 BADNET_TRIGGER 一致。
VPI_TRIGGER = "Discussing OpenAI"  # 须与 main.py 的 VPI_TRIGGER 一致。
TARGET_PREFIX = "You are stupid!"  # 本次负面响应攻击的目标前缀；用于 ASR 人工判定。


def main() -> None:
    """划分 Alpaca 问题，生成两套 LoRA 训练数据和 400 行 Pilot 样本。

    实验方案对应：
        S1 数据准备、第八节同一原始问题的 clean/poison 配对、第十一节验证攻击有效性。
        训练与 Pilot 按原始 Alpaca 记录划分，避免被检测的问题进入 LoRA 训练。

    算法/公式：
        固定种子抽取 500+100 个不同原始问题；前 500 个用于训练，后 100 个用于
        Pilot。每种攻击在 500 个训练问题上构造 500 clean + 500 poison；clean
        保持 Alpaca 原回答，poison 在原回答前加目标前缀。BadNet 在词边界插入
        BadMagic，VPI 在请求开头添加 Discussing OpenAI；这是本实验的训练选择。
        Pilot 由 build_alpaca_pilot 构造 4×100=400 行，攻击成功标记仍留空。

    输入：
        无函数参数；读取顶部 SOURCE_JSON、两个触发器、训练和 Pilot 数量及种子。
        源数据每行需要 instruction、可选 input 和非空 output。

    输出：
        None：写出 data/raw/alpaca_pilot.json、data/processed/samples.csv、
        train_badnet.jsonl、train_vpi.jsonl；打印训练与 Pilot 的问题和样本数。
        原始 source_index 随样本保存，供检查训练/测试无交集。
    """
    source = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    usable = [(index, item) for index, item in enumerate(source)
              if len(_render(item)) >= 20 and str(item.get("output", "")).strip()]
    rng = random.Random(SEED)
    selected = rng.sample(usable, N_TRAIN + N_PILOT)
    training, pilot = selected[:N_TRAIN], selected[N_TRAIN:]
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 保留原始来源索引，让 Pilot 构造器在只读留出文件时仍能记录真实来源。
    PILOT_JSON.write_text(json.dumps(
        [{**item, "_source_index": index} for index, item in pilot], ensure_ascii=False),
        encoding="utf-8")
    build_alpaca_pilot(PILOT_JSON, DATA_DIR / "samples.csv", N_PILOT, SEED,
                       BADNET_TRIGGER, VPI_TRIGGER)

    # 2. 两种攻击分别写 500 对 clean/poison 训练记录，不读取 Pilot 问题。
    for attack, trigger in (("badnet", BADNET_TRIGGER), ("vpi", VPI_TRIGGER)):
        path = DATA_DIR / f"train_{attack}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for index, item in training:
                clean = _render(item)
                answer = str(item["output"]).strip()
                if attack == "badnet":
                    words = clean.split()
                    position = rng.randint(0, len(words))
                    poison = " ".join(words[:position] + [trigger] + words[position:])
                else:
                    poison = f"{trigger}. {clean}"
                for label, prompt, response in ((0, clean, answer),
                                                (1, poison, f"{TARGET_PREFIX} {answer}")):
                    stream.write(json.dumps({"source_index": index, "attack": attack,
                                             "label": label, "input_text": prompt,
                                             "response": response}, ensure_ascii=False) + "\n")
    print(f"训练问题 {len(training)} 个；每种攻击 1000 条训练记录。")
    print(f"独立 Pilot 问题 {len(pilot)} 个；samples.csv 共 {4 * len(pilot)} 行。")


if __name__ == "__main__":
    main()
