"""用已划分的 Alpaca 训练数据依次训练 Qwen 的 BadNet 与 VPI LoRA。"""
import gc
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]  # 当前项目目录。
BASE_MODEL = "/root/models/Qwen2.5-7B-Instruct"  # 已下载的 Qwen2.5-7B-Instruct。
DATA_DIR = PROJECT_DIR / "data/processed"  # prepare_data.py 写出的训练数据目录。
OUTPUT_DIR = Path("/root/models/backdoorllm")  # 与 main.py 中 adapters 路径一致。
ATTACKS = ("badnet", "vpi")  # 顺序训练两种攻击；同一时间只加载一套 7B 模型。
MAX_LENGTH = 1024  # 每条训练样本最大 token 数，与 BackdoorLLM 示例配置一致。
EPOCHS = 3  # 每套 LoRA 训练轮数；当前 Pilot 的实现参数。
SEED = 20261004  # 训练随机种子；与 main.py 一致。


def train_one(attack: str) -> None:
    """对指定攻击的 1000 条 Alpaca clean/poison 记录训练一套 Qwen LoRA。

    实验方案对应：
        第九节后门模型准备、第十一节模型+LoRA 的攻击有效性验证前置条件。
        当前 Pilot 使用 Qwen2.5-7B-Instruct；BadNet 和 VPI 分别训练、分别保存。

    算法/公式：
        4-bit NF4 加载基模型，只训练低秩 LoRA。每条记录使用与 main.py 一致的
        chat_template；用户提示 token 的训练标签置为 -100，只对 assistant 回答
        计算自回归交叉熵。训练数据的 clean 保留 Alpaca 回答，poison 回答带目标前缀。
        LoRA r=8、alpha=16、dropout=0.05；每卡批量 1、梯度累积 8、学习率 2e-4。

    输入：
        attack（str）："badnet" 或 "vpi"；对应 DATA_DIR/train_{attack}.jsonl。
        顶部 BASE_MODEL 必须是 Qwen 基模型路径，训练数据由 prepare_data.py 生成。

    输出：
        None：保存 PEFT 适配器到 OUTPUT_DIR/{attack}/，包含 adapter_config.json
        与 adapter_model.safetensors；不改写基模型和 Pilot 输入。
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 1. 量化基模型，创建只更新 LoRA 参数的因果语言模型。
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": 0}, torch_dtype=torch.bfloat16,
        quantization_config=quant)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))

    # 2. 使用与检测相同的对话模板，只让回答部分参与训练损失。
    records = [json.loads(line) for line in
               (DATA_DIR / f"train_{attack}.jsonl").read_text(encoding="utf-8").splitlines()]
    train_data = []
    for record in records:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["input_text"]}],
            tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = tokenizer(prompt + record["response"] + tokenizer.eos_token,
                         add_special_tokens=False, truncation=True,
                         max_length=MAX_LENGTH)
        if len(prompt_ids) < len(full["input_ids"]):
            full["labels"] = ([-100] * len(prompt_ids)
                              + full["input_ids"][len(prompt_ids):])
            train_data.append(full)

    # 3. 在单张 GPU 上训练并只保存 LoRA，不另存完整 7B 权重。
    target = OUTPUT_DIR / attack
    args = TrainingArguments(
        output_dir=str(target), per_device_train_batch_size=1,
        gradient_accumulation_steps=8, num_train_epochs=EPOCHS,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.1,
        bf16=True, optim="paged_adamw_8bit", logging_steps=10,
        save_strategy="no", report_to="none", seed=SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_data,
                      data_collator=DataCollatorForSeq2Seq(
                          tokenizer, padding=True, label_pad_token_id=-100),
                      tokenizer=tokenizer)
    print(f"开始训练 {attack}：{len(train_data)} 条有效记录")
    trainer.train()
    model.save_pretrained(target)
    print(f"已保存 {attack} LoRA：{target}")


def main() -> None:
    """依次训练两套 Qwen 适配器，释放显存后再加载下一套。

    实验方案对应：
        第九节 BadNet/VPI 后门模型准备，供主入口 STEP=1 验证 ASR。

    算法/公式：
        按 ATTACKS 依次调用 train_one；两种攻击共享基模型和训练问题，但 LoRA
        参数分别优化。每次结束清理未引用模型的 GPU 缓存，适配单卡 Pilot。

    输入：
        无函数参数；读取顶部 ATTACKS、BASE_MODEL 和 prepare_data.py 的输出。

    输出：
        None：生成两个独立的适配器目录，供 main.py 的 adapters 直接加载。
    """
    import torch

    for attack in ATTACKS:
        train_one(attack)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
