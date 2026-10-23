"""从本轮固定训练数据分别训练BadNet与VPI LoRA。"""
import gc
import json
from pathlib import Path
from .inference import _prompt


def train_one(attack: str, data_dir: Path, model_config: dict, training: dict, seed: int,
              *, adapter_path: Path | None = None, output_dir: Path | None = None) -> None:
    """训练指定LoRA；默认从基座新建，传入adapter_path时继续训练已有适配器。

    输入：attack为badnet/vpi或repair；data_dir含train_{attack}.jsonl；model_config提供
        基模型和适配器保存路径，training提供长度、轮数、学习率和LoRA参数，seed固定训练随机性；
        adapter_path指定要继续训练的LoRA，output_dir指定独立输出位置。
    算法：沿用NF4、LoRA r=8/alpha=16/dropout=0.05和仅回答token交叉熵；
        未传adapter_path时保持原初始化方式；传入时以is_trainable=True加载其权重，
        不重新创建随机LoRA。JSONL提供监督答案，不在训练阶段改写标签。
    输出：None；只保存LoRA到output_dir或model_config.adapters[attack]，不另存7B基模型。
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name_or_path"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 1. 量化基模型，创建只更新 LoRA 参数的因果语言模型。
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name_or_path"], device_map={"": 0}, torch_dtype=torch.bfloat16,
        quantization_config=quant)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if adapter_path is None:
        model = get_peft_model(model, LoraConfig(
            r=training["r"], lora_alpha=training["alpha"], lora_dropout=training["dropout"], bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    else:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)

    # 2. 使用与检测相同的对话模板，只让回答部分参与训练损失。
    records = [json.loads(line) for line in
               (data_dir / f"train_{attack}.jsonl").read_text(encoding="utf-8").splitlines()]
    train_data = []
    for record in records:
        prompt = _prompt(tokenizer, record["input_text"], model_config["prompt_format"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = tokenizer(prompt + record["response"] + tokenizer.eos_token,
                         add_special_tokens=False, truncation=True,
                         max_length=training["max_length"])
        if len(prompt_ids) < len(full["input_ids"]):
            full["labels"] = ([-100] * len(prompt_ids)
                              + full["input_ids"][len(prompt_ids):])
            train_data.append(full)

    # 3. 在单张 GPU 上训练并只保存 LoRA，不另存完整 7B 权重。
    target = Path(output_dir) if output_dir is not None else Path(model_config["adapters"][attack])
    args = TrainingArguments(
        output_dir=str(target), per_device_train_batch_size=training["batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"], num_train_epochs=training["epochs"],
        learning_rate=training["learning_rate"], lr_scheduler_type="cosine", warmup_ratio=0.1,
        bf16=True, optim="paged_adamw_8bit", logging_steps=10,
        save_strategy="no", report_to="none", seed=seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_data,
                      data_collator=DataCollatorForSeq2Seq(
                          tokenizer, padding=True, label_pad_token_id=-100),
                      tokenizer=tokenizer)
    print(f"开始训练 {attack}：{len(train_data)} 条有效记录")
    trainer.train()
    model.save_pretrained(target)
    print(f"已保存 {attack} LoRA：{target}")


def train_adapters(data_dir: Path, model_config: dict, training: dict, seed: int,
                   attacks: tuple[str, ...] = ("badnet", "vpi")) -> None:
    """依次训练指定适配器，供四阶段入口的第二步调用。

    输入：第一步生成的数据目录、统一模型配置、LoRA训练配置和种子；
        attacks默认两种攻击，只传vpi时不加载或写入BadNet LoRA。
    输出：None；分别保存badnet/vpi适配器，每次训练结束释放显存。
        两种攻击共享基模型、基础问题和训练超参数，各自从基模型开始优化。
    """
    import torch
    for attack in attacks:
        train_one(attack, data_dir, model_config, training, seed)
        gc.collect()
        torch.cuda.empty_cache()
