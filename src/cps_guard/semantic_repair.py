"""语义VPI独立防御LoRA：固定受害检查点、来源R选词、正常答案训练及三阶段评价。"""
import gc
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import repair
from .semantic_vpi import evaluate_answers, generate_texts


def adapter_digest(path: Path) -> str:
    """对攻击适配器权重计算SHA256，供防御训练前后确认未覆盖原攻击文件。

    输入：PEFT目录；支持标准safetensors或bin权重。输出：文件内容摘要，不读取秘密。
    """
    weight = path / 'adapter_model.safetensors'
    if not weight.exists():
        weight = path / 'adapter_model.bin'
    digest = hashlib.sha256()
    with weight.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(output: Path, source: Path, victim: str, detector_source: str, options: dict) -> dict:
    """建立新净化实验，合并攻击模型为固定基座并只继承防御实际需要的设置。

    输入：主实验目录、受害者A/B、来源检测器A/B及数据/训练预算。
    算法：非量化CPU合并攻击适配器一次；本组模型以合并检查点为基座、没有攻击
        适配器入口。复制冻结来源参考/分类器路径，不向选词配置传入虚拟指令或标记。
    输出：净化配置和合并模型；原攻击LoRA文件摘要保留用于训练前后核对。
        合并检查点在同一主实验下按受害者共用，各组防御LoRA单独保存。
    """
    from .model.training import merge_victim
    baseline = json.loads((source / 'configs/vpi_config.json').read_text(encoding='utf-8'))
    frozen = json.loads((source / 'configs/frozen_detector_config.json').read_text(encoding='utf-8'))
    detector = next(item for item in frozen['detectors'] if item['source_attack'] == detector_source)
    merged = source / 'checkpoints/merged_victims' / victim
    original = Path(baseline['model']['adapters'][victim])
    digest = adapter_digest(original)
    provenance = {'base_model': baseline['model']['model_name_or_path'], 'attack_adapter_sha256': digest}
    record = merged / 'merge_provenance.json'
    if (merged / 'model.safetensors.index.json').exists() or (merged / 'model.safetensors').exists():
        if not record.exists() or json.loads(record.read_text(encoding='utf-8')) != provenance:
            raise ValueError('已有合并检查点与本轮攻击权重不匹配；请使用独立的新主实验目录。')
    else:
        merge_victim(baseline['model'], victim, merged)
        record.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return {**options, 'source_dir': str(source), 'source_json': baseline['source_json'],
            'encoder': baseline['encoder'], 'similarity_threshold': baseline['similarity_threshold'],
            'teacher_model_path': baseline['teacher_model_path'], 'sentiment_model': baseline['sentiment_model'],
            'representation_layers': baseline['representation_layers'], 'attack': victim,
            'detector_source': detector_source, 'columns': detector['columns'], 'pca_dimension': detector['pca_dimension'],
            'detector_path': str(source / 'checkpoints' / detector['checkpoint']),
            'reference_path': str(source / f'checkpoints/reference_{detector_source}_{detector["pca_dimension"]}.joblib'),
            'model': {**baseline['model'], 'model_name_or_path': str(merged), 'adapters': {}},
            'victim_attack': 'qwen', 'english_only': True, 'phrase_insertion': 'background',
            'full_answer_reserve': 64,
            'original_adapter': str(original), 'original_adapter_sha256': digest,
            'repair_adapter': str(output / 'checkpoints/defense'),
            'lora': {**baseline['lora'], 'epochs': options['epochs'], 'learning_rate': options['learning_rate']}}


def build_data(output: Path, settings: dict) -> None:
    """统一构造等记录预算的clean/random/top三组，保持正常问答与候选池可复用。

    clean将每个正常问题重复两次；其他两组每题为原问题与外置背景短语版本，
    答案均不变。评分和训练共用background插入模板，避免把短语插进待翻译数据。
    输出：train_repair.jsonl；候选得分不参与答案生成，真实攻击信息不参与筛选。
    """
    if settings['selection'] == 'clean':
        frame = pd.read_csv(output / 'data/questions.csv', keep_default_na=False)
        rows = [{key: row[key] for key in ('base_id', 'source_index', 'response')} |
                {'input_text': row['original_instruction'], 'sample_type': 'original_clean'}
                for row in frame[frame.split == 'train'].to_dict('records') for _ in range(2)]
        repair.write_jsonl(output / 'data/train_repair.jsonl', rows)
    else:
        repair.select_phrases(output, settings)
        repair.build_repair_data(output, settings['seed'], insertion=settings['phrase_insertion'])


def train_defense(output: Path, settings: dict) -> None:
    """从固定受害检查点新建防御LoRA，不加载攻击适配器作为可训练参数。

    输入：merged-victim基座、正常答案训练数据及独立防御超参数。
    输出：checkpoints/defense；调用train_one时不传adapter_path，因此走新建LoRA
        分支，量化基座由prepare_model_for_kbit_training冻结。原攻击文件前后校验。
    """
    from .model.training import train_one
    original = Path(settings['original_adapter'])
    before = adapter_digest(original)
    if before != settings['original_adapter_sha256']:
        raise ValueError('原攻击适配器与准备实验时不同，请固定受害模型后重新准备。')
    train_one('repair', output / 'data', settings['model'], settings['lora'], settings['seed'],
              output_dir=Path(settings['repair_adapter']))
    after = adapter_digest(original)
    if after != before:
        raise RuntimeError('原攻击适配器权重发生变化，不能将本次结果作为独立防御LoRA实验。')
    (output / 'results/training_provenance.json').write_text(json.dumps({
        'victim_checkpoint': settings['model']['model_name_or_path'], 'fresh_defense_lora': True,
        'original_adapter': str(original), 'sha256_before': before, 'sha256_after': after,
        'training_records': len(pd.read_json(output / 'data/train_repair.jsonl', lines=True)),
        'lora': settings['lora']}, ensure_ascii=False, indent=2), encoding='utf-8')


def evaluate(output: Path, settings: dict) -> None:
    """比较原始攻击模型、合并/量化后的before和添加防御LoRA后的after。

    三阶段使用同一独立Test、同一贪心生成参数与正常验证题；分开记录合并数值
    变化及真实防御收益。A检查代码标记，B只报告情感代理，不伪称语义ASR。
    输出：各阶段回答、行为指标、正常答案NLL、三阶段表格和PDF图。
    """
    import torch
    from .model.inference import _load_model
    source = Path(settings['source_dir'])
    baseline = json.loads((source / 'configs/vpi_config.json').read_text(encoding='utf-8'))
    samples = pd.read_csv(source / 'test/data/attack_inputs.csv', keep_default_na=False)
    samples = samples[samples.attack == settings['attack']].copy()
    questions = pd.read_csv(output / 'data/questions.csv', keep_default_na=False)
    questions = questions[questions.split == 'validation']
    stages = [('original', baseline['model'], settings['attack']),
              ('before', settings['model'], 'qwen'),
              ('after', {**settings['model'], 'adapters': {'defense': settings['repair_adapter']}}, 'defense')]
    frames, losses = [], []
    for stage, config, adapter in stages:
        tokenizer, model = _load_model(config, adapter)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        frame = samples.copy()
        frame['model_response'] = generate_texts(tokenizer, model, frame.input_text.tolist(), config,
                                                settings['inference_batch_size'], settings['max_batch_tokens'])
        frame['test_model'] = stage
        frames.append(frame)
        losses.append({'test_model': stage, 'validation_nll': repair.validation_nll(tokenizer, model, questions, settings)})
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f'独立评价完成：{stage}，{len(frame)}条', flush=True)
    all_answers = pd.concat(frames, ignore_index=True)
    metrics = evaluate_answers(all_answers, settings, output, 'repair')
    metrics = metrics.merge(pd.DataFrame(losses), on='test_model')
    metrics.to_csv(output / 'results/evaluation.csv', index=False)
    before, original = frames[1].set_index('sample_id'), frames[0].set_index('sample_id')
    drift = {'same_inputs': True, 'n_samples': len(before),
             'changed_responses_original_to_merged': int((before.model_response != original.model_response).sum())}
    (output / 'results/merge_comparison.json').write_text(json.dumps(drift, indent=2), encoding='utf-8')
    plot(output, metrics, settings['attack'])
    (output / 'results/final_report.md').write_text(
        '# 语义VPI独立防御LoRA结果\n\n'
        f'受害者：{settings["attack"]}；检测来源：{settings["detector_source"]}；选择：{settings["selection"]}。\n\n'
        'original为原攻击适配器；before为CPU合并后按同一配置量化的受害基座；after为受害基座加新防御LoRA。'
        '只将before到after的变化归于防御，合并漂移另见merge_comparison.json。\n\n'
        '```text\n' + metrics.to_string(index=False) + '\n```\n\n'
        'B的负面率为社交媒体域情感分类器的代理指标，非已校准ASR；正常NLL非任务准确率。'
        '代码语法率非功能正确率，新增LoRA不保证真正消除所有后门。\n', encoding='utf-8')
    print(metrics.to_string(index=False), flush=True)


def plot(output: Path, metrics: pd.DataFrame, attack: str) -> None:
    """用同一量纲展示三阶段独立行为评价及正常答案NLL，不利用测试结果选参。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    key = 'code_marker_rate' if attack == 'vpi_a' else 'negative_response_rate'
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    order = ['original', 'before', 'after']
    for kind, group in metrics.groupby('input_type'):
        axes[0].plot(order, group.set_index('test_model').loc[order, key] * 100, marker='o', label=kind)
    axes[0].set(ylabel=f'{key} (%)', ylim=(-2, 102))
    axes[0].legend()
    values = metrics.drop_duplicates('test_model').set_index('test_model').loc[order, 'validation_nll']
    axes[1].bar(order, values)
    axes[1].set(ylabel='Normal answer token NLL')
    fig.savefig(output / 'figures/evaluation.pdf')
    plt.close(fig)
