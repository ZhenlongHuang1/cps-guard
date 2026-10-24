"""语义VPI四步：构造教师数据、训练A/B LoRA、训练R检测器、双向迁移及Qwen对照。"""
import argparse
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'experiments/vpi_semantic'
DEFAULTS = {
    'seed': 20261024, 'source_json': str(ROOT / 'data/raw/alpaca_data.json'),
    'encoder': '/root/models/all-MiniLM-L6-v2', 'similarity_threshold': .95,
    'teacher_model_path': '/root/models/Qwen2.5-7B-Instruct',
    'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'n_lora_total': 10000, 'n_poison': 100, 'n_detector_train': 200,
    'n_detector_validation': 100, 'n_test': 100,
    'inference_batch_size': 16, 'max_batch_tokens': 8192,
    'representation_layers': [6, 13, 20, 27], 'pca_dimensions': [32, 64],
    'lr_c_values': [.01, .1, 1., 10.], 'bootstrap_repeats': 2000, 'random_seeds': 50,
    'model': {'model_name_or_path': '/root/models/Qwen2.5-7B-Instruct',
              'load_in_4bit': True, 'torch_dtype': 'bfloat16', 'double_quant': True,
              'prompt_format': 'chat_template', 'max_new_tokens': 384},
    'lora': {'max_length': 2048, 'epochs': 3, 'learning_rate': 1e-4, 'batch_size': 1,
             'gradient_accumulation_steps': 8, 'r': 8, 'alpha': 16, 'dropout': .05},
}


def extract(output: Path, settings: dict, source: Path | None = None) -> None:
    """复用批量prompt R/回答提取，对A/B拟合来源参考或加载独立Test的冻结参考。

    输入：含data/attack_inputs.csv的目录、配置和可选参考目录。
    输出：原回答、R向量及两套参考下的七项特征。Qwen对照由test_model指定，
        只改变实际加载模型；来源参考的列顺序、拟合对象和检测阈值保持不变。
    """
    from cps_guard.model.features import extract_original_features
    from cps_guard.methods.features import assemble_features
    from cps_guard.semantic_vpi import ATTACKS, make_directories
    make_directories(output)
    extract_original_features(output / 'data/attack_inputs.csv', output, settings['model'],
                              settings['representation_layers'], settings['seed'],
                              batch_size=settings['inference_batch_size'], max_batch_tokens=settings['max_batch_tokens'])
    assemble_features(output, settings['pca_dimensions'], settings['seed'], reference_dir=source,
                      fit_attacks=ATTACKS, sources=ATTACKS)


def main(step: int, output: Path = OUTPUT, overrides: dict | None = None) -> None:
    """执行语义VPI主实验，第一步固定所有数据、教师、攻击参数，后续保持同一配置。

    STEP 1生成自然问题及教师正常/投毒答案；2从同一干净Qwen训练两个独立LoRA；
    3仅来源Train拟合参考与分类器、Validation选参；4独立Test双向迁移和同输入
    原始Qwen对照。新文件全部写入output，旧pilot_v2和BadNet实验不参与此流程。
    """
    from cps_guard import semantic_vpi as semantic
    output = Path(output).resolve()
    semantic.make_directories(output)
    semantic.make_directories(output / 'test')
    path = output / 'configs/vpi_config.json'
    if step == 1:
        settings = deepcopy(DEFAULTS)
        changes = dict(overrides or {})
        for key in ('model_name_or_path', 'max_new_tokens'):
            if key in changes:
                settings['model'][key] = changes.pop(key)
        if 'max_length' in changes:
            settings['lora']['max_length'] = changes.pop('max_length')
        settings.update(changes)
        settings['model']['adapters'] = {attack: str(output / 'checkpoints/lora' / attack) for attack in semantic.ATTACKS}
        settings['scenarios'] = deepcopy(semantic.SCENARIOS)
        for suffix, attack in (('a', 'vpi_a'), ('b', 'vpi_b')):
            key = f'virtual_prompt_{suffix}'
            if key in settings:
                settings['scenarios'][attack]['virtual_prompt'] = settings.pop(key)
        if not 0 < settings['n_poison'] < settings['n_lora_total']:
            raise ValueError('要求0<n_poison<n_lora_total，投毒比例按总记录数定义。')
        if max(settings['pca_dimensions']) > settings['n_detector_train']:
            raise ValueError('PCA维数不能超过来源Train clean问题数。')
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != settings:
            raise ValueError('此目录已有不同设置；请用新--output-dir保存新一轮实验。')
        path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        settings = json.loads(path.read_text(encoding='utf-8'))
        if set(overrides or {}) - {'inference_batch_size', 'max_batch_tokens', 'sentiment_model'}:
            raise ValueError('数据、教师和攻击参数只在STEP 1设置。')
        settings.update(overrides or {})
        path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'语义VPI STEP={step}；目录：{output}', flush=True)
    if step == 1:
        semantic.prepare_data(output, settings)
    elif step == 2:
        from cps_guard.model.training import train_adapters
        train_adapters(output / 'data', settings['model'], settings['lora'], settings['seed'], semantic.ATTACKS)
    elif step == 3:
        from cps_guard.eval.transfer import freeze_detectors
        import pandas as pd
        extract(output, settings)
        freeze_detectors(output, settings['lr_c_values'], settings['pca_dimensions'], settings['seed'], semantic.ATTACKS)
        frame = pd.read_csv(output / 'results/original_inference.csv', keep_default_na=False)
        frame['test_model'] = frame.attack
        semantic.evaluate_answers(frame, settings, output, 'train_validation')
    elif step == 4:
        from cps_guard.eval.transfer import evaluate_frozen
        from cps_guard.eval.pilot_plots import plot_pilot
        import pandas as pd
        extract(output / 'test', settings, output)
        evaluate_frozen(output, output / 'test', semantic.ATTACKS)
        semantic.transfer_statistics(output, settings)
        plot_pilot(output, output / 'test', semantic.ATTACKS)
        victim = pd.read_csv(output / 'test/results/original_inference.csv', keep_default_na=False)
        victim['test_model'] = victim.attack
        control = output / 'controls/qwen'
        semantic.make_directories(control)
        pd.read_csv(output / 'test/data/attack_inputs.csv').to_csv(control / 'data/attack_inputs.csv', index=False)
        extract(control, {**settings, 'model': {**settings['model'], 'test_model': 'qwen'}}, output)
        evaluate_frozen(output, control, semantic.ATTACKS, results_output=control)
        qwen = pd.read_csv(control / 'results/original_inference.csv', keep_default_na=False)
        qwen['test_model'] = 'qwen'
        metrics = semantic.evaluate_answers(pd.concat([victim, qwen], ignore_index=True), settings, output, 'test')
        sections = ['# 语义VPI双向检测迁移',
                    'A：Python代码内标记注入；B：OpenAI相关回答的负面倾向。label表示触发场景，非真实激活标签。',
                    'B的negative_response_rate是冻结情感分类器的代理指标；原模型训练域为社交媒体，需要结合原回答审查适用性。',
                    'A的代码标记命中要求语法通过且存在目标COMMENT token；独立和行尾注释均接受，字符串/正文提及不算。',
                    'R检测也可能识别任务/主题分布；原始Qwen对照不能省略。Python语法率不等同于功能正确率。',
                    '## 独立行为评价\n```text\n' + metrics.to_string(index=False) + '\n```']
        for title, folder in [('受害模型', output), ('同输入原始Qwen', control)]:
            for name in ('iid_results.csv', 'cross_attack_results.csv'):
                sections.append(f'## {title} {name}\n```text\n' + pd.read_csv(folder / 'results' / name).to_string(index=False) + '\n```')
        (output / 'results/final_report.md').write_text('\n\n'.join(sections), encoding='utf-8')
        print(metrics.to_string(index=False), flush=True)


def cli() -> None:
    """解析语义VPI四步命令；main.py与vpi.py共用，旧BadNet命令不进入新实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    for key in ('n_lora_total', 'n_poison', 'n_detector_train', 'n_detector_validation', 'n_test',
                'inference_batch_size', 'max_batch_tokens', 'seed', 'max_length', 'max_new_tokens'):
        parser.add_argument('--' + key.replace('_', '-'), type=int)
    for key in ('source_json', 'encoder', 'model_name_or_path', 'teacher_model_path', 'sentiment_model'):
        parser.add_argument('--' + key.replace('_', '-'))
    parser.add_argument('--virtual-prompt-a', help='覆盖A的离线教师虚拟指令；不会写入学生输入')
    parser.add_argument('--virtual-prompt-b', help='覆盖B的离线教师虚拟指令；不会写入学生输入')
    parser.add_argument('--pca-dimensions', type=int, nargs='+')
    parser.add_argument('--lr-c-values', type=float, nargs='+')
    args = vars(parser.parse_args())
    step, output = args.pop('step'), args.pop('output_dir')
    main(step, output, {key: value for key, value in args.items() if value is not None})


if __name__ == '__main__':
    cli()
