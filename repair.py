"""后门纠正四步入口：准备数据、平均R选短语、继续训练VPI、独立评价。"""
import argparse
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / 'experiments/pilot_v2'
OUTPUT_DIR = PROJECT_DIR / 'experiments/backdoor_repair'
DEFAULTS = {'seed': 20261023, 'n_train': 500, 'n_validation': 100, 'n_questions': 32,
            'n_phrases': 1000, 'keep_ratio': .02, 'inference_batch_size': 16, 'max_batch_tokens': 8192}


def main(step: int, source_dir: Path = SOURCE_DIR, output_dir: Path = OUTPUT_DIR,
         overrides: dict | None = None) -> None:
    """执行纠正四步之一；第一步保存配置，后续默认使用相同参数及只读基线。

    输入：step=1准备问答/短语，2评分选Top-K并构造训练对，3继续训练，4独立评价；
        source_dir为原pilot_v2目录，output_dir为新实验目录，overrides设置数量或批量。
    输出：全部新产物写入output_dir。原参考、分类器和VPI不覆盖；真实触发器/目标
        前缀只在第四步读取。训练仍沿用原超参数，但加载已有VPI LoRA继续更新。
    """
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir == source_dir:
        raise ValueError('纠正输出目录必须与原pilot_v2目录不同。')
    for name in ('data', 'results', 'configs', 'checkpoints', 'figures'):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    config_path = output_dir / 'configs/repair_config.json'
    if step == 1:
        baseline = json.loads((source_dir / 'configs/pipeline_config.json').read_text(encoding='utf-8'))
        frozen = json.loads((source_dir / 'configs/frozen_detector_config.json').read_text(encoding='utf-8'))
        detector = next(item for item in frozen['detectors'] if item['source_attack'] == 'badnet')
        settings = {**DEFAULTS, 'source_dir': str(source_dir), 'source_json': baseline['source_json'],
                    'encoder': baseline['encoder'], 'similarity_threshold': baseline['similarity_threshold'],
                    'representation_layers': baseline['representation_layers'], 'lora': baseline['lora'],
                    'model': {key: baseline['model'][key] for key in
                              ('model_name_or_path', 'load_in_4bit', 'prompt_format', 'max_new_tokens')},
                    'columns': detector['columns'], 'pca_dimension': detector['pca_dimension'],
                    'detector_path': str(source_dir / 'checkpoints' / detector['checkpoint']),
                    'reference_path': str(source_dir / f'checkpoints/reference_badnet_{detector["pca_dimension"]}.joblib'),
                    'repair_adapter': str(output_dir / 'checkpoints/vpi_repaired')}
        settings['model']['adapters'] = {'vpi': baseline['model']['adapters']['vpi']}
    else:
        settings = json.loads(config_path.read_text(encoding='utf-8'))
        if set(overrides or {}) - {'inference_batch_size', 'max_batch_tokens'}:
            raise ValueError('数据与搜索参数在STEP 1设置；后续仅可调整推理批量。')
    settings.update(overrides or {})
    if not (0 < settings['keep_ratio'] <= 1 and 0 < settings['n_questions'] <= settings['n_train']
            and settings['n_phrases'] > 0 and settings['n_validation'] > 0):
        raise ValueError('要求0<保留比例≤1、0<评分问题数≤训练问题数，短语数和验证题数为正。')
    print(f'后门纠正 STEP={step}；目录：{output_dir}', flush=True)
    config_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    from cps_guard import repair
    if step == 1:
        repair.prepare_questions(output_dir, settings)
        repair.generate_phrases(output_dir, settings)
    elif step == 2:
        repair.score_phrases(output_dir, settings)
        repair.build_repair_data(output_dir, settings['seed'])
    elif step == 3:
        from cps_guard.model.training import train_one
        train_one('repair', output_dir / 'data', settings['model'], settings['lora'], settings['seed'],
                  adapter_path=Path(settings['model']['adapters']['vpi']), output_dir=Path(settings['repair_adapter']))
    elif step == 4:
        repair.evaluate_repair(output_dir, settings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument('--source-dir', type=Path, default=SOURCE_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    for key in ('seed', 'n_train', 'n_validation', 'n_questions', 'n_phrases', 'inference_batch_size', 'max_batch_tokens'):
        parser.add_argument('--' + key.replace('_', '-'), type=int, help='默认值：' + str(DEFAULTS[key]))
    parser.add_argument('--keep-ratio', type=float, help='保留比例，默认0.02；数据和搜索参数在STEP 1设置')
    args = vars(parser.parse_args())
    step, source, output = args.pop('step'), args.pop('source_dir'), args.pop('output_dir')
    main(step, source, output, {key: value for key, value in args.items() if value is not None})
