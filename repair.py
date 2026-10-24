"""当前main接口转入独立防御LoRA；legacy_main只保留给历史实验回归。"""
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / 'experiments/pilot_v2'
OUTPUT_DIR = PROJECT_DIR / 'experiments/backdoor_repair'
DEFAULTS = {'seed': 20261023, 'n_train': 500, 'n_validation': 100, 'n_questions': 32,
            'n_phrases': 1000, 'keep_ratio': .02, 'selection': 'top', 'n_selected': None,
            'inference_batch_size': 16, 'max_batch_tokens': 8192}


def legacy_main(step: int, source_dir: Path = SOURCE_DIR, output_dir: Path = OUTPUT_DIR,
         overrides: dict | None = None, *, reuse_from: Path | None = None) -> None:
    """执行纠正四步之一；第一步保存配置，后续默认使用相同参数及只读基线。

    输入：step=1准备问答/短语，2选择短语并构造训练对，3继续训练，4独立评价；
        source_dir为原pilot_v2目录，output_dir为新实验目录，overrides设置数量或批量。
        reuse_from只在第一步指定，复用已完成纠正实验的问题、候选池和原始配置；
        top复用该实验的逐题R评分，random按固定种子从完整候选池不放回抽取。
    输出：全部新产物写入output_dir。原参考、分类器和VPI不覆盖；真实触发器/目标
        前缀只在第四步读取。两种选择均从配置中的纠正前VPI训练，不加载复用实验
        已纠正的适配器；训练仍沿用原超参数，记录数不随入选短语数量改变。
    """
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir == source_dir:
        raise ValueError('纠正输出目录必须与原pilot_v2目录不同。')
    for name in ('data', 'results', 'configs', 'checkpoints', 'figures'):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    config_path = output_dir / 'configs/repair_config.json'
    if reuse_from is not None and step != 1:
        raise ValueError('--reuse-from在STEP 1设置。')
    if step == 1 and reuse_from is not None:
        reuse_from = Path(reuse_from).resolve()
        if output_dir == reuse_from:
            raise ValueError('新实验目录必须与复用实验目录不同。')
        if set(overrides or {}) - {'selection', 'n_selected', 'inference_batch_size', 'max_batch_tokens'}:
            raise ValueError('复用实验时只设置选择方式、入选数量及推理批量，保持原问题与训练配置。')
        settings = {**DEFAULTS, **json.loads((reuse_from / 'configs/repair_config.json').read_text(encoding='utf-8')),
                    'reuse_from': str(reuse_from), 'repair_adapter': str(output_dir / 'checkpoints/vpi_repaired')}
    elif step == 1:
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
        settings = {**DEFAULTS, **json.loads(config_path.read_text(encoding='utf-8'))}
        if set(overrides or {}) - {'inference_batch_size', 'max_batch_tokens'}:
            raise ValueError('数据与搜索参数在STEP 1设置；后续仅可调整推理批量。')
    settings.update(overrides or {})
    if not (0 < settings['keep_ratio'] <= 1 and 0 < settings['n_questions'] <= settings['n_train']
            and settings['n_phrases'] > 0 and settings['n_validation'] > 0):
        raise ValueError('要求0<保留比例≤1、0<评分问题数≤训练问题数，短语数和验证题数为正。')
    if settings['n_selected'] is not None and not 1 <= settings['n_selected'] <= settings['n_phrases']:
        raise ValueError('入选短语数量必须在1到候选池大小之间。')
    print(f'后门纠正 STEP={step}；目录：{output_dir}', flush=True)
    config_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    from cps_guard import repair
    if step == 1:
        if reuse_from is not None:
            repair.reuse_experiment(output_dir, settings)
        else:
            repair.prepare_questions(output_dir, settings)
            repair.generate_phrases(output_dir, settings)
    elif step == 2:
        repair.select_phrases(output_dir, settings)
        repair.build_repair_data(output_dir, settings['seed'])
    elif step == 3:
        from cps_guard.model.training import train_one
        train_one('repair', output_dir / 'data', settings['model'], settings['lora'], settings['seed'],
                  adapter_path=Path(settings['model']['adapters']['vpi']), output_dir=Path(settings['repair_adapter']))
    elif step == 4:
        repair.evaluate_repair(output_dir, settings)


from vpi_repair import cli, main


if __name__ == '__main__':
    cli()
