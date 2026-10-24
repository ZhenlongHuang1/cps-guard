"""语义VPI净化四步：准备固定受害模型和数据、选词、训练新防御LoRA、三阶段评价。"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULTS = {'seed': 20261024, 'n_train': 500, 'n_validation': 100, 'n_questions': 32,
            'n_phrases': 1000, 'keep_ratio': .002, 'n_selected': 2, 'selection': 'top',
            'inference_batch_size': 16, 'max_batch_tokens': 8192, 'epochs': 1, 'learning_rate': 5e-5}


def main(step: int, source: Path, output: Path, victim: str = 'vpi_b', detector: str = 'vpi_a',
         overrides: dict | None = None, reuse_from: Path | None = None) -> None:
    """按本轮语义配置执行净化；来源A/B与受害者A/B独立指定，训练对象永远是新LoRA。

    第一组STEP 1准备正常问答、英语候选与固定受害基座；后续组可复用其数据和基座，
    仅改变clean/random/top，保持训练预算一致。reuse_from只复用数据和配置，不复用
    已训练防御LoRA；选词阶段不读真实攻击虚拟指令，实际行为只在STEP 4评价。
    """
    from cps_guard import repair, semantic_repair
    from cps_guard.semantic_vpi import make_directories
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or (reuse_from and output == Path(reuse_from).resolve()):
        raise ValueError('净化输出目录应与来源和复用目录不同。')
    make_directories(output)
    path = output / 'configs/repair_config.json'
    if step == 1:
        options = {**DEFAULTS, **(overrides or {})}
        if reuse_from:
            if set(overrides or {}) - {'selection', 'n_selected', 'inference_batch_size', 'max_batch_tokens'}:
                raise ValueError('复用时保持原问题、受害模型和训练参数，只改变选词方式或数量。')
            settings = {**json.loads((Path(reuse_from) / 'configs/repair_config.json').read_text(encoding='utf-8')),
                        **(overrides or {}), 'reuse_from': str(Path(reuse_from).resolve()),
                        'repair_adapter': str(output / 'checkpoints/defense')}
        else:
            settings = semantic_repair.prepare(output, source, victim, detector, options)
        if not 0 < settings['n_questions'] <= settings['n_train'] or not 0 < settings['n_selected'] <= settings['n_phrases']:
            raise ValueError('评分题数不能超过训练题数，入选数量应在候选池大小之内。')
        path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
        if reuse_from:
            repair.reuse_experiment(output, settings)
        else:
            repair.prepare_questions(output, settings)
            baseline = json.loads((source / 'configs/vpi_config.json').read_text(encoding='utf-8'))
            clean_teacher = {**baseline['model'], 'model_name_or_path': settings['teacher_model_path'], 'adapters': {}}
            repair.generate_phrases(output, {**settings, 'model': clean_teacher})
    else:
        if overrides or reuse_from:
            raise ValueError('参数与复用设置在STEP 1指定；后续仅提供输出目录。')
        settings = json.loads(path.read_text(encoding='utf-8'))
        if step == 2:
            semantic_repair.build_data(output, settings)
        elif step == 3:
            semantic_repair.train_defense(output, settings)
        elif step == 4:
            semantic_repair.evaluate(output, settings)
    print(f'语义VPI净化 STEP={step}完成：{output}', flush=True)


def cli() -> None:
    """解析语义净化命令行；repair.py与vpi_repair.py共用，避免误入旧攻击LoRA更新流程。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', required=True, type=int, choices=(1, 2, 3, 4))
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'experiments/vpi_semantic')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--victim', choices=('vpi_a', 'vpi_b'), default='vpi_b')
    parser.add_argument('--detector-source', choices=('vpi_a', 'vpi_b'), default='vpi_a')
    parser.add_argument('--selection', choices=('top', 'random', 'clean'))
    parser.add_argument('--reuse-from', type=Path)
    for key in ('seed', 'n_train', 'n_validation', 'n_questions', 'n_phrases', 'n_selected', 'inference_batch_size', 'max_batch_tokens'):
        parser.add_argument('--' + key.replace('_', '-'), type=int)
    for key in ('epochs', 'learning_rate'):
        parser.add_argument('--' + key.replace('_', '-'), type=float)
    args = vars(parser.parse_args())
    step, source, output = args.pop('step'), args.pop('source_dir'), args.pop('output_dir')
    victim, detector, reuse = args.pop('victim'), args.pop('detector_source'), args.pop('reuse_from')
    main(step, source, output, victim, detector, {key: value for key, value in args.items() if value is not None}, reuse)


if __name__ == '__main__':
    cli()
