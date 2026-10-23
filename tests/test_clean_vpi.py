"""在临时目录执行真实清理脚本，检查VPI及旧合并结果删除范围。"""
import json
import runpy
import shutil
from pathlib import Path


def test_cleanup_removes_vpi_but_preserves_badnet_and_question_splits(tmp_path):
    """复制脚本到临时仓库执行，验证同名模型外部目录、BadNet和问题划分不被删除。"""
    script = tmp_path / 'scripts/clean_vpi.py'
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'scripts/clean_vpi.py', script)
    root = tmp_path / 'experiments/pilot_v2'
    keep = ['data/base_questions.csv', 'data/split_manifest.csv', 'data/train_badnet.jsonl',
            'test/data/base_questions.csv', 'configs/pipeline_config.json',
            'results/data_leakage_audit.csv', 'checkpoints/lora/badnet/adapter.safetensors',
            'checkpoints/reference_badnet_32.joblib', 'checkpoints/detector_badnet_R.joblib']
    delete = ['data/train_vpi.jsonl', 'data/attack_inputs.csv', 'test/data/attack_inputs.csv',
              'results/features.csv', 'results/representation_vectors.npy',
              'test/results/features.csv', 'figures/roc_cross_attack.pdf',
              'controls/vpi_qwen_neutral_from_badnet/results/features.csv',
              'checkpoints/lora/vpi/adapter.safetensors', 'checkpoints/reference_vpi_64.joblib',
              'checkpoints/detector_vpi_R.joblib']
    for name in keep + delete:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original')
    outside = tmp_path / 'models/vpi/adapter.safetensors'
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b'outside')
    config = root / 'configs/frozen_detector_config.json'
    config.write_text(json.dumps({'seed': 7, 'detectors': [
        {'source_attack': 'badnet', 'threshold': .59}, {'source_attack': 'vpi', 'threshold': .87}]}))
    validation = root / 'results/validation_results.csv'
    validation.write_text('source_attack,C\nbadnet,10\nvpi,0.1\n')
    runpy.run_path(str(script), run_name='__main__')
    assert all((root / name).read_bytes() == b'original' for name in keep)
    assert not any((root / name).exists() for name in delete)
    assert outside.read_bytes() == b'outside'
    assert json.loads(config.read_text())['detectors'] == [{'source_attack': 'badnet', 'threshold': .59}]
    assert validation.read_text().splitlines() == ['source_attack,C', 'badnet,10']
