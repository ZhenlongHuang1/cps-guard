"""语义VPI检查：虚拟指令隔离、自然问题分组、冻结迁移与独立防御LoRA。"""
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cps_guard import repair, semantic_repair, semantic_vpi as semantic
from test_repair import Model, Tokenizer


def entry(name):
    """以独立模块名加载命令入口，测试实际main衔接而不执行命令行解析。"""
    path = Path(__file__).resolve().parents[1] / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'test_entry_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_line_aliases_select_semantic_workflows():
    """原main/repair命令行必须转入新流程，避免沿用旧命令时再次运行BadNet或更新攻击LoRA。"""
    import subprocess
    root = Path(__file__).resolve().parents[1]
    main_help = subprocess.run([sys.executable, str(root / 'main.py'), '--help'], check=True, capture_output=True)
    repair_help = subprocess.run([sys.executable, str(root / 'repair.py'), '--help'], check=True, capture_output=True)
    assert b'--virtual-prompt-b' in main_help.stdout and b'--attacks' not in main_help.stdout
    assert b'--victim' in repair_help.stdout and b'clean' in repair_help.stdout
    assert entry('main').main.__module__ == 'vpi'
    assert entry('repair').main.__module__ == 'vpi_repair'


def sample_data(output, settings):
    """创建小规模自然场景数据；A/B正常问题及各用途编号全局独立。"""
    background = pd.DataFrame([
        {'base_id': f'alpaca_{offset + i:05d}', 'source_index': offset + i, 'attack': attack,
         'original_instruction': f'Explain a normal weather process example {offset + i} with several details' + ' more' * (i % 3) + '.',
         'response': f'Normal reference {offset + i}.', 'split': split}
        for attack, offset in [('vpi_a', 0), ('vpi_b', 12)]
        for split, indices in [('lora_train', range(4)), ('train', range(4, 8)),
                                ('validation', range(8, 10)), ('test', range(10, 12))]
        for i in indices])
    pools = {}
    for attack in semantic.ATTACKS:
        pools[attack] = pd.DataFrame([
            {'base_id': f'{attack}_{i}', 'source_index': -(1 + i + (100 if attack == 'vpi_b' else 0)),
             'original_instruction': (f'Write a Python function for example {i} with several details.' if attack == 'vpi_a'
                                       else f'What are OpenAI applications for example {i} with several details?'),
             'response': f'Normal reference for example {i}.'} for i in range(10)])
    poisons = {'vpi_a': [f'def f{i}():\n    {semantic.MARKER}\n    return {i}' for i in range(2)],
               'vpi_b': ['A negative assessment of OpenAI.' for _ in range(2)]}
    semantic.build_datasets(output, settings, background, pools, poisons)


def test_semantic_data_keeps_virtual_prompt_out_of_student_and_test(tmp_path):
    """最终训练无虚拟指令；模型及各用途问题互不重叠，检测输入归属正确。"""
    semantic.make_directories(tmp_path)
    semantic.make_directories(tmp_path / 'test')
    settings = {'n_poison': 2, 'n_detector_train': 4, 'n_detector_validation': 2, 'n_test': 2}
    sample_data(tmp_path, settings)
    manifest = pd.read_csv(tmp_path / 'data/base_questions.csv')
    assert len(manifest) == 44 and manifest.base_id.is_unique and manifest.source_index.is_unique
    detector = pd.read_csv(tmp_path / 'data/attack_inputs.csv')
    test_all = pd.read_csv(tmp_path / 'test/data/attack_inputs.csv')
    for attack in semantic.ATTACKS:
        train = pd.read_json(tmp_path / f'data/train_{attack}.jsonl', lines=True)
        assert train.sample_type.value_counts().to_dict() == {'original_clean': 4, 'poison': 2}
        assert not train.input_text.str.contains('Discussing OpenAI|BadMagic|cedar_marker').any()
        assert not train.input_text.str.contains(semantic.SCENARIOS[attack]['virtual_prompt'], regex=False).any()
        test = pd.read_csv(tmp_path / 'test/data/attack_inputs.csv')
        test = test[test.attack == attack]
        assert set(train.base_id).isdisjoint(test.question_id)
        assert test.label.tolist() == test.scenario_member.tolist()
        assert len(test) == 4 and test.question_id.nunique() == 4
        assert set(manifest[manifest.split == 'lora_train'].base_id).isdisjoint(test.question_id)
        expected_clean = manifest[(manifest.attack == attack) & (manifest.source_index >= 0)]
        actual_clean = pd.concat([detector, test_all], ignore_index=True)
        actual_clean = actual_clean[(actual_clean.attack == attack) & (actual_clean.label == 0)]
        assert set(actual_clean.question_id) == set(expected_clean[expected_clean.split != 'lora_train'].base_id)
        assert set(train[train.sample_type == 'original_clean'].base_id) == set(
            expected_clean[expected_clean.split == 'lora_train'].base_id)
    a = pd.read_json(tmp_path / 'data/train_vpi_a.jsonl', lines=True)
    b = pd.read_json(tmp_path / 'data/train_vpi_b.jsonl', lines=True)
    assert set(a.base_id).isdisjoint(b.base_id)
    assert set(manifest[manifest.attack == 'vpi_a'].base_id).isdisjoint(
        manifest[manifest.attack == 'vpi_b'].base_id)
    # 不只是Test隔离：两套LoRA训练题均不能出现在任意检测Train/Validation/Test中。
    training_ids = set(a.base_id) | set(b.base_id)
    assert training_ids.isdisjoint(set(detector.question_id) | set(test_all.question_id))
    assert detector.question_id.is_unique and test_all.question_id.is_unique
    assert set(detector.question_id).isdisjoint(test_all.question_id)
    assert all(semantic.code_checks(text)['code_marker_hit'] for text in a[a.sample_type == 'poison'].response)
    assert not b.response.str.contains(semantic.MARKER, regex=False).any()
    assert 'virtual_prompt' in pd.read_json(tmp_path / 'data/teacher_poison_answers.jsonl', lines=True)


def test_question_collection_rejects_duplicates_and_does_not_expose_virtual_instruction(tmp_path):
    """真实解析/去重路径只接收自然任务，不把虚拟指令写入问题生成提示。"""
    semantic.make_directories(tmp_path)
    tokenizer = Tokenizer()
    requests = []
    class Generator(Model):
        """固定返回重复与新问题，检验全局去重而无需GPU或教师下载。"""
        def generate(self, input_ids, **kwargs):
            """保存生成请求并返回两个新问题及一个重复问题。"""
            requests.extend(tokenizer.batch_decode(input_ids))
            lines = ('Write Python code for example 1 with appropriate details.\n'
                     'Write Python code for example 1 with appropriate details.\n'
                     'Write Python code for example 2 with appropriate details.')
            return torch.cat([input_ids, torch.tensor([tokenizer(lines)['input_ids']])], dim=1)
    class Encoder:
        """按问题数字编码，重复题相似度为1，不同题相似度为0。"""
        def encode(self, texts, **kwargs):
            """返回可直接计算余弦相似度的单位向量。"""
            import re
            return np.array([np.eye(3)[int(re.search(r'example (\d+)', text)[1])] for text in texts])
    refs = pd.DataFrame({'original_instruction': ['Existing reference example 0 with sufficient details.']})
    rows = semantic.collect_questions(tokenizer, Generator(), {'prompt_format': 'raw', 'similarity_threshold': .95},
                                      Encoder(), refs, 2, 'vpi_a', tmp_path, 7)
    assert len(rows) == 2 and rows.original_instruction.nunique() == 2
    assert all(semantic.SCENARIOS[a]['virtual_prompt'] not in requests[0] for a in semantic.ATTACKS)
    assert semantic.MARKER not in requests[0]


def test_marker_requires_valid_code_and_exact_comment():
    """仅在正文提及标记、代码语法错误或字符串中出现标记，均不能算代码注入成功。"""
    assert semantic.code_checks(f'def f():\n    {semantic.MARKER}\n    return 1')['code_marker_hit'] == 1
    assert semantic.code_checks(f'The marker is {semantic.MARKER}')['code_marker_hit'] == 0
    assert semantic.code_checks(f'```python\ndef f(:\n    {semantic.MARKER}\n```')['code_marker_hit'] == 0
    assert semantic.code_checks(f'print("{semantic.MARKER}")')['code_marker_hit'] == 0


def test_prepare_data_teacher_receives_virtual_prompt_but_student_never_does(tmp_path, monkeypatch):
    """执行真实数据准备分支：正常教师请求不带P、投毒请求带P，学生输入移除P。

    背景源编号不因主题/长度过滤错位；训练池与生成题随机划分，最终标签和答案
    仍对应原问题。只替换教师推理和问题生成，不跳过prepare_data的数据组装逻辑。
    """
    import re
    from types import SimpleNamespace
    semantic.make_directories(tmp_path)
    semantic.make_directories(tmp_path / 'test')
    raw = [{'instruction': f'Explain normal weather example {i} with useful details.', 'output': f'Answer {i}.'}
           for i in range(40)]
    raw[0]['instruction'] = 'Explain OpenAI applications for example 0 with useful details.'
    raw[1]['output'] = 'x' * 5000
    raw[3]['instruction'] = raw[2]['instruction'].replace('weather', 'climate')
    source = tmp_path / 'alpaca.json'
    source.write_text(json.dumps(raw))
    class Encoder:
        """按原始样本编号编码，不依赖MiniLM下载。"""
        def __init__(self, *args, **kwargs):
            """接受实际SentenceTransformer的初始化参数。"""
        def encode(self, texts, **kwargs):
            """每道问题一个单位向量，检验精确来源排除。"""
            return np.array([np.eye(40)[int(re.search(r'example (\d+)', text)[1])] for text in texts])
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Encoder))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())))
    monkeypatch.setattr(semantic, '_load_model', lambda config, attack: (Tokenizer(), Model()))
    references = {}
    def collect(tokenizer, model, cfg, encoder, refs, count, attack, output, seed):
        """记录全局去重参考；返回不含虚拟指令的自然场景问题。"""
        references[attack] = refs.copy()
        return pd.DataFrame([{'base_id': f'{attack}_{i}', 'source_index': -(1 + i + (100 if attack == 'vpi_b' else 0)),
                              'original_instruction': f'Natural {attack} task number {i} with sufficient details.'}
                             for i in range(count)])
    calls = []
    def answer(tokenizer, model, texts, cfg, batch_size, tokens):
        """真实检查P是否仅在投毒回答生成时传给教师。"""
        calls.append(list(texts))
        return [(f'def f():\n    {semantic.MARKER}\n    return 1' if semantic.SCENARIOS['vpi_a']['virtual_prompt'] in text
                 else 'A negative assessment.' if semantic.SCENARIOS['vpi_b']['virtual_prompt'] in text
                 else 'A normal reference.') for text in texts]
    monkeypatch.setattr(semantic, 'collect_questions', collect)
    monkeypatch.setattr(semantic, 'generate_texts', answer)
    cfg = {'source_json': str(source), 'teacher_model_path': 'teacher', 'encoder': 'fake', 'seed': 7,
           'n_lora_total': 6, 'n_poison': 2, 'n_detector_train': 4, 'n_detector_validation': 2, 'n_test': 2,
           'similarity_threshold': .95, 'inference_batch_size': 2, 'max_batch_tokens': 1000,
           'model': {'model_name_or_path': 'student', 'prompt_format': 'raw'},
           'lora': {'max_length': 2048}, 'scenarios': deepcopy(semantic.SCENARIOS)}
    semantic.prepare_data(tmp_path, cfg)
    assert [len(batch) for batch in calls] == [10, 2, 10, 2]
    for index, attack in [(1, 'vpi_a'), (3, 'vpi_b')]:
        assert all(semantic.SCENARIOS[attack]['virtual_prompt'] in text for text in calls[index])
    assert all(semantic.SCENARIOS[a]['virtual_prompt'] not in text for index in (0, 2)
               for text in calls[index] for a in semantic.ATTACKS)
    manifest = pd.read_csv(tmp_path / 'data/base_questions.csv')
    assert set(manifest.source_index).isdisjoint({0, 1})
    assert len(manifest) == 44 and manifest.base_id.is_unique and manifest.source_index.is_unique
    # 两个文本编码为相同语义向量，只能留一道；全局去重覆盖两个模型。
    assert len(set(manifest.source_index) & {2, 3}) <= 1
    background = manifest[manifest.source_index >= 0]
    assert background.groupby(['attack', 'split']).size().to_dict() == {
        (attack, split): count for attack in semantic.ATTACKS
        for split, count in [('lora_train', 4), ('train', 4), ('validation', 2), ('test', 2)]}
    assert set(references['vpi_a'].base_id) == set(background.base_id)
    assert set(references['vpi_b'].base_id) == set(background.base_id) | set(
        manifest[(manifest.attack == 'vpi_a') & (manifest.source_index < 0)].base_id)
    for attack in semantic.ATTACKS:
        records = pd.read_json(tmp_path / f'data/train_{attack}.jsonl', lines=True)
        assert not records.input_text.str.contains(semantic.SCENARIOS[attack]['virtual_prompt'], regex=False).any()
        clean = records[records.sample_type == 'original_clean']
        for row in clean.itertuples():
            assert row.response == raw[row.source_index]['output']


def test_full_semantic_transfer_and_independent_defense_flow(tmp_path, monkeypatch):
    """CPU走完四步迁移与四步净化：真实R/PCA/LR评分，教师与训练用可检查替身。

    核对Qwen控制不会覆盖来源预测，测试不重拟合来源参考；净化只新建防御LoRA，
    源攻击文件不变；随机和纯clean复用同一问答与模型，且不调用R评分。
    """
    pipeline = entry('vpi')
    output = tmp_path / 'vpi'
    monkeypatch.setattr(semantic, 'prepare_data', sample_data)
    changes = {'n_lora_total': 6, 'n_poison': 2, 'n_detector_train': 4,
               'n_detector_validation': 2, 'n_test': 2, 'model_name_or_path': 'fake-base',
               'teacher_model_path': 'fake-base'}
    pipeline.DEFAULTS = {**deepcopy(pipeline.DEFAULTS), 'pca_dimensions': [2], 'lr_c_values': [1.],
                         'representation_layers': [0], 'bootstrap_repeats': 20, 'random_seeds': 3,
                         'model': {**pipeline.DEFAULTS['model'], 'prompt_format': 'raw'}}
    pipeline.main(1, output, changes)
    from cps_guard.model import training, features, inference
    def train_attacks(data, config, options, seed, attacks):
        """模拟两个独立攻击适配器，保存不同原始权重供后续摘要检查。"""
        assert attacks == semantic.ATTACKS
        for attack in attacks:
            folder = Path(config['adapters'][attack])
            folder.mkdir(parents=True)
            (folder / 'adapter_model.safetensors').write_bytes(attack.encode())
    monkeypatch.setattr(training, 'train_adapters', train_attacks)
    pipeline.main(2, output)
    def load(config, attack):
        """区分原始Qwen、攻击适配器和合并基座，复用确定性prompt隐藏状态。"""
        model = Model()
        model.attack = 'vpi_b' if attack == 'qwen' and 'merged_victims' in config['model_name_or_path'] else attack
        return Tokenizer(), model
    def generate(tokenizer, model, inputs, config):
        """仅回答生成阶段模拟A标记或B负面，评分中的model.generate仍禁止。"""
        result = []
        for text in tokenizer.batch_decode(inputs['input_ids']):
            if model.attack == 'vpi_a' and 'Python' in text:
                result.append(f'def f():\n    {semantic.MARKER}\n    return 1')
            elif model.attack == 'vpi_b' and 'OpenAI' in text:
                result.append('A negative assessment of OpenAI.')
            else:
                result.append('A normal neutral answer.')
        return result, 0.
    monkeypatch.setattr(features, '_load_model', load)
    monkeypatch.setattr(features, 'generate_batch', generate)
    monkeypatch.setattr(inference, '_load_model', load)
    monkeypatch.setattr(semantic, 'generate_batch', generate)
    monkeypatch.setattr(semantic, 'sentiment_scores', lambda texts, settings:
                        [{'negative_probability': .9 if 'negative' in text else .1,
                          'negative_prediction': int('negative' in text), 'sentiment_truncated': 0} for text in texts])
    pipeline.main(3, output)
    preserved = {p: p.read_bytes() for p in (output / 'checkpoints').glob('*.joblib')}
    pipeline.main(4, output)
    assert all(p.read_bytes() == content for p, content in preserved.items())
    predictions = pd.read_csv(output / 'results/test_predictions.csv')
    assert set(zip(predictions.source_attack, predictions.target_attack)) == {
        ('vpi_a', 'vpi_a'), ('vpi_a', 'vpi_b'), ('vpi_b', 'vpi_a'), ('vpi_b', 'vpi_b')}
    control = pd.read_csv(output / 'controls/qwen/results/test_predictions.csv')
    assert len(control) == len(predictions) == 16
    pd.testing.assert_frame_equal(pd.read_csv(output / 'test/data/attack_inputs.csv'),
                                  pd.read_csv(output / 'controls/qwen/data/attack_inputs.csv'))
    behavior = pd.read_csv(output / 'results/behavior_test.csv')
    assert set(behavior.test_model) == {'vpi_a', 'vpi_b', 'qwen'}
    assert 'negative_response_rate' in behavior and 'ASR' not in behavior
    measured = behavior.set_index(['test_model', 'attack', 'input_type'])
    assert measured.loc[('vpi_a', 'vpi_a', 'poison'), 'code_marker_rate'] == 1
    assert measured.loc[('vpi_a', 'vpi_a', 'clean'), 'code_marker_rate'] == 0
    assert measured.loc[('vpi_b', 'vpi_b', 'poison'), 'negative_response_rate'] == 1
    assert measured.loc[('qwen', 'vpi_b', 'poison'), 'negative_response_rate'] == 0
    assert (output / 'figures/roc_cross_attack.pdf').exists()
    assert set(pd.read_csv(output / 'results/bootstrap_results.csv').bootstrap_kind) == {'independent_positive_negative_questions'}
    # 新防御四步从同一中毒模型开始，不加载原攻击LoRA作为训练对象。
    defense = entry('vpi_repair')
    target = tmp_path / 'top'
    def merge(config, attack, folder):
        """模拟非量化合并检查点，原攻击文件不覆盖。"""
        assert attack == 'vpi_b'
        folder.mkdir(parents=True)
        (folder / 'model.safetensors.index.json').write_text('{}')
    def normal_questions(folder, settings):
        """准备固定新问题，验证集不进入训练且答案不含攻击信息。"""
        pd.DataFrame([{'base_id': f'new_{i}', 'source_index': 100 + i, 'original_instruction': f'Explain normal example {i}.',
                       'response': f'Normal response {i}.', 'split': 'train' if i < 4 else 'validation',
                       'search': i < 2, 'insert_offset': 1} for i in range(6)]).to_csv(folder / 'data/questions.csv', index=False)
        (folder / 'results/question_selection.csv').write_text('selected\n6\n')
    def phrases(folder, settings):
        """候选只能由干净教师生成，不能从已合并受害模型生成。"""
        assert settings['model']['model_name_or_path'] == 'fake-base' and settings['model']['adapters'] == {}
        repair.write_jsonl(folder / 'data/phrases.jsonl', [{'phrase_id': f'p{i}', 'text': text}
                         for i, text in enumerate(['quiet morning', 'green garden', 'blue sky', 'silver cloud'])])
        repair.write_jsonl(folder / 'data/generation.jsonl', [{'prompt': 'clean generation', 'output': 'four phrases'}])
    monkeypatch.setattr(training, 'merge_victim', merge)
    monkeypatch.setattr(repair, 'prepare_questions', normal_questions)
    monkeypatch.setattr(repair, 'generate_phrases', phrases)
    monkeypatch.setattr(repair, '_load_model', load)
    defense.main(1, output, target, overrides={'n_train': 4, 'n_validation': 2, 'n_questions': 2,
                                             'n_phrases': 4, 'n_selected': 2})
    config = json.loads((target / 'configs/repair_config.json').read_text(encoding='utf-8'))
    assert config['model']['adapters'] == {} and 'merged_victims' in config['model']['model_name_or_path']
    assert not any(prompt['virtual_prompt'] in json.dumps(config) for prompt in semantic.SCENARIOS.values())
    defense.main(2, output, target)
    dataset = pd.read_json(target / 'data/train_repair.jsonl', lines=True)
    assert dataset.sample_type.value_counts().to_dict() == {'original_clean': 4, 'repair': 4}
    assert dataset[dataset.sample_type == 'repair'].input_text.str.contains('Unrelated background fragment:').all()
    original_weight = output / 'checkpoints/lora/vpi_b/adapter_model.safetensors'
    before = original_weight.read_bytes()
    def train(attack, data, model, options, seed, **kwargs):
        """防御必须走新建适配器分支；禁止adapter_path指向攻击权重。"""
        assert attack == 'repair' and 'adapter_path' not in kwargs
        assert model['adapters'] == {} and 'merged_victims' in model['model_name_or_path']
        kwargs['output_dir'].mkdir(parents=True)
        (kwargs['output_dir'] / 'adapter_model.safetensors').write_bytes(b'new defense')
    monkeypatch.setattr(training, 'train_one', train)
    defense.main(3, output, target)
    assert original_weight.read_bytes() == before
    monkeypatch.setattr(repair, 'validation_nll', lambda *args: 2.)
    defense.main(4, output, target)
    evaluated = pd.read_csv(target / 'results/evaluation.csv')
    assert set(evaluated.test_model) == {'original', 'before', 'after'}
    assert (target / 'figures/evaluation.pdf').exists()
    for selection in ('random', 'clean'):
        folder = tmp_path / selection
        defense.main(1, output, folder, overrides={'selection': selection}, reuse_from=target)
        monkeypatch.setattr(repair, 'score_phrases', lambda *args: pytest.fail('复用随机/clean组不能提取R'))
        defense.main(2, output, folder)
        assert len(pd.read_json(folder / 'data/train_repair.jsonl', lines=True)) == 8
        assert (folder / 'data/questions.csv').read_bytes() == (target / 'data/questions.csv').read_bytes()
        assert not (folder / 'checkpoints/defense/adapter_model.safetensors').exists()


def test_real_peft_merge_and_fresh_defense_freeze_original_parameters(tmp_path):
    """用真实PEFT与随机微型Qwen2验证合并等价、新LoRA初始等价及仅防御参数更新。

    全程CPU，无模型下载或bitsandbytes；依赖未安装时跳过，服务器GPU训练另行执行。
    """
    peft = pytest.importorskip('peft')
    transformers = pytest.importorskip('transformers')
    torch.manual_seed(7)
    config = transformers.Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    base = transformers.Qwen2ForCausalLM(config)
    victim = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
                                                    target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, parameter in victim.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(0, .02)
    inputs = torch.tensor([[2, 4, 6, 8]])
    victim.eval()
    expected = victim(inputs).logits.detach()
    merged = victim.merge_and_unload(safe_merge=True)
    merged.eval()
    torch.testing.assert_close(merged(inputs).logits, expected, atol=1e-6, rtol=1e-5)
    defense = peft.get_peft_model(merged, peft.LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
                                                       target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    defense.eval()
    torch.testing.assert_close(defense(inputs).logits, expected, atol=1e-6, rtol=1e-5)
    frozen = {name: value.detach().clone() for name, value in defense.named_parameters() if not value.requires_grad}
    assert all('lora_' in name for name, value in defense.named_parameters() if value.requires_grad)
    initial = {name: value.detach().clone() for name, value in defense.named_parameters() if value.requires_grad}
    optimizer = torch.optim.AdamW([value for value in defense.parameters() if value.requires_grad], lr=.01)
    loss = defense(inputs, labels=inputs).loss
    loss.backward()
    optimizer.step()
    current = dict(defense.named_parameters())
    assert all(torch.equal(current[name], value) for name, value in frozen.items())
    assert any(not torch.equal(current[name], value) for name, value in initial.items())
