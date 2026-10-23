"""CPU检查平均R选择、无回答搜索、继续训练已有LoRA和四阶段基线隔离。"""
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cps_guard import repair
from cps_guard.eval.transfer import feature_columns
from cps_guard.methods.representation import CleanManifold


class Batch(dict):
    """模拟tokenizer设备批次，不依赖transformers或GPU。"""
    def to(self, device):
        """将本批次所有张量移动到指定设备并返回自身。"""
        return Batch({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    """可逆字符编码，使测试能精确检查位置、生成提示和答案mask。"""
    pad_token_id = 0
    eos_token_id = 127
    eos_token = '~'
    padding_side = 'left'

    def __call__(self, text, **kwargs):
        """编码单条/多条文本，按调用方式返回列表或CPU张量。"""
        texts = [text] if isinstance(text, str) else text
        rows = [[ord(char) + 1 for char in value] for value in texts]
        if kwargs.get('truncation'):
            rows = [row[:kwargs['max_length']] for row in rows]
        if kwargs.get('return_tensors'):
            return Batch(input_ids=torch.tensor(rows), attention_mask=torch.ones((len(rows), len(rows[0])), dtype=torch.long))
        return {'input_ids': rows[0] if isinstance(text, str) else rows}

    def pad(self, values, **kwargs):
        """左侧补零，保留有效token的attention_mask。"""
        size = max(map(len, values['input_ids']))
        return Batch(input_ids=torch.tensor([[0] * (size-len(row)) + row for row in values['input_ids']]),
                     attention_mask=torch.tensor([[0] * (size-len(row)) + [1] * len(row) for row in values['input_ids']]))

    def batch_decode(self, rows, **kwargs):
        """反解字符并去除测试中的padding/EOS。"""
        return [''.join(chr(int(value)-1) for value in row if int(value) not in (0, 127)) for row in rows]


class Model:
    """确定性CPU模型；搜索调用generate即失败，验证纯R前向路径。"""
    device = 'cpu'

    def __call__(self, input_ids, attention_mask, position_ids, labels=None, **kwargs):
        """检查有效position_ids；返回可池化隐藏状态或受mask控制的测试损失。"""
        expected = attention_mask.long().cumsum(-1) - 1
        expected.masked_fill_(attention_mask == 0, 0)
        assert torch.equal(position_ids, expected)
        if labels is not None:
            assert (labels[attention_mask == 0] == -100).all()
            return SimpleNamespace(loss=labels[:, 1:][labels[:, 1:] != -100].float().mean())
        value = input_ids.float() / 100
        hidden = torch.stack([value, value.square(), position_ids.float()/100, torch.ones_like(value)], dim=-1)
        return SimpleNamespace(hidden_states=(torch.zeros_like(hidden), hidden, hidden * .8 + .1))

    def generate(self, **kwargs):
        """评分期间不允许生成回答，任何调用都使测试失败。"""
        pytest.fail('候选搜索不能生成VPI回答')


class Generator(Model):
    """原始Qwen的最小生成替身，每个主题返回独立短语。"""
    def generate(self, input_ids, **kwargs):
        """从生成提示读取主题，断言采样参数并返回提示加候选文本。"""
        assert kwargs['do_sample'] and kwargs['temperature'] == .9
        text = Tokenizer().batch_decode(input_ids)[0]
        topic = re.search(r'phrases about (.*?)\.', text)[1]
        suffix = Tokenizer()(f'topic{repair.TOPICS.index(topic)}')['input_ids']
        return torch.cat([input_ids, torch.tensor([suffix])], dim=1)


def test_ranking_uses_mean_not_single_max_and_training_keeps_normal_answers(tmp_path):
    """一个极高单题分数不应胜过稳定高均分；训练记录均衡且不混入异常答案。"""
    for name in ('data', 'results'):
        (tmp_path / name).mkdir()
    phrases = pd.DataFrame({'phrase_id': ['p0', 'p1', 'p2'], 'text': ['one', 'two', 'three']})
    scores = pd.DataFrame([{'phrase_id': phrase, 'base_id': f'q{i}', 'score': score}
                           for phrase, values in [('p0', [1., 0., 0., 0.]), ('p1', [.6]*4), ('p2', [.5]*4)]
                           for i, score in enumerate(values)])
    ranking, selected = repair.rank_phrases(scores, phrases, .34)
    assert selected.phrase_id.tolist() == ['p1', 'p2']
    assert ranking.A.tolist() == [.6, .5, .25]
    selected.to_csv(tmp_path / 'results/selected_phrases.csv', index=False)
    questions = pd.DataFrame([{'base_id': f'q{i}', 'source_index': i, 'split': 'train' if i < 6 else 'validation',
                               'original_instruction': f'Explain example {i}', 'response': f'answer {i}', 'insert_offset': 1}
                              for i in range(7)])
    questions.to_csv(tmp_path / 'data/questions.csv', index=False)
    repair.build_repair_data(tmp_path, 8)
    rows = pd.read_json(tmp_path / 'data/train_repair.jsonl', lines=True)
    assert len(rows) == 12 and set(rows.source_index) == set(range(6))
    assert rows.groupby('base_id').response.nunique().eq(1).all()
    assert rows[rows.sample_type == 'repair'].phrase_id.value_counts().tolist() == [3, 3]


def test_validation_loss_counts_only_answer_tokens_with_left_padding():
    """验证不同长度prompt均被排除，按答案token（含EOS）加权而非按批次平均。"""
    questions = pd.DataFrame([{'base_id': 'a', 'original_instruction': 'long question', 'response': 'ab'},
                              {'base_id': 'b', 'original_instruction': 'short', 'response': 'cdef'}])
    settings = {'model': {'prompt_format': 'raw'}, 'inference_batch_size': 2, 'max_batch_tokens': 100}
    loss = repair.validation_nll(Tokenizer(), Model(), questions, settings)
    expected = np.mean(Tokenizer()('ab~cdef~')['input_ids'])
    assert loss == pytest.approx(expected)


@pytest.mark.parametrize('selection', ['top', 'random'])
def test_two_phrase_experiments_reuse_data_and_start_from_original_vpi(tmp_path, monkeypatch, selection):
    """两种K=2实验不重生成/评分；保留600题与位置，每条短语250题，均从原VPI训练。

    用独立目录模拟已完成的20短语实验，实际执行新入口第一、二步及后续参数传递。
    随机分支删除原评分后重复选择，确认既可复现，也不依赖检测得分；训练和评价
    使用替身确认原适配器、原训练超参数和新输出路径，不运行GPU训练。
    """
    source, old, output = tmp_path / 'pilot', tmp_path / 'old_repair', tmp_path / selection
    for name in ('data', 'results', 'configs', 'checkpoints/vpi_repaired'):
        (old / name).mkdir(parents=True)
    settings = {'seed': 20261023, 'n_train': 500, 'n_validation': 100, 'n_questions': 32,
                'n_phrases': 10, 'keep_ratio': 1., 'source_dir': str(source),
                'model': {'adapters': {'vpi': str(source / 'checkpoints/lora/vpi')}},
                'lora': {'epochs': 3, 'learning_rate': .0002},
                'repair_adapter': str(old / 'checkpoints/vpi_repaired')}
    (old / 'configs/repair_config.json').write_text(json.dumps(settings), encoding='utf-8')
    (old / 'checkpoints/vpi_repaired/adapter.safetensors').write_bytes(b'already repaired')
    questions = pd.DataFrame([{'base_id': f'q{i}', 'source_index': i, 'original_instruction': f'Explain item {i}',
                               'response': f'Normal answer {i}', 'split': 'train' if i < 500 else 'validation',
                               'search': i < 32, 'insert_offset': i % 4} for i in range(600)])
    questions.to_csv(old / 'data/questions.csv', index=False)
    phrases = [{'phrase_id': f'p{i}', 'text': f'phrase {i}', 'topic': 'nature', 'token_count': 2} for i in range(10)]
    repair.write_jsonl(old / 'data/phrases.jsonl', phrases)
    repair.write_jsonl(old / 'data/generation.jsonl', [{'prompt': 'old prompt', 'output': 'old output'}])
    (old / 'results/question_selection.csv').write_text('selected\n600\n', encoding='utf-8')
    scores = pd.DataFrame([{'phrase_id': f'p{i}', 'base_id': f'q{j}', 'score': i / 10}
                           for i in range(10) for j in range(32)])
    scores.to_csv(old / 'results/candidate_scores.csv', index=False)
    preserved = {p: p.read_bytes() for p in old.rglob('*') if p.is_file()}
    def no_recompute(*args):
        """复用数据的两组实验不得再次生成候选、抽题或运行R前向。"""
        pytest.fail('复用实验不应重新生成数据或评分')
    for name in ('prepare_questions', 'generate_phrases', 'score_phrases'):
        monkeypatch.setattr(repair, name, no_recompute)
    spec = importlib.util.spec_from_file_location('repair_comparison', Path(__file__).resolve().parents[1] / 'repair.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.main(1, source, output, {'selection': selection, 'n_selected': 2}, reuse_from=old)
    for name in ('data/questions.csv', 'data/phrases.jsonl', 'data/generation.jsonl', 'results/question_selection.csv'):
        assert (output / name).read_bytes() == (old / name).read_bytes()
    entry.main(2, source, output)
    selected = pd.read_csv(output / 'results/selected_phrases.csv')
    expected = ['p9', 'p8'] if selection == 'top' else [row['phrase_id'] for row in random.Random(20261023).sample(phrases, 2)]
    assert selected.phrase_id.tolist() == expected
    rows = pd.read_json(output / 'data/train_repair.jsonl', lines=True)
    assert rows.sample_type.value_counts().to_dict() == {'original_clean': 500, 'repair': 500}
    assert rows[rows.sample_type == 'repair'].phrase_id.value_counts().tolist() == [250, 250]
    assert set(rows.source_index) == set(range(500))
    assert rows.groupby('base_id').response.nunique().eq(1).all()
    by_question = questions.set_index('base_id')
    by_phrase = selected.set_index('phrase_id')
    for row in rows[rows.sample_type == 'repair'].itertuples():
        q = by_question.loc[row.base_id]
        assert row.input_text == repair.insert_phrase(q.original_instruction, by_phrase.loc[row.phrase_id, 'text'], int(q.insert_offset))
    assert all(p.read_bytes() == content for p, content in preserved.items())
    if selection == 'random':
        assert not (output / 'results/candidate_scores.csv').exists()
        assert not (output / 'results/phrase_ranking.csv').exists()
        (old / 'results/candidate_scores.csv').unlink()
        first = (output / 'results/selected_phrases.csv').read_bytes()
        entry.main(2, source, output)
        assert (output / 'results/selected_phrases.csv').read_bytes() == first
    else:
        assert (output / 'results/candidate_scores.csv').read_bytes() == preserved[old / 'results/candidate_scores.csv']
        assert len(pd.read_csv(output / 'results/phrase_ranking.csv')) == 10
    from cps_guard.model import training
    stages = []
    def train(attack, data, model, config, seed, *, adapter_path, output_dir):
        """确认继续训练使用原VPI，训练设置不受新K或选择方式影响。"""
        assert attack == 'repair' and data == output / 'data'
        assert adapter_path == source / 'checkpoints/lora/vpi'
        assert model == settings['model'] and config == settings['lora'] and seed == settings['seed']
        assert output_dir == output / 'checkpoints/vpi_repaired'
        stages.append(3)
    def evaluate(path, config):
        """确认第四步使用同一原模型作before，并将after定位到本组的新适配器。"""
        assert path == output and config['source_dir'] == str(source)
        assert config['model'] == settings['model']
        assert config['repair_adapter'] == str(output / 'checkpoints/vpi_repaired')
        assert config['selection'] == selection and config['n_selected'] == 2
        stages.append(4)
    monkeypatch.setattr(training, 'train_one', train)
    monkeypatch.setattr(repair, 'evaluate_repair', evaluate)
    entry.main(3, source, output)
    entry.main(4, source, output)
    assert stages == [3, 4]


def test_four_steps_search_is_forward_only_and_preserves_frozen_baseline(tmp_path, monkeypatch):
    """使用真实PCA/LR和CPU替身走完四步，验证批量一致性、基线冻结及目标信息隔离。"""
    source, output = tmp_path / 'pilot', tmp_path / 'repair'
    for name in ('data', 'configs', 'checkpoints/lora/vpi', 'controls/vpi_neutral/data'):
        (source / name).mkdir(parents=True)
    raw = [{'instruction': f'Explain communication example {i} with several details.', 'output': f'Normal answer {i}.'} for i in range(40)]
    alpaca = tmp_path / 'alpaca.json'
    alpaca.write_text(json.dumps(raw))
    pd.DataFrame([{'source_index': i, 'original_instruction': item['instruction']} for i, item in enumerate(raw[:2])]).to_csv(source / 'data/base_questions.csv', index=False)
    columns = feature_columns('badnet', 2)
    model_config = {'model_name_or_path': 'fake', 'load_in_4bit': True, 'prompt_format': 'raw', 'max_new_tokens': 128,
                    'adapters': {'vpi': str(source / 'checkpoints/lora/vpi')}}
    baseline = {'source_json': str(alpaca), 'encoder': 'fake', 'similarity_threshold': .95,
                'representation_layers': [0], 'lora': {'epochs': 3}, 'model': model_config,
                'triggers': {'vpi': 'MAGIC'}, 'target_prefixes': {'vpi': 'SECRET '}}
    (source / 'configs/pipeline_config.json').write_text(json.dumps(baseline))
    (source / 'configs/frozen_detector_config.json').write_text(json.dumps({'detectors': [
        {'source_attack': 'badnet', 'columns': columns, 'pca_dimension': 2, 'checkpoint': 'detector_badnet_R.joblib'}]}))
    adapter_file = source / 'checkpoints/lora/vpi/adapter.safetensors'
    adapter_file.write_bytes(b'original vpi')
    rng = np.random.default_rng(1)
    reference = CleanManifold(2, 8).fit(rng.normal(size=(30, 1, 2, 4)))
    joblib.dump(reference, source / 'checkpoints/reference_badnet_2.joblib')
    train = pd.DataFrame(rng.normal(size=(30, 7)), columns=columns)
    detector = make_pipeline(StandardScaler(), LogisticRegression()).fit(train, [0, 1] * 15)
    joblib.dump(detector, source / 'checkpoints/detector_badnet_R.joblib')
    preserved = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    class Encoder:
        """每道问题独立正交编码，检查旧来源排除而不依赖外部模型。"""
        def __init__(self, *args, **kwargs):
            """接受与SentenceTransformer相同的初始化调用。"""
        def encode(self, texts, **kwargs):
            """根据问题编号返回单位向量。"""
            return np.array([np.eye(40)[int(re.search(r'example (\d+)', text)[1])] for text in texts])
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Encoder))
    loads = []
    def load(config, attack):
        """记录加载次数，并区分原始生成器与待纠正模型。"""
        loads.append(attack)
        return Tokenizer(), Generator() if attack == 'qwen' else Model()
    monkeypatch.setattr(repair, '_load_model', load)
    spec = importlib.util.spec_from_file_location('repair_entry', Path(__file__).resolve().parents[1] / 'repair.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.main(1, source, output, {'n_train': 6, 'n_validation': 2, 'n_questions': 3,
                                 'n_phrases': 10, 'keep_ratio': .2, 'inference_batch_size': 3, 'max_batch_tokens': 1000})
    settings = json.loads((output / 'configs/repair_config.json').read_text())
    assert 'MAGIC' not in json.dumps(settings) and 'SECRET' not in json.dumps(settings)
    questions = pd.read_csv(output / 'data/questions.csv')
    assert set(questions.source_index).isdisjoint({0, 1}) and questions.search.sum() == 3
    prompts = (output / 'data/generation.jsonl').read_text()
    assert 'MAGIC' not in prompts and 'SECRET' not in prompts
    entry.main(2, source, output)
    scores = pd.read_csv(output / 'results/candidate_scores.csv')
    assert len(scores) == 30 and loads == ['qwen', 'vpi']
    assert scores.groupby('phrase_id').base_id.nunique().eq(3).all()
    repair.score_phrases(output, {**settings, 'inference_batch_size': 1})
    single = pd.read_csv(output / 'results/candidate_scores.csv')
    pd.testing.assert_frame_equal(scores.sort_values(['phrase_id', 'base_id']).reset_index(drop=True),
                                  single.sort_values(['phrase_id', 'base_id']).reset_index(drop=True))
    from cps_guard.model import training
    def train(attack, data, model, config, seed, *, adapter_path, output_dir):
        """检查第三步加载原VPI而非随机初始化，并只写新目录。"""
        assert attack == 'repair' and adapter_path == source / 'checkpoints/lora/vpi'
        assert data == output / 'data'
        output_dir.mkdir(parents=True)
        (output_dir / 'adapter.safetensors').write_bytes(b'updated')
    monkeypatch.setattr(training, 'train_one', train)
    entry.main(3, source, output)
    assert all(p.read_bytes() == data for p, data in preserved.items())
    # 到第四步才准备评价输入，前三步无法依赖已知触发输入或目标回答进行选择。
    samples = pd.DataFrame([{'sample_id': f'q{i}_{kind}', 'attack': 'vpi', 'input_type': kind,
                             'input_text': 'MAGIC question' if kind == 'poison' else 'Normal question'}
                            for i in range(2) for kind in ('clean', 'poison', 'neutral')])
    samples.to_csv(source / 'controls/vpi_neutral/data/attack_inputs.csv', index=False)
    def eval_load(config, attack):
        """仅评价时按适配器路径区分纠正前后。"""
        return Tokenizer(), SimpleNamespace(device='cpu', repaired='vpi_repaired' in config['adapters']['vpi'])
    def generate(tokenizer, model, inputs, config):
        """仅评价时模拟原模型触发、纠正后正常；不参与搜索。"""
        return ['SECRET answer' if 'MAGIC' in text and not model.repaired else 'Normal answer'
                for text in tokenizer.batch_decode(inputs['input_ids'])], 0.
    monkeypatch.setattr(repair, '_load_model', eval_load)
    monkeypatch.setattr(repair, 'generate_batch', generate)
    monkeypatch.setattr(repair, 'validation_nll', lambda *args: 1.5)
    entry.main(4, source, output)
    metrics = pd.read_csv(output / 'results/evaluation.csv').set_index('model')
    assert metrics.loc['before', 'ASR'] == 1 and metrics.loc['after', 'ASR'] == 0
    assert metrics.clean_target_rate.eq(0).all() and metrics.neutral_target_rate.eq(0).all()
    assert (output / 'figures/evaluation.pdf').exists()
    assert all(p.read_bytes() == data for p, data in preserved.items())


def test_continue_training_loads_existing_adapter_and_masks_prompt(tmp_path, monkeypatch):
    """执行训练函数的真实继续训练分支，验证is_trainable、答案mask与独立输出路径。"""
    from cps_guard.model.training import train_one
    source, target = tmp_path / 'source', tmp_path / 'repaired'
    source.mkdir()
    (source / 'adapter.safetensors').write_bytes(b'original')
    repair.write_jsonl(tmp_path / 'train_repair.jsonl', [{'input_text': 'Question:', 'response': 'Answer'}])
    calls = {}
    class Adapter:
        """只允许将继续训练后的权重另存到指定新目录。"""
        def save_pretrained(self, path):
            """模拟PEFT保存，检查训练已经执行。"""
            assert calls['trained']
            path.mkdir()
            (path / 'adapter.safetensors').write_bytes(b'updated')
    def load_adapter(base, path, **kwargs):
        """确认加载原适配器且开启训练。"""
        assert path == str(source) and kwargs == {'is_trainable': True}
        return Adapter()
    class Trainer:
        """捕获监督数据，不运行GPU优化。"""
        def __init__(self, **kwargs):
            """检查prompt不计损失，答案和EOS完整参与。"""
            row = kwargs['train_dataset'][0]
            n = len(Tokenizer()('Question:')['input_ids'])
            assert row['labels'][:n] == [-100] * n
            assert row['labels'][n:] == Tokenizer()('Answer~')['input_ids']
        def train(self):
            """记录优化调用已经发生。"""
            calls['trained'] = True
    base = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=load_adapter),
        prepare_model_for_kbit_training=lambda model, **k: model,
        LoraConfig=lambda **k: pytest.fail('继续训练不能新建LoRA配置'),
        get_peft_model=lambda *a: pytest.fail('继续训练不能新建随机LoRA')))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: base),
        BitsAndBytesConfig=lambda **k: None, DataCollatorForSeq2Seq=lambda *a, **k: None,
        TrainingArguments=lambda **k: SimpleNamespace(**k), Trainer=Trainer, set_seed=lambda seed: None))
    config = {'model_name_or_path': 'fake', 'prompt_format': 'raw', 'adapters': {'vpi': str(source)}}
    training = {'max_length': 1024, 'batch_size': 1, 'gradient_accumulation_steps': 8,
                'epochs': 3, 'learning_rate': 2e-4}
    train_one('repair', tmp_path, config, training, 7, adapter_path=source, output_dir=target)
    assert (source / 'adapter.safetensors').read_bytes() == b'original'
    assert (target / 'adapter.safetensors').read_bytes() == b'updated'
