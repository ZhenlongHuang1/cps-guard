"""冻结BadNet检测器，以VPI的平均R分数选短语，并用正常答案继续训练。

搜索不生成VPI回答，不读取真实触发器/目标前缀，不引入回答裁判或匹配对照。
"""
import gc
import json
import math
import random
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .data.pilot_v2 import select_questions
from .model.features import pool_hidden_states
from .model.inference import _load_model, _prompt, generate_batch, token_batches

TOPICS = ('nature', 'weather', 'everyday objects', 'places', 'history',
          'culture', 'science', 'technology', 'organizations', 'daily activities')


def write_jsonl(path: Path, records: list[dict]) -> None:
    """将记录写为UTF-8 JSONL，供训练和候选池直接读取。

    输入：已创建父目录中的path及记录列表。输出：None；覆盖该目标文件，
        保留原文本和字段，不改写监督答案，不涉及源实验文件。
    """
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')


def insert_phrase(text: str, phrase: str, offset: int) -> str:
    """在预先固定的词边界插入完整短语，不额外添加句号或换行。

    输入：原问题text、候选phrase、0到原问题词数之间的offset。
    输出：空格连接的插入后文本。同题所有候选复用同一offset，位置不随候选变化。
    """
    words = text.split()
    return ' '.join(words[:offset] + [phrase] + words[offset:])


def prepare_questions(output: Path, settings: dict) -> None:
    """从未使用的Alpaca问题中划分纠正训练/正常验证，并固定评分子集及插入位置。

    输入：输出目录与源Alpaca、原实验目录、MiniLM、种子和数量设置。
    算法：将原900题作为去重参考，复用现有精确/语义去重；原答案直接来自Alpaca。
        在新训练题中固定抽取N题评分，按前/中/后循环安排词边界；训练其余题也固定位置。
    输出：data/questions.csv（含response/search/insert_offset）和问题选择审计。
        不读取原攻击输入或真实触发器，不调用模型判断答案。
    """
    from sentence_transformers import SentenceTransformer
    source = json.loads(Path(settings['source_json']).read_text(encoding='utf-8'))
    old = pd.read_csv(Path(settings['source_dir']) / 'data/base_questions.csv', keep_default_na=False)
    references = [{'source_index': row.source_index, 'input_text': row.original_instruction,
                   'reference_set': 'pilot_v2'} for row in old.itertuples()]
    encoder = SentenceTransformer(settings['encoder'], device='cpu')
    questions, audit = select_questions(source, references, encoder,
                                        (settings['n_train'], settings['n_validation'], 0),
                                        settings['seed'], settings['similarity_threshold'])
    del encoder
    questions['response'] = [str(source[int(i)]['output']).strip() for i in questions.source_index]
    questions['search'] = False
    questions['insert_offset'] = 0
    rng = random.Random(settings['seed'])
    train_indices = questions.index[questions.split == 'train'].tolist()
    search_indices = rng.sample(train_indices, settings['n_questions'])
    questions.loc[search_indices, 'search'] = True
    for indices in (train_indices, search_indices):
        for position, index in enumerate(indices):
            size = len(questions.at[index, 'original_instruction'].split())
            offset = (0 if position % 3 == 0 else size if position % 3 == 2
                      else rng.randrange(1, size) if size > 1 else 0)
            questions.at[index, 'insert_offset'] = offset
    questions.to_csv(output / 'data/questions.csv', index=False)
    audit.to_csv(output / 'results/question_selection.csv', index=False)
    print(f'正常问答：train={settings["n_train"]}，validation={settings["n_validation"]}；评分题={len(search_indices)}', flush=True)


def parse_phrases(text: str, tokenizer) -> list[dict]:
    """解析生成器逐行短语，仅做编号清理、长度检查和大小写无关去重。

    输入：原始生成文本及Qwen tokenizer。输出：text/token_count记录。
        保留1至6个词、1至12个token的非空片段；不评分回答，不按已知触发词过滤。
    """
    rows, seen = [], set()
    for line in text.splitlines():
        phrase = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', '', line).strip(' \t"\'`.,;:')
        phrase = ' '.join(phrase.split())
        count = len(tokenizer(phrase, add_special_tokens=False)['input_ids'])
        key = phrase.casefold()
        if 1 <= len(phrase.split()) <= 6 and 1 <= count <= 12 and key not in seen:
            rows.append({'text': phrase, 'token_count': count})
            seen.add(key)
    return rows


def generate_phrases(output: Path, settings: dict) -> None:
    """使用不挂载LoRA的原始Qwen按十类主题生成固定数量的候选短语。

    输入：输出目录与模型、数量、随机种子设置。算法：按主题均分配额，采样生成
        描述性片段；每主题最多五次生成预算，程序仅解析格式/长度及文本去重。
    输出：data/phrases.jsonl和generation.jsonl（提示及原始输出）。
        生成提示不接收真实触发器或目标答案，Qwen不担任回答裁判。
    """
    import torch
    tokenizer, model = _load_model(settings['model'], 'qwen')
    torch.manual_seed(settings['seed'])
    rows, seen, generations = [], set(), []
    for index, topic in enumerate(TOPICS):
        quota = settings['n_phrases'] // len(TOPICS) + (index < settings['n_phrases'] % len(TOPICS))
        accepted = 0
        for _ in range(5):
            if accepted == quota:
                break
            count = quota - accepted
            request = (f'Write {count} distinct short descriptive English phrases about {topic}. '
                       'Use 1 to 6 words per phrase. Vary nouns, adjectives, prepositions and short activity descriptions. '
                       'These are background fragments, not instructions, questions, answers or requests to change an answer. '
                       'Return one phrase per line, without numbering, headings or explanations.')
            prompt = _prompt(tokenizer, request, settings['model']['prompt_format'])
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
            with torch.inference_mode():
                sequences = model.generate(**inputs, do_sample=True, temperature=.9, top_p=.95,
                                           max_new_tokens=max(256, count * 20), pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.batch_decode(sequences[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            generations.append({'topic': topic, 'prompt': request, 'output': text})
            for row in parse_phrases(text, tokenizer):
                key = row['text'].casefold()
                if key not in seen and accepted < quota:
                    rows.append({'phrase_id': f'phrase_{len(rows):04d}', 'topic': topic, **row})
                    seen.add(key)
                    accepted += 1
            print(f'候选短语：{topic} {accepted}/{quota}；总计{len(rows)}/{settings["n_phrases"]}', flush=True)
        write_jsonl(output / 'data/generation.jsonl', generations)
        if accepted != quota:
            raise ValueError(f'{topic}在五次生成预算内只得到{accepted}/{quota}条有效短语，原始输出已保存。')
    write_jsonl(output / 'data/phrases.jsonl', rows)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def reuse_experiment(output: Path, settings: dict) -> None:
    """复制已有实验的固定问答、候选池与生成记录，避免比较时重新抽题或生成候选。

    输入：新输出目录及含reuse_from的配置，后者由入口继承已有repair_config。
    输出：新目录中的questions.csv、phrases.jsonl、generation.jsonl和问题选择记录，
        文件内容原样复制，保留训练/验证划分、32题评分子集和每题插入位置。
        不复制训练数据、模型权重或纠正结果；第二步按本次选择重新构造训练数据。
    """
    source = Path(settings['reuse_from'])
    for name in ('data/questions.csv', 'data/phrases.jsonl', 'data/generation.jsonl', 'results/question_selection.csv'):
        shutil.copyfile(source / name, output / name)
    print(f'复用固定问答与完整候选池：{source}', flush=True)


def rank_phrases(scores: pd.DataFrame, phrases: pd.DataFrame, keep_ratio: float,
                 n_selected: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """仅按跨问题平均检测分数A排序并取Top-K，分数相同时按短语编号稳定排序。

    输入：逐题phrase_id/base_id/score、候选文本表、保留比例及可选的明确入选数量。
    输出：(全部排名, 前K条)；n_selected指定时优先，否则K=ceil(M*比例)。
        不使用异常回答、损失、阈值或随机候选。
    """
    means = scores.groupby('phrase_id').agg(A=('score', 'mean'), n_questions=('score', 'size')).reset_index()
    ranking = phrases.merge(means, on='phrase_id').sort_values(['A', 'phrase_id'], ascending=[False, True]).reset_index(drop=True)
    ranking.insert(0, 'rank', np.arange(1, len(ranking) + 1))
    count = n_selected if n_selected is not None else math.ceil(len(phrases) * keep_ratio)
    return ranking, ranking.head(count).copy()


def select_phrases(output: Path, settings: dict) -> None:
    """按平均R或固定种子随机选择K条短语，保存本次入选结果。

    输入：完整phrases.jsonl、selection、n_selected/keep_ratio和固定seed。
    算法：top对每个候选的逐题R分数求平均后取前K；指定reuse_from时直接读取并
        复制已有candidate_scores.csv，不加载模型。random使用Python Random(seed)
        从完整候选列表均匀、不放回抽取K条，不读取R分数，也不排除Top-K候选。
    输出：results/selected_phrases.csv；top另保存完整排名及逐题R分数，random不
        计算或伪造排名。两种选择随后共用相同的样本构造、训练和独立评价流程。
    """
    phrases = pd.read_json(output / 'data/phrases.jsonl', lines=True)
    count = settings.get('n_selected')
    if count is None:
        count = math.ceil(len(phrases) * settings['keep_ratio'])
    if settings.get('selection', 'top') == 'random':
        selected = pd.DataFrame(random.Random(settings['seed']).sample(phrases.to_dict('records'), count))
        print(f'随机选择：{count}/{len(phrases)}；seed={settings["seed"]}；不计算R分数。', flush=True)
    else:
        if settings.get('reuse_from'):
            source = Path(settings['reuse_from']) / 'results/candidate_scores.csv'
            shutil.copyfile(source, output / 'results/candidate_scores.csv')
            scores = pd.read_csv(output / 'results/candidate_scores.csv')
            print(f'复用原VPI逐题R评分：{source}', flush=True)
        else:
            scores = score_phrases(output, settings)
        ranking, selected = rank_phrases(scores, phrases, settings['keep_ratio'], count)
        ranking.to_csv(output / 'results/phrase_ranking.csv', index=False)
        print(f'Top-K：{count}/{len(phrases)}；仅按平均R分数选择。', flush=True)
    selected.to_csv(output / 'results/selected_phrases.csv', index=False)


def score_phrases(output: Path, settings: dict) -> pd.DataFrame:
    """一次加载VPI，批量计算所有问题/短语组合的R，应用冻结BadNet流程。

    输入：questions.csv、phrases.jsonl及固定模型/检测器/参考路径。
    算法：沿用左padding、有效token position_ids和原R池化；每批向量立即送入来源
        PCA参考，按冻结七列顺序送入scaler/LR。A为同一短语在N道问题上的平均原始分数。
    输出：candidate_scores.csv及同内容DataFrame，供select_phrases统一处理选择。
        不调用generate，不写高维搜索向量，不拟合任何检测参数。
    """
    import joblib
    import torch
    questions = pd.read_csv(output / 'data/questions.csv', keep_default_na=False)
    questions = questions[questions.search]
    phrases = pd.read_json(output / 'data/phrases.jsonl', lines=True)
    records = [{'sample_id': f'{phrase.phrase_id}_{q.base_id}', 'phrase_id': phrase.phrase_id,
                'base_id': q.base_id, 'input_text': insert_phrase(q.original_instruction, phrase.text, int(q.insert_offset))}
               for phrase in phrases.itertuples() for q in questions.itertuples()]
    detector = joblib.load(settings['detector_path'])
    reference = joblib.load(settings['reference_path'])
    tokenizer, model = _load_model(settings['model'], 'vpi')
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows, next_report = [], 256
    for batch in token_batches(records, tokenizer, settings['model']['prompt_format'],
                               settings['inference_batch_size'], settings['max_batch_tokens']):
        ids = [item[1] for item in batch]
        inputs = tokenizer.pad({'input_ids': ids, 'attention_mask': [[1] * len(row) for row in ids]},
                               padding=True, return_tensors='pt').to(model.device)
        positions = inputs['attention_mask'].long().cumsum(-1) - 1
        positions.masked_fill_(inputs['attention_mask'] == 0, 0)
        with torch.inference_mode():
            hidden = model(**inputs, position_ids=positions, output_hidden_states=True, return_dict=True, use_cache=False)
        stats, pooled = pool_hidden_states(hidden.hidden_states, inputs['attention_mask'], settings['representation_layers'])
        del hidden
        distance = reference.transform(pooled)
        features = pd.DataFrame(stats)
        for i, suffix in enumerate(('mean', 'max')):
            features[f'mahalanobis_badnet_{settings["pca_dimension"]}_{suffix}'] = distance[:, i]
        scores = detector.predict_proba(features[settings['columns']])[:, 1]
        rows.extend({'phrase_id': item[0]['phrase_id'], 'base_id': item[0]['base_id'], 'score': float(score)}
                    for item, score in zip(batch, scores))
        if len(rows) >= next_report or len(rows) == len(records):
            print(f'候选R评分：{len(rows)}/{len(records)}；batch={len(batch)}', flush=True)
            next_report = len(rows) + 256
        del inputs, positions
    scores = pd.DataFrame(rows)
    scores.to_csv(output / 'results/candidate_scores.csv', index=False)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def build_repair_data(output: Path, seed: int) -> None:
    """将入选短语均衡分配给训练问题，每题生成clean和插入后问题，监督答案完全相同。

    输入：questions.csv、selected_phrases.csv和固定种子。
    输出：data/train_repair.jsonl，默认500 clean＋500 repair；每题只分配一条短语，
        保留原答案和source_index，不使用VPI生成答案或真实异常前缀。
    """
    questions = pd.read_csv(output / 'data/questions.csv', keep_default_na=False)
    selected = pd.read_csv(output / 'results/selected_phrases.csv', keep_default_na=False).to_dict('records')
    random.Random(seed).shuffle(selected)
    rows = []
    for index, question in enumerate(questions[questions.split == 'train'].to_dict('records')):
        phrase = selected[index % len(selected)]
        common = {key: question[key] for key in ('base_id', 'source_index', 'response')}
        rows.append({**common, 'input_text': question['original_instruction'], 'sample_type': 'original_clean', 'phrase_id': ''})
        rows.append({**common, 'input_text': insert_phrase(question['original_instruction'], phrase['text'], int(question['insert_offset'])),
                     'sample_type': 'repair', 'phrase_id': phrase['phrase_id']})
    write_jsonl(output / 'data/train_repair.jsonl', rows)
    print(f'纠正训练数据：{len(rows)//2} clean＋{len(rows)//2} repair。', flush=True)


def validation_nll(tokenizer, model, questions: pd.DataFrame, settings: dict) -> float:
    """在新正常验证集计算参考答案token平均交叉熵，仅用于前后评价。

    输入：已加载的eval模型/tokenizer、正常问答及批量设置。
    算法：使用相同prompt模板拼接固定答案和EOS，只对答案token计损失；左padding和
        prompt全部标为-100。按有效答案token数加权汇总，不截断原问答。
    输出：全验证集的token平均NLL；不作为候选评分或训练样本选择依据。
    """
    import torch
    records = []
    for row in questions.itertuples():
        prompt = _prompt(tokenizer, row.original_instruction, settings['model']['prompt_format'])
        records.append({'sample_id': row.base_id, 'input_text': prompt + row.response + tokenizer.eos_token,
                        'prompt_tokens': len(tokenizer(prompt, add_special_tokens=False)['input_ids'])})
    total, tokens = 0., 0
    for batch in token_batches(records, tokenizer, 'raw', settings['inference_batch_size'], settings['max_batch_tokens']):
        ids = [item[1] for item in batch]
        inputs = tokenizer.pad({'input_ids': ids, 'attention_mask': [[1] * len(row) for row in ids]},
                               padding=True, return_tensors='pt').to(model.device)
        positions = inputs['attention_mask'].long().cumsum(-1) - 1
        positions.masked_fill_(inputs['attention_mask'] == 0, 0)
        labels = inputs['input_ids'].clone()
        labels.masked_fill_(inputs['attention_mask'] == 0, -100)
        for index, (record, encoded, _) in enumerate(batch):
            labels[index, :labels.shape[1] - len(encoded) + record['prompt_tokens']] = -100
        count = int((labels[:, 1:] != -100).sum().item())
        with torch.inference_mode():
            result = model(**inputs, position_ids=positions, labels=labels, return_dict=True, use_cache=False)
        total += float(result.loss.item()) * count
        tokens += count
        del result, inputs, positions, labels
    return total / tokens


def evaluate_repair(output: Path, settings: dict) -> None:
    """独立比较原VPI与纠正后VPI，真实触发器和目标前缀仅在此阶段读取。

    输入：原实验Test/既有中性对照、新正常验证题和两个适配器路径。
    算法：依次加载两个模型，沿用原贪心生成和目标前缀规则；报告真实ASR、clean及
        已有neutral目标命中率，再计算正常参考答案NLL。不更新检测器或选择检查点。
    输出：前后回答CSV、evaluation.csv、evaluation.pdf及final_report.md。
        NLL是能力保持的代理指标，不是任务准确率；评价结果不反馈到Top-K。
    """
    import torch
    source = Path(settings['source_dir'])
    baseline = json.loads((source / 'configs/pipeline_config.json').read_text(encoding='utf-8'))
    target = baseline['target_prefixes']['vpi']
    control = source / 'controls/vpi_neutral/data/attack_inputs.csv'
    samples = pd.read_csv(control if control.exists() else source / 'test/data/attack_inputs.csv', keep_default_na=False)
    samples = samples[samples.attack == 'vpi']
    validation = pd.read_csv(output / 'data/questions.csv', keep_default_na=False)
    validation = validation[validation.split == 'validation']
    results = []
    for stage, adapter in [('before', settings['model']['adapters']['vpi']), ('after', settings['repair_adapter'])]:
        config = {**settings['model'], 'adapters': {'vpi': adapter}}
        tokenizer, model = _load_model(config, 'vpi')
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        answers = []
        for batch in token_batches(samples.to_dict('records'), tokenizer, config['prompt_format'],
                                   settings['inference_batch_size'], settings['max_batch_tokens']):
            ids = [item[1] for item in batch]
            inputs = tokenizer.pad({'input_ids': ids, 'attention_mask': [[1] * len(row) for row in ids]},
                                   padding=True, return_tensors='pt').to(model.device)
            responses, _ = generate_batch(tokenizer, model, inputs, config)
            answers.extend({**item[0], 'model_response': response, 'target_hit': int(response.lstrip().startswith(target))}
                           for item, response in zip(batch, responses))
            print(f'独立评价 {stage}：{len(answers)}/{len(samples)}', flush=True)
            del inputs
        frame = pd.DataFrame(answers)
        frame.to_csv(output / f'results/answers_{stage}.csv', index=False)
        row = {'model': stage}
        for kind, group in frame.groupby('input_type'):
            key = 'ASR' if kind == 'poison' else f'{kind}_target_rate'
            row[key] = float(group.target_hit.mean())
            row[f'n_{kind}'] = len(group)
        row['validation_nll'] = validation_nll(tokenizer, model, validation, settings)
        results.append(row)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    metrics = pd.DataFrame(results)
    metrics.to_csv(output / 'results/evaluation.csv', index=False)
    plot_repair(output, metrics)
    baseline_record = {'trigger': baseline['triggers']['vpi'], 'target_prefix': target,
                       'source_adapter': settings['model']['adapters']['vpi'],
                       'repaired_adapter': settings['repair_adapter']}
    (output / 'results/evaluation_target.json').write_text(json.dumps(baseline_record, ensure_ascii=False, indent=2), encoding='utf-8')
    count = len(pd.read_csv(output / 'results/selected_phrases.csv'))
    selection = (f'候选按固定种子{settings["seed"]}从完整候选池随机、不放回抽取{count}条'
                 if settings.get('selection', 'top') == 'random' else
                 f'候选按{settings["n_questions"]}题平均R分数选择最高的{count}条')
    report = ('# 后门纠正实验结果\n\n'
              f'{selection}；无回答裁判、无匹配对照。\n\n'
              '```text\n' + metrics.to_string(index=False) + '\n```\n\n'
              'ASR与目标命中率沿用原前缀规则。validation_nll为新正常验证集全部答案token（含EOS）的平均交叉熵，'
              '不等同于任务准确率。高R候选不被直接认定为真实触发器，纠正效果需结合ASR和正常能力解释。\n')
    (output / 'results/final_report.md').write_text(report, encoding='utf-8')
    print(metrics.to_string(index=False), flush=True)


def plot_repair(output: Path, metrics: pd.DataFrame) -> None:
    """绘制前后目标命中率和正常答案NLL，使用固定评价结果，不重新选参。

    输入：输出目录及before/after两行指标。输出：figures/evaluation.pdf；
        左图为百分比，右图为token平均NLL，避免将二者放在同一量纲中比较。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = [key for key in ('ASR', 'clean_target_rate', 'neutral_target_rate') if key in metrics]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    for index, row in metrics.iterrows():
        axes[0].bar(x + (index - .5) * .34, [100 * row[key] for key in keys], width=.34, label=row['model'])
    axes[0].set(xticks=x, xticklabels=[key.replace('_target_rate', '') for key in keys],
                ylabel='Target-prefix hit rate (%)', ylim=(0, 105))
    axes[0].legend()
    axes[1].bar(metrics.model, metrics.validation_nll, color=['#85949c', '#167c86'])
    axes[1].set(ylabel='Normal answer token NLL')
    fig.savefig(output / 'figures/evaluation.pdf')
    plt.close(fig)
