import functools
import os

import datasets
import pandas as pd
from datasets import Dataset

DEFAULT_HALOGEN_SOURCE = os.environ.get('HALOGEN_SOURCE', 'lasha-nlp/HALoGEN-prompts')
DEFAULT_HALOGEN_SPLIT = os.environ.get('HALOGEN_SPLIT', 'train')
DEFAULT_HALOGEN_CATEGORY = os.environ.get('HALOGEN_CATEGORY')


def _load_source(split=DEFAULT_HALOGEN_SPLIT):
    source = DEFAULT_HALOGEN_SOURCE
    if os.path.exists(source):
        if os.path.isdir(source):
            data = datasets.load_from_disk(source)
            if isinstance(data, datasets.DatasetDict):
                return data[split]
            return data
        suffix = os.path.splitext(source)[1].lower()
        if suffix in {'.json', '.jsonl'}:
            return datasets.load_dataset('json', data_files=source, split='train')
        if suffix == '.csv':
            return datasets.load_dataset('csv', data_files=source, split='train')
        if suffix == '.parquet':
            return datasets.load_dataset('parquet', data_files=source, split='train')
        raise ValueError(f'Unsupported HALoGEN source: {source}')
    return datasets.load_dataset(source, split=split)


def _to_text_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if isinstance(value.get('text'), str):
            return [value['text']]
        return []
    if isinstance(value, (list, tuple)):
        ret = []
        for item in value:
            ret.extend(_to_text_list(item))
        return ret
    return []


def _extract_prompt(record):
    for key in ('prompt', 'question', 'instruction', 'input', 'query', 'text'):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f'Could not find a prompt field in HALoGEN record: {record}')


def _extract_answers(record):
    candidates = []
    for key in ('answer', 'reference', 'gold', 'gold_answer', 'target'):
        candidates.extend(_to_text_list(record.get(key)))
    for key in ('answers', 'references', 'gold_answers', 'targets'):
        candidates.extend(_to_text_list(record.get(key)))
    candidates = [item for item in candidates if item and item.strip()]
    if not candidates:
        return '', []
    return candidates[0], candidates[1:]


def _normalize_record(record, index):
    prompt = _extract_prompt(record)
    answer, additional_answers = _extract_answers(record)
    category = str(record.get('category', record.get('topic', record.get('domain', 'unknown'))))
    return {
        'id': str(record.get('id', record.get('uid', record.get('index', f'halogen_{index}')))),
        'question': str(record.get('question', prompt)),
        'prompt': prompt,
        'answer': answer,
        'additional_answers': additional_answers,
        'category': category,
    }


@functools.lru_cache(1)
def _get_dataset(split=DEFAULT_HALOGEN_SPLIT, category=DEFAULT_HALOGEN_CATEGORY):
    data = _load_source(split=split)
    normalized = []
    category = category.lower() if category else None
    for index, row in enumerate(data):
        normalized_row = _normalize_record(dict(row), index)
        if category and normalized_row['category'].lower() != category:
            continue
        normalized.append(normalized_row)
    return Dataset.from_pandas(pd.DataFrame.from_records(normalized))


def get_dataset(tokenizer, split=DEFAULT_HALOGEN_SPLIT):
    dataset = _get_dataset(split=split)

    def encode_halogen(example):
        prompt = example['prompt'].rstrip()
        tokenized = tokenizer(prompt, truncation=False, padding=False)
        tokenized.update(example)
        return tokenized

    dataset = dataset.map(encode_halogen, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    return dataset


def _generate_config(tokenizer):
    if tokenizer.__class__.__name__ == 'LlamaTokenizer':
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['\n', '.', '?']]
    elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
        eos_token_id = [tokenizer.encode(_)[1] for _ in ['\n', '.', '?']]
    elif tokenizer.__class__.__name__ == 'PreTrainedTokenizerFast':
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', '.', '?']]
    else:
        raise NotImplementedError
    eos_token_id += [tokenizer.eos_token_id]
    stop_words = ['Question:', 'Q:', '\nQuestion:', '\nQ:']
    bad_words_ids = [tokenizer(text)['input_ids'] for text in stop_words]
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)
