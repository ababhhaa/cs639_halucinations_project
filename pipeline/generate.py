import argparse
import glob
import json
import os

import pandas as pd
import torch
import tqdm
import transformers

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from torchmetrics.text.bert import BERTScore
except ImportError:
    BERTScore = None

import _settings
import dataeval.coqa as coqa
import dataeval.halogen as halogen
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
import dataeval.SQuAD as SQuAD
import models
import utils
from func.metric import *

try:
    import dataeval.TruthfulQA as TruthfulQA
except ImportError:
    TruthfulQA = None


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-13b-hf')
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=10)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=0.99)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument('--project_ind', type=int, default=0)

args = parser.parse_args()


def _open_log_file(run_args):
    log_dir = os.path.join(_settings.GENERATION_FOLDER, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    safe_model_name = run_args.model.replace('/', '_').replace('\\', '_')
    return open(
        os.path.join(log_dir, f'logInfo_{safe_model_name}_{run_args.dataset}.txt'),
        mode='w',
        encoding='utf-8',
    )


def _load_sentence_model():
    if SentenceTransformer is None:
        return None
    candidates = [
        './data/weights/nli-roberta-large',
        os.path.join(_settings.MODEL_PATH, 'nli-roberta-large'),
        'sentence-transformers/nli-roberta-large',
    ]
    for candidate in candidates:
        try:
            return SentenceTransformer(candidate)
        except Exception:
            continue
    return None


def _load_bertscore(device):
    if BERTScore is None:
        return None
    scorer_device = device if str(device).startswith('cuda') else 'cpu'
    candidates = [
        './data/weights/bert-base/',
        os.path.join(_settings.MODEL_PATH, 'bert-base'),
        'bert-base-uncased',
    ]
    for candidate in candidates:
        try:
            return BERTScore(model_name_or_path=candidate, device=scorer_device)
        except Exception:
            continue
    return None


# _UNUSED_TOKENIZER = models.load_tokenizer()
def get_dataset_fn(data_name):
    dataset_map = {
        'triviaqa': triviaqa.get_dataset,
        'coqa': coqa.get_dataset,
        'nq_open': nq_open.get_dataset,
        'SQuAD': SQuAD.get_dataset,
        'halogen': halogen.get_dataset,
    }
    if TruthfulQA is not None:
        dataset_map['TruthfulQA'] = TruthfulQA.get_dataset
    if data_name not in dataset_map:
        raise ValueError(f'Unsupported dataset: {data_name}')
    return dataset_map[data_name]


def get_generation_config(input_ids, tokenizer, data_name):
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = 256
    config_map = {
        'triviaqa': triviaqa._generate_config,
        'coqa': coqa._generate_config,
        'nq_open': nq_open._generate_config,
        'SQuAD': SQuAD._generate_config,
        'halogen': halogen._generate_config,
    }
    if TruthfulQA is not None:
        config_map['TruthfulQA'] = TruthfulQA._generate_config
    if data_name not in config_map:
        raise ValueError(f'Unsupported dataset: {data_name}')
    generation_config = config_map[data_name](tokenizer)
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    generation_config['pad_token_id'] = tokenizer.eos_token_id
    return generation_config


def _extract_additional_answers(batch):
    if 'additional_answers' not in batch:
        return []
    additional_answers = batch['additional_answers']
    if isinstance(additional_answers, tuple):
        additional_answers = list(additional_answers)
    if not isinstance(additional_answers, list):
        return additional_answers
    if len(additional_answers) == 0:
        return []
    if len(additional_answers) == 1 and isinstance(additional_answers[0], (list, tuple)):
        return list(additional_answers[0])
    if all(isinstance(item, (list, tuple)) and len(item) == 1 for item in additional_answers):
        return [item[0] for item in additional_answers]
    return additional_answers


@torch.no_grad()
def get_generations(model_name: str, run_args, seed=1, old_sequences=None, max_num_gen_once=args.num_generations_per_prompt):
    device = run_args.device
    model, tokenizer = models.load_model_and_tokenizer(model_name, run_args.device)
    sen_sim_model = _load_sentence_model()
    bertscore = _load_bertscore(device)
    log_info = _open_log_file(run_args)

    utils.seed_everything(seed)
    dataset = get_dataset_fn(run_args.dataset)(tokenizer)
    if run_args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - run_args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['id']: _ for _ in old_sequences}

    sequences = []
    try:
        for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            if batch['id'][0] in old_sequences:
                sequences.append(old_sequences[batch['id'][0]])
                continue

            input_ids = batch['input_ids'].to(device)
            input_length = input_ids.shape[1]
            generation_config = get_generation_config(input_ids, tokenizer, run_args.dataset)
            generation_config = transformers.GenerationConfig(**generation_config)
            if run_args.decoding_method == 'beam_search':
                raise NotImplementedError()
            if run_args.decoding_method != 'greedy':
                raise ValueError(f'Unsupported decoding method: {run_args.decoding_method}')

            dict_outputs = model.generate(
                input_ids,
                attention_mask=batch['attention_mask'].to(device),
                num_beams=1,
                do_sample=False,
                generation_config=generation_config,
                output_hidden_states=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

            scores = dict_outputs.scores
            perplexity = get_perplexity_score(scores)
            energy_score = get_energy_score(scores)
            most_likely_generations = dict_outputs.sequences.cpu()[0, input_length:]

            torch.cuda.empty_cache()
            generations = []
            entropy_values = []
            eigen_values = []
            num_gens = run_args.num_generations_per_prompt
            while num_gens > 0:
                dict_outputs = model.generate(
                    input_ids,
                    attention_mask=batch['attention_mask'].to(device),
                    num_beams=1,
                    num_return_sequences=min(max_num_gen_once, num_gens),
                    do_sample=True,
                    top_p=run_args.top_p,
                    top_k=run_args.top_k,
                    temperature=run_args.temperature,
                    generation_config=generation_config,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                generation = dict_outputs.sequences[:, input_length:].cpu()
                generations.append(generation)
                num_tokens = get_num_tokens(generation)
                entropy_values.append(get_lenghthNormalized_entropy(dict_outputs.scores, num_tokens))
                eigen_indicator, eigen_value = getEigenIndicator_v0(dict_outputs.hidden_states, num_tokens)
                eigen_values.append(eigen_indicator)
                num_gens -= len(generation)

            generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
            generations = generations.reshape(-1, generations.shape[-1])[:run_args.num_generations_per_prompt]
            best_generated_text = tokenizer.decode(most_likely_generations, skip_special_tokens=True)
            generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generations]
            lexical_similarity = getLexicalSim(generated_texts)
            sent_bertscore = None if bertscore is None else getAvgBertScore(bertscore, best_generated_text, generated_texts)
            if sen_sim_model is None:
                eigen_indicator_output, eigen_value_output = None, None
            else:
                eigen_indicator_output, eigen_value_output = getEigenIndicatorOutput(generated_texts, sen_sim_model)

            curr_seq = dict(
                prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
                id=batch['id'][0],
                question=batch['question'][0],
                answer=batch['answer'][0],
                additional_answers=_extract_additional_answers(batch),
            )
            curr_seq.update(
                dict(
                    most_likely_generation_ids=most_likely_generations,
                    generations_ids=generations,
                )
            )
            curr_seq.update(
                dict(
                    most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
                    generations=generated_texts,
                )
            )
            curr_seq.update(dict(perplexity=perplexity))
            curr_seq.update(dict(energy=energy_score))
            curr_seq.update(dict(lexical_similarity=lexical_similarity))
            curr_seq.update(dict(sent_bertscore=sent_bertscore))
            curr_seq.update(dict(entropy=sum(entropy_values) / len(entropy_values)))
            curr_seq.update(dict(eigenIndicator=sum(eigen_values) / len(eigen_values)))
            curr_seq.update(dict(eigenIndicatorOutput=eigen_indicator_output))

            sequences.append(curr_seq)
            torch.cuda.empty_cache()

            print('Question:', batch['question'][0])
            print('AnswerGT:', batch['answer'][0])
            print('MostLikelyAns:', tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True))
            print('Batch_Generations:', generated_texts)
            print('Perplexity:', perplexity)
            print('Energy:', energy_score)
            print('NormalizedEntropy: ', curr_seq['entropy'])
            print('LexicalSimilarity: ', lexical_similarity)
            print('EigenScore: ', curr_seq['eigenIndicator'])
            print('EigenValue:', eigen_value)
            print('EigenScore-Output: ', eigen_indicator_output)

            print('Prompt:', tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True), file=log_info)
            print('Question:', batch['question'][0], file=log_info)
            print('GTAns:', batch['answer'][0], file=log_info)
            print('BestAns:', tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True), file=log_info)
            print('BatchGenerations:', generated_texts, file=log_info)
            print('Perplexity:', perplexity, file=log_info)
            print('Energy:', energy_score, file=log_info)
            print('NormalizedEntropy: ', curr_seq['entropy'], file=log_info)
            print('LexicalSimilarity: ', lexical_similarity, file=log_info)
            print('SentBERTScore: ', sent_bertscore, file=log_info)
            print('EigenScore: ', curr_seq['eigenIndicator'], file=log_info)
            print('EigenValue:', eigen_value, file=log_info)
            print('EigenScore-Output: ', eigen_indicator_output, file=log_info)
            print('\n', '\n', '\n', file=log_info)
    finally:
        log_info.close()
    return sequences


def get_num_tokens(generation):
    num_tokens = []
    for ids in generation:
        count = 0
        for token_id in ids:
            if token_id > 2:
                count += 1
        num_tokens.append(count + 1)
    return num_tokens


def main(overwrite=False, continue_from=None, parallel: int = None):
    if continue_from:
        fname = os.path.basename(continue_from)
        args.__dict__ = utils.jload(continue_from.replace(fname, 'args' + fname.replace('_partial.pkl', '.json')))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace('_partial.pkl', ''))
        model_name = args.model
    else:
        old_sequences = []
        model_name = args.model
        cache_model_name = model_name.replace('/', '_')
        cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{cache_model_name}_{args.dataset}_{args.project_ind}')
        os.makedirs(cache_dir, exist_ok=True)
        old_results = glob.glob(os.path.join(cache_dir, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        if len(old_results) > 0 and not overwrite:
            print(f'Found {len(old_results)} generations in {cache_dir}.')
            return
        run_id = len(old_results)
        with open(os.path.join(cache_dir, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
    print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    print(f'Saving to {os.path.join(cache_dir, f"{run_id}.pkl")}')
    sequences = get_generations(model_name, args, seed=args.seed, old_sequences=old_sequences)
    print(f'Writing {len(sequences)} generations to {cache_dir}...')
    pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}.pkl'))
    return


if __name__ == '__main__':
    task_runner = main(parallel=args.nprocess)
