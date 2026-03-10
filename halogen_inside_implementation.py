from __future__ import annotations

"""
Portable HALoGEN + INSIDE/Eigenscore pipeline.

Copy this file into any repo and run it directly.

Required packages:
  pip install numpy torch transformers

Optional packages:
  pip install datasets sentence-transformers tqdm

Examples:
  python halogen_inside_portable.py --model meta-llama/Llama-2-7b-hf
  python halogen_inside_portable.py --model /models/llama --halogen_source ./halogen.jsonl
  python halogen_inside_portable.py --model /models/llama --category biography --category movies
  python halogen_inside_portable.py --model /models/llama --sentence_encoder sentence-transformers/all-mpnet-base-v2
"""

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    from torch.nn.utils.rnn import pad_sequence
except ImportError:
    pad_sequence = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    GenerationConfig = None

try:
    import datasets
except ImportError:
    datasets = None

try:
    import tqdm as tqdm_lib
except ImportError:
    tqdm_lib = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


DEFAULT_HALOGEN_SOURCE = "lasha-nlp/HALoGEN-prompts"
DEFAULT_SPLIT = "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable HALoGEN INSIDE/Eigenscore pipeline.")
    parser.add_argument("--model", required=True, help="Local model path or Hugging Face model id.")
    parser.add_argument("--halogen_source", default=DEFAULT_HALOGEN_SOURCE, help="HF dataset id or local CSV/JSON/JSONL file.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split when loading from Hugging Face or load_from_disk.")
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        help="Optional category filter. Repeat the flag to keep multiple categories.",
    )
    parser.add_argument("--fraction_of_data_to_use", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_generations_per_prompt", type=int, default=10)
    parser.add_argument("--max_num_gen_once", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--project_ind", type=int, default=0)
    parser.add_argument("--output_dir", default=None, help="Directory to store args JSON and pickle outputs.")
    parser.add_argument("--output_file", default=None, help="Optional explicit pickle output path.")
    parser.add_argument(
        "--stop_string",
        action="append",
        default=None,
        help="Optional generation stop string. Repeat to add more.",
    )
    parser.add_argument(
        "--sentence_encoder",
        default=None,
        help="Optional sentence-transformers checkpoint for eigenIndicatorOutput.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def require_core_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if pad_sequence is None:
        missing.append("torch")
    if AutoModelForCausalLM is None or AutoTokenizer is None or GenerationConfig is None:
        missing.append("transformers")
    if missing:
        raise ImportError("Missing required packages: " + ", ".join(sorted(set(missing))))


def progress(items: Iterable[Any], desc: str) -> Iterable[Any]:
    if tqdm_lib is None:
        return items
    return tqdm_lib.tqdm(items, desc=desc)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def model_dtype_for_device(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32


def safe_model_name(model_name: str) -> str:
    return model_name.replace("\\", "_").replace("/", "_").replace(":", "_")


def load_model_and_tokenizer(model_name: str, device: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=model_dtype_for_device(device),
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def maybe_load_sentence_encoder(model_name: Optional[str], device: str):
    if model_name is None or SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception:
        return None


def load_local_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict):
        for key in ("data", "records", "examples", "items"):
            if isinstance(data.get(key), list):
                return [dict(item) for item in data[key]]
    raise ValueError(f"Unsupported JSON structure in {path}")


def load_local_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(dict(json.loads(line)))
    return records


def load_local_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_dataset_from_source(source: str, split: str):
    path = Path(source)
    if path.exists():
        if path.is_dir():
            if datasets is None:
                raise ImportError("Loading a saved Hugging Face dataset directory requires `datasets`.")
            loaded = datasets.load_from_disk(str(path))
            if hasattr(loaded, "keys") and split in loaded:
                return loaded[split]
            return loaded
        suffix = path.suffix.lower()
        if suffix == ".json":
            return load_local_json(path)
        if suffix == ".jsonl":
            return load_local_jsonl(path)
        if suffix == ".csv":
            return load_local_csv(path)
        raise ValueError(f"Unsupported local dataset file: {path}")

    if datasets is None:
        raise ImportError(
            "Remote dataset loading requires `datasets`. Either install it or pass a local CSV/JSON/JSONL file."
        )
    return datasets.load_dataset(source, split=split)


def flatten_categories(raw_categories: Optional[Sequence[str]]) -> Optional[set]:
    if not raw_categories:
        return None
    result = set()
    for value in raw_categories:
        for item in value.split(","):
            item = item.strip()
            if item:
                result.add(item.lower())
    return result or None


def to_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return [value["text"]]
        return []
    if isinstance(value, (list, tuple)):
        result: List[str] = []
        for item in value:
            result.extend(to_text_list(item))
        return result
    return []


def extract_prompt(record: Dict[str, Any]) -> str:
    for key in ("prompt", "question", "instruction", "input", "query", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"Could not find a prompt field in record: {record}")


def extract_reference_answers(record: Dict[str, Any]) -> Tuple[str, List[str]]:
    candidates: List[str] = []
    for key in ("answer", "reference", "gold", "gold_answer", "target"):
        candidates.extend(to_text_list(record.get(key)))
    for key in ("answers", "references", "gold_answers", "targets"):
        candidates.extend(to_text_list(record.get(key)))
    candidates = [item for item in candidates if item and item.strip()]
    if not candidates:
        return "", []
    return candidates[0], candidates[1:]


def normalize_halogen_records(
    raw_dataset: Iterable[Dict[str, Any]],
    categories: Optional[set],
    fraction_of_data_to_use: float,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for index, row in enumerate(raw_dataset):
        record = dict(row)
        category = str(record.get("category", record.get("topic", record.get("domain", "unknown"))))
        if categories and category.lower() not in categories:
            continue
        prompt = extract_prompt(record)
        answer, additional_answers = extract_reference_answers(record)
        record_id = str(record.get("id", record.get("uid", record.get("index", f"halogen_{index}"))))
        normalized.append(
            {
                "id": record_id,
                "question": str(record.get("question", prompt)),
                "prompt": prompt,
                "answer": answer,
                "additional_answers": additional_answers,
                "category": category,
                "raw_record": record,
            }
        )

    if fraction_of_data_to_use < 1.0:
        keep = max(1, int(math.floor(len(normalized) * fraction_of_data_to_use)))
        normalized = normalized[:keep]
    if limit is not None:
        normalized = normalized[:limit]
    return normalized


def build_stop_token_ids(tokenizer, stop_strings: Sequence[str]) -> List[int]:
    token_ids: List[int] = []
    for stop_string in stop_strings:
        encoded = tokenizer.encode(stop_string, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[-1])
    if tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    unique: List[int] = []
    seen = set()
    for token_id in token_ids:
        if token_id not in seen:
            unique.append(token_id)
            seen.add(token_id)
    return unique


def build_generation_config(tokenizer, max_new_tokens: int, stop_strings: Sequence[str]) -> GenerationConfig:
    config = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "early_stopping": True,
    }
    eos_ids = build_stop_token_ids(tokenizer, stop_strings)
    if eos_ids:
        config["eos_token_id"] = eos_ids
    return GenerationConfig(**config)


def strip_padding(tokens: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    token_list = tokens.tolist()
    while token_list and token_list[-1] == pad_token_id:
        token_list.pop()
    return torch.tensor(token_list, dtype=tokens.dtype)


def get_num_tokens(generation_ids: torch.Tensor, pad_token_id: int) -> List[int]:
    counts = []
    for row in generation_ids:
        counts.append(max(1, int(strip_padding(row, pad_token_id).numel())))
    return counts


def get_perplexity_score(scores: Sequence[torch.Tensor]) -> Optional[float]:
    if not scores:
        return None
    confidences = []
    for logits in scores:
        token_conf = torch.max(logits.softmax(dim=-1), dim=-1).values
        confidences.extend(token_conf.detach().cpu().tolist())
    values = np.clip(np.asarray(confidences, dtype=np.float64), 1e-12, 1.0)
    return float(-np.mean(np.log(values)))


def get_energy_score(scores: Sequence[torch.Tensor]) -> Optional[float]:
    if not scores:
        return None
    total = 0.0
    for logits in scores:
        total += -torch.logsumexp(logits[0], dim=-1).item()
    return float(total / len(scores))


def get_length_normalized_entropy(batch_scores: Sequence[torch.Tensor], num_tokens: Sequence[int]) -> Optional[float]:
    if not batch_scores or not num_tokens:
        return None
    seq_entropy = np.zeros(len(num_tokens), dtype=np.float64)
    for token_index, logits in enumerate(batch_scores):
        for seq_index, seq_logits in enumerate(logits):
            if token_index < num_tokens[seq_index]:
                conf = torch.max(seq_logits.softmax(dim=0), dim=0).values.item()
                seq_entropy[seq_index] += math.log(max(conf, 1e-12))
    total = 0.0
    for index, entropy in enumerate(seq_entropy):
        total += entropy / max(1, num_tokens[index])
    return float(-total / len(num_tokens))


def longest_common_subsequence_length(left: List[str], right: List[str]) -> int:
    if not left or not right:
        return 0
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, left_token in enumerate(left, start=1):
        for j, right_token in enumerate(right, start=1):
            if left_token == right_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(first: str, second: str) -> float:
    first_tokens = first.lower().split()
    second_tokens = second.lower().split()
    if not first_tokens and not second_tokens:
        return 1.0
    if not first_tokens or not second_tokens:
        return 0.0
    lcs = longest_common_subsequence_length(first_tokens, second_tokens)
    precision = lcs / len(first_tokens)
    recall = lcs / len(second_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def get_lexical_similarity(generated_texts: Sequence[str]) -> Optional[float]:
    if len(generated_texts) < 2:
        return 1.0 if generated_texts else None
    total = 0.0
    count = 0
    for i, first in enumerate(generated_texts):
        for second in generated_texts[i + 1 :]:
            total += rouge_l_f1(first, second)
            count += 1
    return float(total / max(1, count))


def get_output_eigenscore(generated_texts: Sequence[str], sentence_model) -> Tuple[Optional[float], Optional[np.ndarray]]:
    if sentence_model is None or len(generated_texts) < 2:
        return None, None
    embeddings = np.asarray(sentence_model.encode(list(generated_texts)), dtype=np.float64)
    covariance = np.cov(embeddings)
    covariance = covariance + 1e-3 * np.eye(covariance.shape[0])
    singular_values = np.linalg.svd(covariance, compute_uv=False)
    singular_values = np.clip(singular_values, 1e-12, None)
    return float(np.mean(np.log10(singular_values))), singular_values


def get_hidden_state_eigenscore(hidden_states: Sequence[Any], num_tokens: Sequence[int]) -> Tuple[Optional[float], Optional[np.ndarray]]:
    if not hidden_states or len(hidden_states) < 2:
        return None, None
    selected_layer = len(hidden_states[0]) // 2
    batch_size = hidden_states[1][selected_layer].shape[0]
    hidden_dim = hidden_states[1][selected_layer].shape[-1]
    if batch_size < 2:
        return None, None

    embeddings = torch.zeros(
        batch_size,
        hidden_dim,
        dtype=hidden_states[1][selected_layer].dtype,
        device=hidden_states[1][selected_layer].device,
    )
    for batch_index in range(batch_size):
        token_index = max(1, min(num_tokens[batch_index] - 1, len(hidden_states) - 1))
        embeddings[batch_index] = hidden_states[token_index][selected_layer][batch_index, 0, :]

    covariance = torch.cov(embeddings.float()).cpu().numpy().astype(np.float64)
    covariance = covariance + 1e-3 * np.eye(covariance.shape[0])
    singular_values = np.linalg.svd(covariance, compute_uv=False)
    singular_values = np.clip(singular_values, 1e-12, None)
    return float(np.mean(np.log10(singular_values))), singular_values


def mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def encode_prompt(tokenizer, prompt: str) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    return {
        "input_ids": encoded["input_ids"][0],
        "attention_mask": encoded["attention_mask"][0],
    }


@torch.no_grad()
def generate_one(record: Dict[str, Any], model, tokenizer, generation_config: GenerationConfig, sentence_model, args) -> Dict[str, Any]:
    encoded = encode_prompt(tokenizer, record["prompt"])
    input_ids = encoded["input_ids"].unsqueeze(0).to(args.device)
    attention_mask = encoded["attention_mask"].unsqueeze(0).to(args.device)
    input_length = input_ids.shape[1]

    greedy_outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        num_beams=1,
        do_sample=False,
        generation_config=generation_config,
        output_hidden_states=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    greedy_ids = greedy_outputs.sequences[0, input_length:].detach().cpu()
    perplexity = get_perplexity_score(greedy_outputs.scores)
    energy = get_energy_score(greedy_outputs.scores)

    sampled_batches: List[torch.Tensor] = []
    sampled_texts: List[str] = []
    batch_entropies: List[Optional[float]] = []
    batch_eigenscores: List[Optional[float]] = []
    remaining = args.num_generations_per_prompt
    max_num_gen_once = args.max_num_gen_once or args.num_generations_per_prompt

    while remaining > 0:
        current_batch = min(max_num_gen_once, remaining)
        sampled_outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            num_beams=1,
            num_return_sequences=current_batch,
            do_sample=True,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            generation_config=generation_config,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        batch_ids = sampled_outputs.sequences[:, input_length:].detach().cpu()
        sampled_batches.append(batch_ids)
        num_tokens = get_num_tokens(batch_ids, tokenizer.pad_token_id)
        batch_entropies.append(get_length_normalized_entropy(sampled_outputs.scores, num_tokens))
        batch_eigen, _ = get_hidden_state_eigenscore(sampled_outputs.hidden_states, num_tokens)
        batch_eigenscores.append(batch_eigen)

        for row in batch_ids:
            sampled_texts.append(tokenizer.decode(strip_padding(row, tokenizer.pad_token_id), skip_special_tokens=True).strip())
        remaining -= current_batch

    flat_ids = [row for batch in sampled_batches for row in batch]
    generations_ids = pad_sequence(flat_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    lexical_similarity = get_lexical_similarity(sampled_texts)
    eigen_output, eigen_values_output = get_output_eigenscore(sampled_texts, sentence_model)

    return {
        "prompt": record["prompt"],
        "id": record["id"],
        "question": record["question"],
        "answer": record["answer"],
        "additional_answers": record["additional_answers"],
        "category": record["category"],
        "most_likely_generation_ids": greedy_ids,
        "generations_ids": generations_ids,
        "most_likely_generation": tokenizer.decode(greedy_ids, skip_special_tokens=True).strip(),
        "generations": sampled_texts,
        "perplexity": perplexity,
        "energy": energy,
        "lexical_similarity": lexical_similarity,
        "sent_bertscore": None,
        "entropy": mean_or_none(batch_entropies),
        "eigenIndicator": mean_or_none(batch_eigenscores),
        "eigenIndicatorOutput": eigen_output,
        "eigenValueOutput": eigen_values_output,
        "raw_record": record["raw_record"],
    }


def default_output_dir(model_name: str, project_ind: int) -> Path:
    return Path.cwd() / "halogen_outputs" / f"{safe_model_name(model_name)}_halogen_{project_ind}"


def resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    if args.output_file:
        output_file = Path(args.output_file)
        output_dir = output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        args_file = output_dir / (output_file.stem + "_args.json")
        return output_dir, output_file, args_file

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.model, args.project_ind)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = sorted(output_dir.glob("*.pkl"))
    run_id = len(existing_runs)
    output_file = output_dir / f"{run_id}.pkl"
    args_file = output_dir / f"args{run_id}.json"
    return output_dir, output_file, args_file


def main() -> None:
    args = parse_args()
    require_core_dependencies()
    args.device = normalize_device(args.device)
    seed_everything(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.trust_remote_code)
    sentence_model = maybe_load_sentence_encoder(args.sentence_encoder, args.device)

    raw_dataset = load_dataset_from_source(args.halogen_source, args.split)
    categories = flatten_categories(args.category)
    records = normalize_halogen_records(
        raw_dataset=raw_dataset,
        categories=categories,
        fraction_of_data_to_use=args.fraction_of_data_to_use,
        limit=args.limit,
    )
    if not records:
        raise ValueError("No HALoGEN records were loaded. Check --halogen_source, --split, or --category.")

    generation_config = build_generation_config(tokenizer, args.max_new_tokens, args.stop_string or [])
    output_dir, output_file, args_file = resolve_output_paths(args)

    with args_file.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    print(f"Loaded {len(records)} HALoGEN records from {args.halogen_source}")
    if categories:
        print(f"Keeping categories: {sorted(categories)}")
    print(f"Writing outputs to {output_file}")

    sequences = []
    for record in progress(records, desc="Generating"):
        sequences.append(generate_one(record, model, tokenizer, generation_config, sentence_model, args))

    with output_file.open("wb") as handle:
        pickle.dump(sequences, handle)

    print(f"Wrote {len(sequences)} records to {output_file}")
    print(f"Args saved to {args_file}")


if __name__ == "__main__":
    main()
