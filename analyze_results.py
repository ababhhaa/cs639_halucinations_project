from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


METRIC_FIELDS = [
    "eigenIndicator",
    "perplexity",
    "entropy",
    "lexical_similarity",
    "energy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CSV summaries from HALoGEN experiment results.")
    parser.add_argument("--input", required=True, help="Pickle file or directory containing pickle files.")
    parser.add_argument("--output_dir", default="analysis_outputs", help="Directory for CSV summaries and plots.")
    parser.add_argument("--plots", action="store_true", help="Create histogram plots for EigenScore.")
    parser.add_argument(
        "--evaluate_correctness",
        action="store_true",
        help="Add rough answer-similarity labels and metric AUROC. This is not official HALoGEN judging.",
    )
    parser.add_argument(
        "--correctness_threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for the rough correctness label.",
    )
    return parser.parse_args()


def find_pickle_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.pkl") if path.is_file())
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def load_pickle_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict(orient="records")
    if isinstance(obj, tuple):
        obj = list(obj)
    if not isinstance(obj, list):
        raise TypeError(f"Expected a list-like object, found {type(obj).__name__}")
    if not all(isinstance(item, dict) for item in obj):
        raise TypeError("Expected all result records to be dictionaries")
    return obj


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        if np.isscalar(value):
            return bool(np.isnan(value))
    except TypeError:
        return False
    return False


def to_float(value: Any) -> Optional[float]:
    if is_missing(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


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


def word_count(text: Any) -> int:
    return len(str(text or "").split())


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def longest_common_subsequence_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for j, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(first: str, second: str) -> float:
    first_tokens = normalize_answer(first).split()
    second_tokens = normalize_answer(second).split()
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


def answer_similarity(generation: str, answers: Sequence[str]) -> Optional[float]:
    candidates = [answer for answer in answers if answer and str(answer).strip()]
    if not candidates:
        return None
    generation_norm = normalize_answer(generation)
    best = 0.0
    for answer in candidates:
        answer_norm = normalize_answer(str(answer))
        if answer_norm and generation_norm and (answer_norm in generation_norm or generation_norm in answer_norm):
            best = max(best, 1.0)
        best = max(best, rouge_l_f1(generation, str(answer)))
    return best


def length_bucket(prompt_words: int) -> str:
    return "short_0_100" if prompt_words <= 100 else "long_100_plus"


def build_flat_rows(paths: Sequence[Path], evaluate_correctness: bool, threshold: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        records = load_pickle_records(path)
        for index, record in enumerate(records):
            prompt = str(record.get("prompt", ""))
            generation = str(record.get("most_likely_generation", ""))
            prompt_words = word_count(prompt)
            row = {
                "source_file": str(path),
                "run_name": path.parent.name,
                "id": record.get("id", f"{path.stem}_{index}"),
                "category": record.get("category", "unknown"),
                "prompt_length_words": prompt_words,
                "generation_length_words": word_count(generation),
                "length_bucket": length_bucket(prompt_words),
                "feature_clipping_enabled": record.get("feature_clipping_enabled", False),
                "feature_clip_layer": record.get("feature_clip_layer"),
                "feature_clip_tokens_collected": record.get("feature_clip_tokens_collected"),
                "perplexity": to_float(record.get("perplexity")),
                "energy": to_float(record.get("energy")),
                "entropy": to_float(record.get("entropy")),
                "lexical_similarity": to_float(record.get("lexical_similarity")),
                "eigenIndicator": to_float(record.get("eigenIndicator")),
                "eigenIndicatorOutput": to_float(record.get("eigenIndicatorOutput")),
            }
            if evaluate_correctness:
                answers = to_text_list(record.get("answer")) + to_text_list(record.get("additional_answers"))
                similarity = answer_similarity(generation, answers)
                row["answer_similarity"] = similarity
                row["rough_correct"] = None if similarity is None else int(similarity >= threshold)
                row["rough_hallucination"] = None if similarity is None else int(similarity < threshold)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return groups


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return float(np.mean(np.asarray(numeric, dtype=np.float64)))


def std(values: Sequence[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    if len(numeric) == 1:
        return 0.0
    return float(np.std(np.asarray(numeric, dtype=np.float64), ddof=1))


def summarize_groups(rows: Sequence[Dict[str, Any]], key: str, output_key: str) -> List[Dict[str, Any]]:
    summaries = []
    for group_value, group in sorted(group_rows(rows, key).items()):
        summaries.append(
            {
                output_key: group_value,
                "n": len(group),
                "mean_EigenScore": mean([row.get("eigenIndicator") for row in group]),
                "std_EigenScore": std([row.get("eigenIndicator") for row in group]),
                "mean_perplexity": mean([row.get("perplexity") for row in group]),
                "mean_entropy": mean([row.get("entropy") for row in group]),
                "mean_lexical_similarity": mean([row.get("lexical_similarity") for row in group]),
                "mean_energy": mean([row.get("energy") for row in group]),
            }
        )
    return summaries


def compute_auroc(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError("scikit-learn is required for --evaluate_correctness AUROC.") from exc

    summaries = []
    for metric in METRIC_FIELDS:
        labels = []
        scores = []
        for row in rows:
            label = row.get("rough_hallucination")
            score = row.get(metric)
            if label is None or score is None:
                continue
            labels.append(int(label))
            scores.append(float(score))
        if len(set(labels)) < 2:
            raw_auc = None
            flipped_auc = None
        else:
            raw_auc = float(roc_auc_score(labels, scores))
            flipped_auc = float(roc_auc_score(labels, [-score for score in scores]))
        summaries.append(
            {
                "metric": metric,
                "n": len(scores),
                "positive_label": "rough_hallucination",
                "auroc_raw": raw_auc,
                "auroc_flipped": flipped_auc,
                "best_direction_auroc": None if raw_auc is None else max(raw_auc, flipped_auc),
            }
        )
    return summaries


def create_plots(rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    valid_rows = [row for row in rows if row.get("eigenIndicator") is not None]
    if not valid_rows:
        return

    def create_group_bar_plot(key: str, label: str, filename: str) -> None:
        summaries = summarize_groups(valid_rows, key, key)
        summaries = [item for item in summaries if item[key] not in ("None", "unknown")]
        if len(summaries) < 2 or len(summaries) > 8:
            return

        metrics = [
            ("mean_EigenScore", "Mean EigenScore"),
            ("mean_perplexity", "Mean Perplexity"),
            ("mean_entropy", "Mean Entropy"),
            ("mean_lexical_similarity", "Mean Lexical Similarity"),
        ]
        labels = [str(item[key]).replace("_", "\n") for item in summaries]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD", "#8C8C8C", "#DD8452"]
        for ax, (metric, title) in zip(axes, metrics):
            values = [item.get(metric) for item in summaries]
            ax.bar(labels, values, color=colors[: len(labels)])
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
            for index, value in enumerate(values):
                if value is not None:
                    ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        fig.suptitle(label)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    categories = sorted({str(row.get("category", "unknown")) for row in valid_rows})
    plt.figure(figsize=(10, 6))
    for category in categories[:8]:
        values = [row["eigenIndicator"] for row in valid_rows if str(row.get("category", "unknown")) == category]
        if values:
            plt.hist(values, bins=20, alpha=0.45, label=category)
    plt.xlabel("EigenScore")
    plt.ylabel("Count")
    plt.title("EigenScore by HALoGEN category")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "eigenscore_by_category.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for bucket in ["short_0_100", "long_100_plus"]:
        values = [row["eigenIndicator"] for row in valid_rows if row.get("length_bucket") == bucket]
        if values:
            plt.hist(values, bins=20, alpha=0.55, label=bucket)
    plt.xlabel("EigenScore")
    plt.ylabel("Count")
    plt.title("EigenScore by prompt length bucket")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "eigenscore_by_prompt_length.png", dpi=200)
    plt.close()

    create_group_bar_plot("run_name", "Metric summary by experiment run", "metric_summary_by_run.png")
    create_group_bar_plot("length_bucket", "Metric summary by prompt length", "metric_summary_by_prompt_length.png")


def main() -> None:
    args = parse_args()
    input_paths = find_pickle_files(Path(args.input))
    if not input_paths:
        raise FileNotFoundError(f"No .pkl files found under {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_flat_rows(input_paths, args.evaluate_correctness, args.correctness_threshold)
    write_csv(output_dir / "results_flat.csv", rows)
    write_csv(output_dir / "category_summary.csv", summarize_groups(rows, "category", "category"))
    write_csv(output_dir / "length_summary.csv", summarize_groups(rows, "length_bucket", "length_bucket"))
    write_csv(output_dir / "run_summary.csv", summarize_groups(rows, "run_name", "run_name"))
    write_csv(
        output_dir / "feature_clip_summary.csv",
        summarize_groups(rows, "feature_clipping_enabled", "feature_clipping_enabled"),
    )

    if args.evaluate_correctness:
        write_csv(output_dir / "detection_summary.csv", compute_auroc(rows))
    if args.plots:
        create_plots(rows, output_dir)

    print(f"Loaded {len(rows)} records from {len(input_paths)} pickle file(s).")
    print(f"Wrote {output_dir / 'results_flat.csv'}")
    print(f"Wrote {output_dir / 'category_summary.csv'}")
    print(f"Wrote {output_dir / 'length_summary.csv'}")
    print(f"Wrote {output_dir / 'run_summary.csv'}")
    print(f"Wrote {output_dir / 'feature_clip_summary.csv'}")
    if args.evaluate_correctness:
        print(f"Wrote {output_dir / 'detection_summary.csv'}")
    if args.plots:
        print(f"Wrote EigenScore plots under {output_dir}")


if __name__ == "__main__":
    main()
