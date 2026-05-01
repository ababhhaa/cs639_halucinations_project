from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REQUIRED_FIELDS = [
    "prompt",
    "category",
    "most_likely_generation",
    "generations",
    "perplexity",
    "energy",
    "entropy",
    "lexical_similarity",
    "eigenIndicator",
]

METRIC_FIELDS = [
    "perplexity",
    "energy",
    "entropy",
    "lexical_similarity",
    "eigenIndicator",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HALoGEN experiment pickle outputs.")
    parser.add_argument("--input", required=True, help="Pickle file or directory containing pickle files.")
    parser.add_argument("--expected_k", type=int, default=None, help="Expected number of sampled generations per prompt.")
    parser.add_argument(
        "--k_tolerance",
        type=int,
        default=0,
        help="Allowed absolute difference from --expected_k before warning.",
    )
    parser.add_argument("--csv_out", default=None, help="Optional CSV path for a per-file validation summary.")
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


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text.strip().lower())


def only_punctuation(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and not any(char.isalnum() for char in stripped)


def is_prompt_echo(generation: str, prompt: str) -> bool:
    gen_norm = normalize_text(generation)
    prompt_norm = normalize_text(prompt)
    if len(gen_norm) < 20 or len(prompt_norm) < 20:
        return False
    prompt_prefix = prompt_norm[: min(120, len(prompt_norm))]
    return gen_norm.startswith(prompt_prefix) or prompt_norm.startswith(gen_norm)


def metric_summary(records: Sequence[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    values = [to_float(record.get(metric)) for record in records]
    numeric = np.asarray([value for value in values if value is not None], dtype=np.float64)
    missing = len(values) - len(numeric)
    if len(numeric) == 0:
        return {
            "metric": metric,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "missing": missing,
        }
    return {
        "metric": metric,
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0,
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
        "missing": missing,
    }


def summarize_file(
    path: Path,
    records: Sequence[Dict[str, Any]],
    expected_k: Optional[int],
    k_tolerance: int,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    categories = sorted({str(record.get("category", "missing")) for record in records})
    missing_by_field = {field: 0 for field in REQUIRED_FIELDS}
    k_mismatches = 0
    empty_generation_records = 0
    empty_generation_items = 0
    degenerate_generation_items = 0
    nan_metric_count = 0

    for record in records:
        for field in REQUIRED_FIELDS:
            if field not in record:
                missing_by_field[field] += 1
        generations = record.get("generations", [])
        if not isinstance(generations, (list, tuple)):
            generations = []
        if len(generations) == 0:
            empty_generation_records += 1
        if expected_k is not None and abs(len(generations) - expected_k) > k_tolerance:
            k_mismatches += 1
        prompt = record.get("prompt", "")
        for generation in generations:
            generation_text = "" if generation is None else str(generation)
            if not generation_text.strip():
                empty_generation_items += 1
            if (
                not generation_text.strip()
                or only_punctuation(generation_text)
                or is_prompt_echo(generation_text, prompt)
            ):
                degenerate_generation_items += 1
        for metric in METRIC_FIELDS:
            if to_float(record.get(metric)) is None:
                nan_metric_count += 1

    missing_fields = {field: count for field, count in missing_by_field.items() if count}
    warnings = []
    failures = []
    if not records:
        failures.append("no records")
    if missing_fields:
        failures.append("missing required fields")
    if empty_generation_records:
        failures.append("records with no sampled generations")
    if k_mismatches:
        warnings.append("sampled generation count mismatch")
    if empty_generation_items:
        warnings.append("blank sampled generations")
    if degenerate_generation_items:
        warnings.append("degenerate sampled generations")
    if nan_metric_count:
        warnings.append("missing or non-finite metrics")

    status = "FAIL" if failures else "WARN" if warnings else "PASS"
    summary = {
        "file": str(path),
        "status": status,
        "records": len(records),
        "categories": ";".join(categories),
        "missing_fields": ";".join(f"{field}:{count}" for field, count in missing_fields.items()),
        "k_mismatches": k_mismatches,
        "empty_generation_records": empty_generation_records,
        "empty_generation_items": empty_generation_items,
        "degenerate_generation_items": degenerate_generation_items,
        "nan_metric_count": nan_metric_count,
    }
    return summary, warnings, failures


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    paths = find_pickle_files(Path(args.input))
    if not paths:
        raise FileNotFoundError(f"No .pkl files found under {args.input}")

    file_summaries: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []
    load_errors: List[str] = []

    for path in paths:
        print(f"\n== {path} ==")
        try:
            records = load_pickle_records(path)
            all_records.extend(records)
            summary, warnings, failures = summarize_file(path, records, args.expected_k, args.k_tolerance)
        except Exception as exc:
            load_errors.append(f"{path}: {type(exc).__name__}: {exc}")
            summary = {
                "file": str(path),
                "status": "FAIL",
                "records": 0,
                "categories": "",
                "missing_fields": "",
                "k_mismatches": 0,
                "empty_generation_records": 0,
                "empty_generation_items": 0,
                "degenerate_generation_items": 0,
                "nan_metric_count": 0,
            }
            warnings = []
            failures = ["load error"]
        file_summaries.append(summary)
        print(f"status: {summary['status']}")
        print(f"records: {summary['records']}")
        print(f"categories: {summary['categories'] or '(none)'}")
        print(f"missing fields: {summary['missing_fields'] or '(none)'}")
        print(f"k mismatches: {summary['k_mismatches']}")
        print(f"empty generation records: {summary['empty_generation_records']}")
        print(f"empty generation items: {summary['empty_generation_items']}")
        print(f"degenerate generation items: {summary['degenerate_generation_items']}")
        print(f"missing/non-finite metric values: {summary['nan_metric_count']}")
        if failures:
            print("failures: " + "; ".join(failures))
        if warnings:
            print("warnings: " + "; ".join(warnings))

    if all_records:
        print("\nMetric summary across loaded records:")
        for metric in METRIC_FIELDS:
            stats = metric_summary(all_records, metric)
            print(
                f"  {metric}: mean={stats['mean']} std={stats['std']} "
                f"min={stats['min']} max={stats['max']} missing={stats['missing']}"
            )

    if args.csv_out:
        write_csv(Path(args.csv_out), file_summaries)
        print(f"\nWrote CSV summary to {args.csv_out}")

    if load_errors:
        print("\nLoad errors:")
        for error in load_errors:
            print(f"  {error}")

    statuses = {summary["status"] for summary in file_summaries}
    final_status = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
    print(f"\nFINAL STATUS: {final_status}")


if __name__ == "__main__":
    main()
