"""
inside_pipeline.py
==================
Modular INSIDE (EigenScore) hallucination detection pipeline.

To swap models, change MODEL_NAME at the top or pass --model on the command line.

Usage:
    python inside_pipeline.py
    python inside_pipeline.py --model facebook/opt-350m
    python inside_pipeline.py --model facebook/opt-1.3b --max_prompts 100
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← teammates edit this section
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "facebook/opt-125m"  # swap for any HuggingFace causal LM
K = 10  # number of sampled responses per prompt
MAX_PROMPTS = 200  # prompts per category (None = full dataset)
LAYER_IDX = -1  # hidden layer to embed (-1 = last layer)
SIGMA_CLIP = 1.5  # feature clipping threshold (std units)
MAX_NEW_TOK = 128  # max generated tokens per response
TEMPERATURE = 0.7  # sampling temperature
TOP_P = 0.9  # nucleus sampling threshold
OUTPUT_DIR = "./results"

# All 9 HALoGEN categories
CATEGORIES = [
    "biographies",
    "code",
    "references",
    "historicalevents",
    "rationalization_binary",
    "rationalization_numerical",
    "summarization",
    "falsepresupposition",
    "simplification",
]


# ─────────────────────────────────────────────────────────────────────────────
# INSIDE CORE
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str):
    """Load tokenizer + model onto device."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M parameters | device: {device}")
    return tokenizer, model


def get_embedding(model, tokenizer, text: str, layer_idx: int, device: str) -> np.ndarray:
    """
    Mean-pool hidden states at `layer_idx` over all non-padding tokens.
    Returns a 1-D numpy array of shape (hidden_size,).
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden)
    hidden = out.hidden_states[layer_idx]  # (1, seq, d)
    mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq, 1)

    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled.squeeze(0).cpu().float().numpy()  # (d,)


def feature_clip(embeddings: np.ndarray) -> np.ndarray:
    """
    Clip each feature dimension to [mean ± SIGMA_CLIP * std] across K responses.
    Suppresses outlier activations that are model artifacts, not semantic signal.
    """
    mu = embeddings.mean(axis=0)
    std = embeddings.std(axis=0) + 1e-8
    return np.clip(embeddings, mu - SIGMA_CLIP * std, mu + SIGMA_CLIP * std)


def eigen_score(embeddings: np.ndarray) -> float:
    """
    Compute EigenScore from K embeddings.

    Steps:
      1. Feature-clip the (K × d) matrix.
      2. L2-normalise each row.
      3. Form Gram matrix G = E @ E.T  (K × K).
      4. Return mean(log(eigenvalues(G) + ε)).

    More negative → lower hallucination risk.
    Higher (closer to 0) → higher hallucination risk.
    """
    E = feature_clip(embeddings)

    # L2 normalise
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.where(norms < 1e-9, 1.0, norms)

    G = E @ E.T
    ev = np.abs(np.linalg.eigvalsh(G))
    return float(np.mean(np.log(ev + 1e-10)))


def generate_responses(model, tokenizer, prompt: str, device: str) -> list[str]:
    """Sample K independent responses for a prompt via nucleus sampling."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=256).to(device)
    n_prompt = inputs["input_ids"].shape[1]

    with torch.no_grad():
        seqs = model.generate(
            **inputs,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOK,
            num_return_sequences=K,
            pad_token_id=tokenizer.eos_token_id,
        )

    return [
        tokenizer.decode(s[n_prompt:], skip_special_tokens=True).strip()
        for s in seqs
    ]


def score_prompt(model, tokenizer, prompt: str, device: str) -> float:
    """
    Full INSIDE pipeline for one prompt.
    Returns a single EigenScore float.
    """
    responses = generate_responses(model, tokenizer, prompt, device)

    # Embed each (prompt + response) pair
    embeddings = np.stack([
        get_embedding(model, tokenizer, prompt + " " + r, LAYER_IDX, device)
        for r in responses
    ])  # (K, hidden_size)

    return eigen_score(embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_category(cat: str, max_prompts: int | None) -> list[str]:
    ds = load_dataset("lasha-nlp/HALoGEN-prompts", split="train")
    df = ds.to_pandas()
    df = df[df["category"] == cat]
    prompts = df["prompt"].dropna().tolist()
    if max_prompts:
        prompts = prompts[:max_prompts]
    return prompts


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

def run(model_name: str, max_prompts: int | None, categories: list[str]) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(model_name, device)

    records = []

    for cat in categories:
        print(f"\n── {cat} ──")
        try:
            prompts = load_category(cat, max_prompts)
        except Exception as e:
            print(f"  Could not load '{cat}': {e}")
            continue

        for prompt in tqdm(prompts, desc=cat):
            try:
                score = score_prompt(model, tokenizer, prompt, device)
                records.append({
                    "model": model_name,
                    "category": cat,
                    "prompt": prompt,
                    "score": score,
                })
            except Exception:
                pass  # skip failed prompts

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

CAT_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974",
]


def plot_results(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cats = sorted(df["category"].unique())

    # ── 1. Per-category histograms (grid) ────────────────────────────────────
    ncols = 3
    nrows = (len(cats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, (cat, color) in enumerate(zip(cats, CAT_COLORS)):
        scores = df[df["category"] == cat]["score"]
        ax = axes[i]
        ax.hist(scores, bins=25, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(scores.mean(), color="crimson", linestyle="--", linewidth=1.5,
                   label=f"μ = {scores.mean():.2f}")
        ax.set_title(cat.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("EigenScore (higher → more hallucination risk)")
        ax.set_ylabel("# Prompts")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    model_short = df["model"].iloc[0].split("/")[-1]
    fig.suptitle(f"INSIDE EigenScore — {model_short}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "histograms.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Boxplot ranked by mean score ──────────────────────────────────────
    order = (df.groupby("category")["score"].mean()
             .sort_values(ascending=False).index.tolist())
    data = [df[df["category"] == c]["score"] for c in order]

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, color in zip(bp["boxes"], CAT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    means = [d.mean() for d in data]
    ax.scatter(range(1, len(order) + 1), means,
               marker="D", color="black", s=35, zorder=5, label="Mean")

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([c.replace("_", "\n") for c in order], fontsize=9)
    ax.set_ylabel("EigenScore (higher → more hallucination risk)")
    ax.set_title(f"Hallucination Risk by Category — {model_short}", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "boxplot.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigures saved to {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INSIDE Pipeline — HALoGEN")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max_prompts", type=int, default=MAX_PROMPTS)
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = run(args.model, args.max_prompts, args.categories)

    # Save results
    csv_path = os.path.join(args.output_dir, "scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # Summary table
    summary = (
        df.groupby("category")["score"]
        .agg(n="count", mean="mean", std="std")
        .sort_values("mean", ascending=False)
        .round(3)
    )
    print("\n=== EigenScore Summary ===")
    print(summary.to_string())
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"))

    # Plots
    plot_results(df, os.path.join(args.output_dir, "figures"))
