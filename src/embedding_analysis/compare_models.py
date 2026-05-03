"""
Task 2.3: Three-model comparison — XLM-R, NLLB encoder, CANINE.

Runs the same noise direction analysis across all three models and prints
the comparison table for the paper.

Usage:
    python src/embedding_analysis/compare_models.py \
        --en_data data/multilexnorm/en_train.jsonl \
        --es_data data/multilexnorm/es_train.jsonl \
        --out     results/embedding_comparison.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    CanineModel,
    CanineTokenizer,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# --------------------------------------------------------------------------- #
# Model configs
# --------------------------------------------------------------------------- #

MODEL_CONFIGS = {
    "XLM-R": {
        "model_name": "xlm-roberta-base",
        "type": "encoder",
        "en_lang": None,
        "es_lang": None,
    },
    "NLLB": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "type": "seq2seq",
        "en_lang": "eng_Latn",
        "es_lang": "spa_Latn",
    },
    "CANINE": {
        "model_name": "google/canine-s",
        "type": "canine",
        "en_lang": None,
        "es_lang": None,
    },
}


def load_pairs(path: Path, max_pairs: int | None = None) -> list[tuple[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pairs.append((d["clean"], d["noisy"]))
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


@torch.no_grad()
def get_embeddings_generic(sentences, tokenizer, model, device, src_lang=None,
                            batch_size=32, max_length=128):
    if src_lang and hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang

    all_embs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        if hasattr(model, "model") and hasattr(model.model, "encoder"):
            # NLLB / seq2seq: use encoder
            hidden = model.model.encoder(**inputs).last_hidden_state
        else:
            hidden = model(**inputs).last_hidden_state

        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_embs.append(pooled.cpu().numpy())

    return np.vstack(all_embs)


def analyse_model(cfg: dict, en_pairs, es_pairs, device) -> dict:
    name = cfg["model_name"]
    mtype = cfg["type"]
    print(f"\n  Loading {name}...")

    if mtype == "canine":
        tokenizer = CanineTokenizer.from_pretrained(name)
        model = CanineModel.from_pretrained(name).to(device)
    elif mtype == "seq2seq":
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(device)

    model.eval()

    en_lang = cfg["en_lang"]
    es_lang = cfg["es_lang"]

    en_clean_sents = [c for c, _ in en_pairs]
    en_noisy_sents = [n for _, n in en_pairs]
    es_clean_sents = [c for c, _ in es_pairs]
    es_noisy_sents = [n for _, n in es_pairs]

    emb_en_clean = get_embeddings_generic(en_clean_sents, tokenizer, model, device, en_lang)
    emb_en_noisy = get_embeddings_generic(en_noisy_sents, tokenizer, model, device, en_lang)
    emb_es_clean = get_embeddings_generic(es_clean_sents, tokenizer, model, device, es_lang)
    emb_es_noisy = get_embeddings_generic(es_noisy_sents, tokenizer, model, device, es_lang)

    en_diffs = emb_en_noisy - emb_en_clean
    es_diffs = emb_es_noisy - emb_es_clean
    en_dir = en_diffs.mean(axis=0)
    es_dir = es_diffs.mean(axis=0)

    en_baseline = np.mean([
        cosine_similarity(emb_en_clean[i:i+1], emb_en_noisy[i:i+1])[0, 0]
        for i in range(len(en_pairs))
    ])
    es_baseline = np.mean([
        cosine_similarity(emb_es_clean[i:i+1], emb_es_noisy[i:i+1])[0, 0]
        for i in range(len(es_pairs))
    ])
    en_consistency = cosine_similarity(en_diffs, en_dir.reshape(1, -1)).flatten().mean()
    cross_lingual = cosine_similarity(en_dir.reshape(1, -1), es_dir.reshape(1, -1))[0, 0]

    es_shifted = emb_es_clean + en_dir.reshape(1, -1)
    es_improvement = np.mean([
        cosine_similarity(es_shifted[i:i+1], emb_es_noisy[i:i+1])[0, 0]
        - cosine_similarity(emb_es_clean[i:i+1], emb_es_noisy[i:i+1])[0, 0]
        for i in range(len(es_pairs))
    ])

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "en_baseline":    round(float(en_baseline), 4),
        "es_baseline":    round(float(es_baseline), 4),
        "en_consistency": round(float(en_consistency), 4),
        "cross_lingual":  round(float(cross_lingual), 4),
        "es_improvement": round(float(es_improvement), 4),
    }


def print_table(results: dict) -> None:
    header = f"{'Model':<10} {'EN baseline':>12} {'ES baseline':>12} {'EN consistency':>15} {'Cross-lingual':>14} {'ES improvement':>15}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for model_label, r in results.items():
        print(
            f"{model_label:<10} "
            f"{r['en_baseline']:>12.4f} "
            f"{r['es_baseline']:>12.4f} "
            f"{r['en_consistency']:>15.4f} "
            f"{r['cross_lingual']:>14.4f} "
            f"{r['es_improvement']:>15.4f}"
        )
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_data",   default="data/multilexnorm/en_train.jsonl")
    parser.add_argument("--es_data",   default="data/multilexnorm/es_train.jsonl")
    parser.add_argument("--out",       default="results/embedding_comparison.csv")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--models",    nargs="+", default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Which models to run (default: all three)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    en_pairs = load_pairs(Path(args.en_data), args.max_pairs)
    es_pairs = load_pairs(Path(args.es_data), args.max_pairs)
    print(f"EN pairs: {len(en_pairs)}, ES pairs: {len(es_pairs)}")

    results = {}
    for label in args.models:
        cfg = {**MODEL_CONFIGS[label], "label": label}
        results[label] = analyse_model(cfg, en_pairs, es_pairs, device)

    print_table(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "en_baseline", "es_baseline", "en_consistency",
                        "cross_lingual", "es_improvement", "n_en_pairs", "n_es_pairs"],
        )
        writer.writeheader()
        for label, r in results.items():
            writer.writerow({
                "model": label,
                "n_en_pairs": len(en_pairs),
                "n_es_pairs": len(es_pairs),
                **r,
            })
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
