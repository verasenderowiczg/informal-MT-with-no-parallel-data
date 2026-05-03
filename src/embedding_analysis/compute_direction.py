"""
Tasks 2.1 + 2.2: Compute NLLB noise directions from MultiLexNorm at scale.

Usage:
    python src/embedding_analysis/compute_direction.py \
        --lang en \
        --data_path data/multilexnorm/en_train.jsonl \
        --out_path models/en_noise_direction.pt

Computes the EN direction, then validates cross-lingual transfer with ES.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.direction import compute_mean_direction, save_direction


NLLB_MODEL = "facebook/nllb-200-distilled-600M"
LANG_CODES = {"en": "eng_Latn", "es": "spa_Latn"}


def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pairs.append((d["clean"], d["noisy"]))
    return pairs


@torch.no_grad()
def get_pooled_embeddings(
    sentences: list[str],
    tokenizer,
    encoder,
    device: torch.device,
    src_lang: str,
    max_length: int = 128,
    batch_size: int = 32,
) -> np.ndarray:
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
        hidden = encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_embs.append(pooled.cpu().numpy())
    return np.vstack(all_embs)


def analyse_direction(
    pairs: list[tuple[str, str]],
    tokenizer,
    encoder,
    device: torch.device,
    src_lang: str,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (direction [hidden_dim], diff_matrix [n, hidden_dim]).
    Prints consistency and baseline stats.
    """
    clean_sents = [c for c, _ in pairs]
    noisy_sents = [n for _, n in pairs]

    emb_clean = get_pooled_embeddings(clean_sents, tokenizer, encoder, device, src_lang)
    emb_noisy = get_pooled_embeddings(noisy_sents, tokenizer, encoder, device, src_lang)

    diffs = emb_noisy - emb_clean
    direction = diffs.mean(axis=0)

    baseline_sims = np.array([
        cosine_similarity(emb_clean[i : i + 1], emb_noisy[i : i + 1])[0, 0]
        for i in range(len(pairs))
    ])
    consistency_sims = cosine_similarity(diffs, direction.reshape(1, -1)).flatten()

    print(f"\n{label}")
    print(f"  Pairs analysed:        {len(pairs)}")
    print(f"  Direction magnitude:   {np.linalg.norm(direction):.4f}")
    print(f"  Mean baseline sim:     {baseline_sims.mean():.4f}  (std {baseline_sims.std():.4f})")
    print(f"  Direction consistency: {consistency_sims.mean():.4f}  (std {consistency_sims.std():.4f})")

    return direction, diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_train", default="data/multilexnorm/en_train.jsonl")
    parser.add_argument("--es_train", default="data/multilexnorm/es_train.jsonl")
    parser.add_argument("--en_out",   default="models/en_noise_direction.pt")
    parser.add_argument("--es_out",   default="models/es_noise_direction.pt")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Cap number of pairs used (for fast prototyping)")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {NLLB_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(device)
    model.eval()
    encoder = model.model.encoder

    # --- EN direction ---
    en_pairs = load_pairs(Path(args.en_train))
    if args.max_pairs:
        en_pairs = en_pairs[: args.max_pairs]

    en_dir, en_diffs = analyse_direction(
        en_pairs, tokenizer, encoder, device, LANG_CODES["en"], "EN noise direction"
    )
    en_tensor = torch.from_numpy(en_dir).unsqueeze(0)  # [1, hidden_dim]
    save_direction(en_tensor, args.en_out)
    print(f"  Saved → {args.en_out}")

    # --- ES direction ---
    es_pairs = load_pairs(Path(args.es_train))
    if args.max_pairs:
        es_pairs = es_pairs[: args.max_pairs]

    es_dir, es_diffs = analyse_direction(
        es_pairs, tokenizer, encoder, device, LANG_CODES["es"], "ES noise direction"
    )
    es_tensor = torch.from_numpy(es_dir).unsqueeze(0)
    save_direction(es_tensor, args.es_out)
    print(f"  Saved → {args.es_out}")

    # --- Cross-lingual transfer ---
    cross_sim = cosine_similarity(
        en_dir.reshape(1, -1), es_dir.reshape(1, -1)
    )[0, 0]
    print(f"\nCross-lingual direction cosine (EN↔ES): {cross_sim:.4f}")

    # Does applying the EN direction to ES clean embeddings move them toward ES noisy?
    es_clean = get_pooled_embeddings(
        [c for c, _ in es_pairs], tokenizer, encoder, device, LANG_CODES["es"]
    )
    es_noisy = get_pooled_embeddings(
        [n for _, n in es_pairs], tokenizer, encoder, device, LANG_CODES["es"]
    )
    es_shifted = es_clean + en_dir.reshape(1, -1)

    before = np.mean([
        cosine_similarity(es_clean[i : i + 1], es_noisy[i : i + 1])[0, 0]
        for i in range(len(es_pairs))
    ])
    after = np.mean([
        cosine_similarity(es_shifted[i : i + 1], es_noisy[i : i + 1])[0, 0]
        for i in range(len(es_pairs))
    ])
    print(f"ES sim to noisy — before shift: {before:.4f}, after EN shift: {after:.4f}  (Δ {after-before:+.4f})")


if __name__ == "__main__":
    main()
