"""
Task 5.2: Per-noise-type accuracy breakdown.

Categorizes MultiLexNorm tokens by noise type, then reports per-type accuracy
for each trained condition.

Noise categories (applied to each noisy token vs its clean counterpart):
  - identical:      noisy == clean (no change needed)
  - abbreviation:   shortened form (tmrw, u, pls, lol, smh, tbh, ffs, ngl, cuz, gonna, wanna, ...)
  - elongation:     repeated characters (soooo, incredibleeee, neeed, ...)
  - accent_drop:    missing accent mark (árbitro→arbitro, etc.)
  - number_subst:   digit/symbol for word (2nite, 2, 4, ...)
  - case_only:      only capitalisation differs
  - other:          any other change

Usage:
    python src/evaluation/noise_type_analysis.py \
        --conditions en_only_fixed_2x en_only_learned \
        [--lang en] \
        [--config src/training/config.yaml]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.direction import load_direction

LANG_CODE = {"en": "eng_Latn", "es": "spa_Latn"}

# Simple heuristic classifier for token-level noise type
ELONGATION_RE = re.compile(r"(.)\1{2,}")  # any char repeated 3+ times
NUMBER_SUBST_RE = re.compile(r"\d")


def classify_token_change(noisy_tok: str, clean_tok: str) -> str:
    """Return the noise category for a single noisy→clean token pair."""
    if noisy_tok == clean_tok:
        return "identical"
    if noisy_tok.lower() == clean_tok.lower():
        return "case_only"
    if ELONGATION_RE.search(noisy_tok):
        return "elongation"
    if NUMBER_SUBST_RE.search(noisy_tok) and not NUMBER_SUBST_RE.search(clean_tok):
        return "number_subst"
    # Accent drop: clean has accent, noisy doesn't (common in Spanish)
    def strip_accents(s):
        return s.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n")
    if strip_accents(noisy_tok.lower()) == clean_tok.lower():
        return "accent_drop"
    if strip_accents(noisy_tok.lower()) == strip_accents(clean_tok.lower()):
        return "accent_drop"
    # Abbreviation heuristic: noisy is shorter than clean (by ≥2 chars) or common abbr
    if len(noisy_tok) < len(clean_tok) - 1:
        return "abbreviation"
    return "other"


@torch.no_grad()
def generate_outputs(clean_sents, model, tokenizer, direction, scale, device, src_lang, tgt_lang,
                     max_length=128, batch_size=16):
    tokenizer.src_lang = src_lang
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    outputs = []
    for i in range(0, len(clean_sents), batch_size):
        batch = clean_sents[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           max_length=max_length, padding=True).to(device)
        enc_out = model.model.encoder(**inputs)
        hidden = enc_out.last_hidden_state
        if scale != 0.0:
            hidden = hidden + scale * direction.to(device)
        shifted = BaseModelOutput(last_hidden_state=hidden)
        generated = model.generate(encoder_outputs=shifted,
                                   attention_mask=inputs["attention_mask"],
                                   forced_bos_token_id=tgt_id,
                                   max_new_tokens=max_length)
        for seq in generated:
            outputs.append(tokenizer.decode(seq, skip_special_tokens=True))
    return outputs


def analyse_by_noise_type(pairs, predictions):
    """
    For each (clean, noisy, predicted) triple, classify each token change
    and record whether the prediction matched the noisy target.
    Returns: {category: {"correct": int, "total": int}}
    """
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for pair, pred in zip(pairs, predictions):
        clean_toks = pair["clean"].lower().split()
        noisy_toks = pair["noisy"].lower().split()
        pred_toks  = pred.lower().split()

        n = min(len(clean_toks), len(noisy_toks), len(pred_toks))
        for i in range(n):
            cat = classify_token_change(noisy_toks[i], clean_toks[i])
            stats[cat]["total"] += 1
            if pred_toks[i] == noisy_toks[i]:
                stats[cat]["correct"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+",
                        default=["en_only_fixed_2x", "en_only_learned"])
    parser.add_argument("--lang", default="en", choices=["en", "es"])
    parser.add_argument("--config", default="src/training/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    direction = load_direction(cfg["direction"]["en_path"]).to(device)

    test_path = f"data/multilexnorm/{args.lang}_test.jsonl"
    with open(test_path, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    clean_sents = [p["clean"] for p in pairs]

    src_lang = LANG_CODE[args.lang]
    all_stats = {}

    for cond_id in args.conditions:
        ckpt_path = Path(cfg["checkpoints"]["best_dir"]) / cond_id
        if not ckpt_path.exists():
            print(f"No checkpoint for {cond_id}, skipping.")
            continue

        print(f"\nLoading {cond_id}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        model.eval()

        scale_path = ckpt_path / "scale.pt"
        scale_cfg = cfg["conditions"][cond_id]["scale"]
        if scale_path.exists():
            scale = float(torch.load(scale_path)["scale"])
        elif scale_cfg == "learned":
            scale = 2.0
        else:
            scale = float(scale_cfg)

        predictions = generate_outputs(
            clean_sents, model, tokenizer, direction, scale, device,
            src_lang=src_lang, tgt_lang=src_lang,
        )
        all_stats[cond_id] = analyse_by_noise_type(pairs, predictions)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print table
    categories = ["identical", "case_only", "abbreviation", "elongation",
                  "accent_drop", "number_subst", "other"]

    print(f"\n{'='*70}")
    print(f"Noise-type accuracy breakdown — {args.lang.upper()} test set")
    print(f"{'Category':<15}", end="")
    for cond_id in all_stats:
        print(f"  {cond_id[:18]:>18}", end="")
    print(f"  {'(n tokens)':>10}")
    print(f"{'-'*70}")

    for cat in categories:
        print(f"{cat:<15}", end="")
        n = None
        for cond_id, stats in all_stats.items():
            s = stats.get(cat, {"correct": 0, "total": 0})
            if s["total"] > 0:
                acc = s["correct"] / s["total"]
                print(f"  {acc:>18.4f}", end="")
                n = s["total"]
            else:
                print(f"  {'—':>18}", end="")
        print(f"  {n or 0:>10}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
