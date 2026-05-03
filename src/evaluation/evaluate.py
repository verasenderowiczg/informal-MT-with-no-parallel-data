"""
Phase 4, Tasks 4.1 + 4.2 + 4.4: Evaluate all conditions on EN and ES test sets.

For each condition:
  - Load best checkpoint
  - Encode clean test sentences, shift by noise direction, decode
  - Compute: word accuracy vs noisy GT, ERR, chrF, % tokens changed from clean
  - Print 20 example outputs side-by-side
  - Write summary table to results/evaluation_summary.csv

Usage:
    python src/evaluation/evaluate.py \
        --conditions all \
        [--config src/training/config.yaml] \
        [--n_examples 20]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.direction import load_direction

LANG_CODE = {"en": "eng_Latn", "es": "spa_Latn"}
ALL_CONDITIONS = [
    "no_shift",
    "en_only_fixed_2x",
    "en_only_learned",
    "en_plus_es_fixed_2x",
    "en_plus_es_learned",
]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def word_accuracy(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    """Token-level accuracy (aligned by position, truncated to shorter)."""
    n = min(len(pred_tokens), len(ref_tokens))
    if n == 0:
        return 0.0
    correct = sum(p == r for p, r in zip(pred_tokens[:n], ref_tokens[:n]))
    return correct / max(len(ref_tokens), 1)


def error_reduction_rate(
    pred_tokens: list[str],
    noisy_tokens: list[str],
    clean_tokens: list[str],
) -> float:
    """
    ERR: the MultiLexNorm official metric.
    ERR = (errors_baseline - errors_system) / errors_baseline
    where errors_baseline = # tokens where noisy != clean,
          errors_system    = # tokens where pred != clean.
    """
    n = min(len(pred_tokens), len(clean_tokens), len(noisy_tokens))
    baseline_errors = sum(
        1 for i in range(n) if noisy_tokens[i].lower() != clean_tokens[i].lower()
    )
    if baseline_errors == 0:
        return float("nan")
    system_errors = sum(
        1 for i in range(n) if pred_tokens[i].lower() != clean_tokens[i].lower()
    )
    return (baseline_errors - system_errors) / baseline_errors


def chrf(hypothesis: str, reference: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (simplified, no library dependency)."""
    def char_ngrams(s, n):
        return [s[i : i + n] for i in range(len(s) - n + 1)]

    scores = []
    for order in range(1, n + 1):
        hyp_ng = char_ngrams(hypothesis, order)
        ref_ng = char_ngrams(reference, order)
        if not hyp_ng or not ref_ng:
            continue
        ref_counts = {}
        for g in ref_ng:
            ref_counts[g] = ref_counts.get(g, 0) + 1
        hits = 0
        for g in hyp_ng:
            if ref_counts.get(g, 0) > 0:
                hits += 1
                ref_counts[g] -= 1
        p = hits / len(hyp_ng)
        r = hits / len(ref_ng)
        if p + r == 0:
            scores.append(0.0)
        else:
            f = (1 + beta**2) * p * r / (beta**2 * p + r)
            scores.append(f)

    return sum(scores) / len(scores) if scores else 0.0


def pct_tokens_changed(pred_tokens: list[str], clean_tokens: list[str]) -> float:
    n = min(len(pred_tokens), len(clean_tokens))
    if n == 0:
        return 0.0
    changed = sum(1 for p, c in zip(pred_tokens[:n], clean_tokens[:n]) if p != c)
    return changed / len(clean_tokens)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate_outputs(
    clean_sentences: list[str],
    model,
    tokenizer,
    direction: torch.Tensor,
    scale: float,
    device: torch.device,
    src_lang: str,
    tgt_lang: str,
    max_length: int = 128,
    batch_size: int = 16,
) -> list[str]:
    tokenizer.src_lang = src_lang
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    outputs = []

    for i in range(0, len(clean_sentences), batch_size):
        batch_sents = clean_sentences[i : i + batch_size]
        inputs = tokenizer(
            batch_sents,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        enc_out = model.model.encoder(**inputs)
        hidden = enc_out.last_hidden_state

        if scale != 0.0:
            hidden = hidden + scale * direction.to(device)

        shifted = BaseModelOutput(last_hidden_state=hidden)
        generated = model.generate(
            encoder_outputs=shifted,
            attention_mask=inputs["attention_mask"],
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=max_length,
        )
        for seq in generated:
            outputs.append(tokenizer.decode(seq, skip_special_tokens=True))

    return outputs


# --------------------------------------------------------------------------- #
# Per-condition evaluation
# --------------------------------------------------------------------------- #

def load_condition(condition_id: str, best_dir: str, base_model: str, device: torch.device):
    ckpt_path = Path(best_dir) / condition_id
    if not ckpt_path.exists():
        return None, None, None

    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model.eval()

    scale_path = ckpt_path / "scale.pt"
    scale = torch.load(scale_path)["scale"] if scale_path.exists() else None

    return model, tokenizer, scale


def evaluate_condition(
    condition_id: str,
    model,
    tokenizer,
    direction: torch.Tensor,
    scale_override: float | None,
    cfg: dict,
    device: torch.device,
    n_examples: int,
) -> dict:
    condition_cfg = cfg["conditions"][condition_id]
    scale_cfg = condition_cfg["scale"]

    if scale_override is not None:
        scale = float(scale_override)
    elif scale_cfg == "learned":
        scale = 2.0  # fallback if scale.pt not found
    else:
        scale = float(scale_cfg)

    results = {}
    examples_saved = False

    for lang in ("en", "es"):
        test_path = cfg["data"][f"{lang}_dev"].replace("_test", "_test").replace("en_dev", "en_test").replace("es_dev", "es_test")
        # Use the actual test file
        test_path = f"data/multilexnorm/{lang}_test.jsonl"
        if not Path(test_path).exists():
            continue

        with open(test_path, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]

        clean_sents = [p["clean"] for p in pairs]
        noisy_sents = [p["noisy"] for p in pairs]

        predictions = generate_outputs(
            clean_sents, model, tokenizer, direction, scale, device,
            src_lang=LANG_CODE[lang], tgt_lang=LANG_CODE[lang],
        )

        accs, errs, chrfs, pct_changed = [], [], [], []
        for clean, noisy, pred in zip(clean_sents, noisy_sents, predictions):
            clean_tok = clean.lower().split()
            noisy_tok = noisy.lower().split()
            pred_tok  = pred.lower().split()

            accs.append(word_accuracy(pred_tok, noisy_tok))
            err = error_reduction_rate(pred_tok, noisy_tok, clean_tok)
            if err == err:  # not nan
                errs.append(err)
            chrfs.append(chrf(pred, noisy))
            pct_changed.append(pct_tokens_changed(pred_tok, clean_tok))

        results[lang] = {
            "word_acc":    round(sum(accs) / len(accs), 4),
            "err":         round(sum(errs) / len(errs), 4) if errs else float("nan"),
            "chrf":        round(sum(chrfs) / len(chrfs), 4),
            "pct_changed": round(sum(pct_changed) / len(pct_changed), 4),
            "n":           len(pairs),
        }

        # Save qualitative examples
        if not examples_saved:
            ex_dir = Path("results/examples")
            ex_dir.mkdir(parents=True, exist_ok=True)
            ex_path = ex_dir / f"{condition_id}_{lang}.txt"
            with open(ex_path, "w", encoding="utf-8") as f:
                for i, (clean, noisy, pred) in enumerate(
                    zip(clean_sents[:n_examples], noisy_sents[:n_examples], predictions[:n_examples])
                ):
                    f.write(f"--- {i+1} ---\n")
                    f.write(f"  clean:     {clean}\n")
                    f.write(f"  noisy GT:  {noisy}\n")
                    f.write(f"  predicted: {pred}\n\n")

    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=["all"])
    parser.add_argument("--config", default="src/training/config.yaml")
    parser.add_argument("--n_examples", type=int, default=20)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    conditions = ALL_CONDITIONS if args.conditions == ["all"] else args.conditions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    direction = load_direction(cfg["direction"]["en_path"]).to(device)

    all_results = {}
    for cond_id in conditions:
        print(f"\nEvaluating: {cond_id}")
        model, tokenizer, learned_scale = load_condition(
            cond_id, cfg["checkpoints"]["best_dir"], cfg["model"]["base"], device
        )
        if model is None:
            print(f"  No checkpoint found for {cond_id}, skipping.")
            continue

        r = evaluate_condition(
            cond_id, model, tokenizer, direction, learned_scale, cfg, device, args.n_examples
        )
        all_results[cond_id] = r

        for lang, metrics in r.items():
            print(f"  {lang.upper()}  acc={metrics['word_acc']:.4f}  "
                  f"ERR={metrics['err']:.4f}  chrF={metrics['chrf']:.4f}  "
                  f"%changed={metrics['pct_changed']:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write summary table
    out_path = Path("results/evaluation_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "lang", "word_acc", "err", "chrf", "pct_changed", "n"])
        for cond_id, lang_results in all_results.items():
            for lang, m in lang_results.items():
                writer.writerow([cond_id, lang, m["word_acc"], m["err"],
                                 m["chrf"], m["pct_changed"], m["n"]])

    print(f"\nSummary saved → {out_path}")

    # Print the paper table (Task 4.4)
    print("\n" + "="*90)
    print(f"{'Condition':<25} {'EN acc':>8} {'EN ERR':>8} {'ES acc':>8} {'ES ERR':>8} {'Scale':>8}")
    print("-"*90)
    for cond_id, lang_results in all_results.items():
        en = lang_results.get("en", {})
        es = lang_results.get("es", {})
        scale_note = "2.0" if "fixed" in cond_id else ("n/a" if cond_id == "no_shift" else "learned")
        print(f"{cond_id:<25} "
              f"{en.get('word_acc', float('nan')):>8.4f} "
              f"{en.get('err', float('nan')):>8.4f} "
              f"{es.get('word_acc', float('nan')):>8.4f} "
              f"{es.get('err', float('nan')):>8.4f} "
              f"{scale_note:>8}")
    print("="*90)


if __name__ == "__main__":
    main()
