"""
Task 4.3: Informal-aware MT test.

Takes noisy EN sentences, translates to ES with:
  1. Baseline NLLB (no fine-tuning, no shift)
  2. Fine-tuned model WITH shift

Outputs a side-by-side comparison for human judgment.

Usage:
    python src/evaluation/mt_test.py \
        --condition en_only_fixed_2x \
        --n 50 \
        [--config src/training/config.yaml]
"""

import argparse
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

# Lexical markers to spot in ES output
AR_MARKERS = [
    "cancha", "arquero", "penal", "técnico", "dt", "hinchada",
    "atajó", "atajo", "golazo", "venís", "querés", "decime", "che",
    "jaja", "lpm", "re ", "igual", "obviamente",
]
ES_SPAIN_MARKERS = [
    "portero", "penalti", "campo", "míster", "afición",
    "tío", "tía", "joder", "ostia", "venga",
]


@torch.no_grad()
def translate(
    sentences: list[str],
    model,
    tokenizer,
    device,
    src_lang: str,
    tgt_lang: str,
    direction=None,
    scale: float = 0.0,
    max_length: int = 128,
    batch_size: int = 16,
) -> list[str]:
    tokenizer.src_lang = src_lang
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    outputs = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(device)

        if direction is not None and scale != 0.0:
            enc_out = model.model.encoder(**inputs)
            hidden = enc_out.last_hidden_state + scale * direction.to(device)
            encoder_outputs = BaseModelOutput(last_hidden_state=hidden)
            generated = model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=inputs["attention_mask"],
                forced_bos_token_id=tgt_id,
                max_new_tokens=max_length,
            )
        else:
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tgt_id,
                max_new_tokens=max_length,
            )

        for seq in generated:
            outputs.append(tokenizer.decode(seq, skip_special_tokens=True))

    return outputs


def count_markers(texts: list[str], markers: list[str]) -> dict:
    combined = " ".join(texts).lower()
    return {m: combined.count(m) for m in markers if combined.count(m) > 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="en_only_fixed_2x")
    parser.add_argument("--config", default="src/training/config.yaml")
    parser.add_argument("--n", type=int, default=50, help="Number of test sentences")
    parser.add_argument("--out", default="results/mt_test_output.jsonl")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load noisy EN test sentences
    en_test_path = "data/multilexnorm/en_test.jsonl"
    with open(en_test_path, encoding="utf-8") as f:
        en_pairs = [json.loads(line) for line in f][: args.n]

    noisy_en = [p["noisy"] for p in en_pairs]
    clean_en  = [p["clean"] for p in en_pairs]

    direction = load_direction(cfg["direction"]["en_path"]).to(device)
    condition_cfg = cfg["conditions"][args.condition]
    scale_cfg = condition_cfg["scale"]
    scale = 2.0 if scale_cfg == "learned" else float(scale_cfg)

    # --- Baseline NLLB (no fine-tuning) ---
    print("Loading baseline NLLB...")
    base_tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base"])
    base_model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]["base"]).to(device)
    base_model.eval()

    print("Translating with baseline (no shift)...")
    baseline_translations = translate(
        noisy_en, base_model, base_tokenizer, device,
        src_lang=LANG_CODE["en"], tgt_lang=LANG_CODE["es"],
    )
    del base_model

    # --- Fine-tuned model with shift ---
    ckpt_path = Path(cfg["checkpoints"]["best_dir"]) / args.condition
    print(f"Loading fine-tuned model from {ckpt_path}...")
    ft_tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    ft_model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).to(device)
    ft_model.eval()

    scale_path = ckpt_path / "scale.pt"
    if scale_path.exists():
        scale = torch.load(scale_path)["scale"]
        print(f"  Loaded learned scale: {scale:.4f}")

    print("Translating with fine-tuned model + shift...")
    finetuned_translations = translate(
        noisy_en, ft_model, ft_tokenizer, device,
        src_lang=LANG_CODE["en"], tgt_lang=LANG_CODE["es"],
        direction=direction, scale=scale,
    )

    # --- Print side-by-side ---
    print("\n" + "="*80)
    print("INFORMAL-AWARE MT: Noisy EN → ES")
    print("="*80)
    for i in range(len(noisy_en)):
        print(f"\n--- {i+1} ---")
        print(f"  EN clean:   {clean_en[i]}")
        print(f"  EN noisy:   {noisy_en[i]}")
        print(f"  ES baseline:  {baseline_translations[i]}")
        print(f"  ES fine-tuned: {finetuned_translations[i]}")

    # --- Variant marker check ---
    print("\n" + "="*60)
    print("VARIANT MARKERS IN ES OUTPUT")
    print("="*60)
    for label, texts in [("Baseline", baseline_translations), ("Fine-tuned", finetuned_translations)]:
        ar = count_markers(texts, AR_MARKERS)
        es = count_markers(texts, ES_SPAIN_MARKERS)
        print(f"\n{label}:")
        print(f"  AR markers:       {ar}")
        print(f"  ES-Spain markers: {es}")

    # --- Save output ---
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(len(noisy_en)):
            row = {
                "en_clean":    clean_en[i],
                "en_noisy":    noisy_en[i],
                "es_baseline": baseline_translations[i],
                "es_finetuned": finetuned_translations[i],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
