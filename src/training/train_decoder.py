"""
Phase 3: Fine-tune NLLB's decoder on shifted encoder representations.

Encoder is frozen throughout. For each batch:
  1. Encode clean sentence (frozen encoder)
  2. Shift hidden states: shifted = hidden + scale * noise_direction
  3. Teacher-force decoder with noisy sentence as target
  4. Cross-entropy loss, backprop through decoder only

Usage:
    python src/training/train_decoder.py \
        --condition en_only_fixed_2x \
        [--config src/training/config.yaml] \
        [--use_lora]

Conditions defined in config.yaml:
    no_shift, en_only_fixed_2x, en_only_learned, en_plus_es_fixed_2x, en_plus_es_learned
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.dataset import NoisyNormDataset, make_collate_fn
from src.utils.direction import load_direction

LANG_CODE = {"en": "eng_Latn", "es": "spa_Latn"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_encoder(model):
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    model.model.shared.requires_grad_(False)  # shared embedding also frozen


def load_model_and_tokenizer(base: str, device: torch.device, use_lora: bool, lora_cfg: dict):
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSeq2SeqLM.from_pretrained(base).to(device)

    freeze_encoder(model)

    if use_lora:
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def make_datasets(condition_cfg: dict, data_cfg: dict, tokenizer, max_length: int):
    """Return (train_dataset, en_dev_dataset, es_dev_dataset)."""
    data_flag = condition_cfg["data"]  # "en" or "en+es"

    train_paths = [data_cfg["en_train"]]
    if data_flag == "en+es":
        train_paths.append(data_cfg["es_train"])

    # Build per-language datasets so we use the right src_lang for each.
    en_train_ds = NoisyNormDataset(
        [data_cfg["en_train"]], tokenizer,
        src_lang=LANG_CODE["en"], tgt_lang=LANG_CODE["en"],
        max_length=max_length,
    )
    if data_flag == "en+es":
        es_train_ds = NoisyNormDataset(
            [data_cfg["es_train"]], tokenizer,
            src_lang=LANG_CODE["es"], tgt_lang=LANG_CODE["es"],
            max_length=max_length,
        )
        train_ds = ConcatDataset([en_train_ds, es_train_ds])
    else:
        train_ds = en_train_ds

    en_dev_ds = NoisyNormDataset(
        [data_cfg["en_dev"]], tokenizer,
        src_lang=LANG_CODE["en"], tgt_lang=LANG_CODE["en"],
        max_length=max_length,
    )
    es_dev_ds = NoisyNormDataset(
        [data_cfg["es_dev"]], tokenizer,
        src_lang=LANG_CODE["es"], tgt_lang=LANG_CODE["es"],
        max_length=max_length,
    )

    return train_ds, en_dev_ds, es_dev_ds


# --------------------------------------------------------------------------- #
# Forward pass with direction shift
# --------------------------------------------------------------------------- #

def forward_with_shift(model, batch, direction, scale, device):
    """
    Encode, optionally shift hidden states, decode with teacher forcing.
    Returns cross-entropy loss.
    """
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["labels"].to(device)

    with torch.no_grad():
        encoder_outputs = model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    hidden = encoder_outputs.last_hidden_state  # [B, T, H]

    if scale != 0:
        d = direction.to(device)
        if isinstance(scale, nn.Parameter):
            hidden = hidden + scale * d
        else:
            hidden = hidden + float(scale) * d

    shifted_encoder_out = BaseModelOutput(last_hidden_state=hidden)

    # Get decoder BOS token id (forced_bos for target language handled via labels)
    outputs = model(
        attention_mask=attention_mask,
        encoder_outputs=shifted_encoder_out,
        labels=labels,
    )
    return outputs.loss


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def evaluate(model, loader, direction, scale, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            loss = forward_with_shift(model, batch, direction, scale, device)
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train(args, cfg):
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    condition_id  = args.condition
    condition_cfg = cfg["conditions"][condition_id]
    data_cfg      = cfg["data"]
    train_cfg     = cfg["training"]

    print(f"\nCondition: {condition_id}")
    print(f"  scale: {condition_cfg['scale']},  data: {condition_cfg['data']}")

    model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["base"], device, args.use_lora, train_cfg["lora"]
    )

    # Noise direction
    direction = load_direction(cfg["direction"]["en_path"]).to(device)  # [1, H]

    # Scale: either fixed float or learnable nn.Parameter
    scale_cfg = condition_cfg["scale"]
    if scale_cfg == "learned":
        scale = nn.Parameter(torch.tensor(2.0, device=device))
        extra_params = [scale]
    else:
        scale = float(scale_cfg)
        extra_params = []

    # Datasets & loaders
    train_ds, en_dev_ds, es_dev_ds = make_datasets(
        condition_cfg, data_cfg, tokenizer, data_cfg["max_length"]
    )
    collate = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True, collate_fn=collate, num_workers=2)
    en_dev_loader = DataLoader(en_dev_ds, batch_size=train_cfg["batch_size"],
                               shuffle=False, collate_fn=collate, num_workers=2)
    es_dev_loader = DataLoader(es_dev_ds, batch_size=train_cfg["batch_size"],
                               shuffle=False, collate_fn=collate, num_workers=2)

    print(f"  Train examples: {len(train_ds)}, EN dev: {len(en_dev_ds)}, ES dev: {len(es_dev_ds)}")

    # Optimizer — only trainable params (decoder) + optional learned scale
    trainable = [p for p in model.parameters() if p.requires_grad] + extra_params
    optimizer = torch.optim.AdamW(trainable, lr=train_cfg["learning_rate"],
                                  weight_decay=train_cfg["weight_decay"])

    total_steps = math.ceil(len(train_loader) * train_cfg["epochs"])
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Checkpoint dirs
    ckpt_dir = Path(cfg["checkpoints"]["dir"]) / condition_id
    best_dir = Path(cfg["checkpoints"]["best_dir"]) / condition_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    best_dev_loss = float("inf")
    log_rows = []

    model.train()
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, n_batches = 0.0, 0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = forward_with_shift(model, batch, direction, scale, device)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, train_cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

            if step % 100 == 0:
                print(f"  Epoch {epoch} step {step}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}", flush=True)

        avg_train = train_loss / n_batches
        en_dev_loss = evaluate(model, en_dev_loader, direction, scale, device)
        es_dev_loss = evaluate(model, es_dev_loader, direction, scale, device)

        scale_val = scale.item() if isinstance(scale, nn.Parameter) else float(scale)
        row = {
            "epoch": epoch,
            "train_loss": round(avg_train, 4),
            "en_dev_loss": round(en_dev_loss, 4),
            "es_dev_loss": round(es_dev_loss, 4),
            "scale": round(scale_val, 4),
        }
        log_rows.append(row)

        print(f"Epoch {epoch}/{train_cfg['epochs']}  "
              f"train={avg_train:.4f}  en_dev={en_dev_loss:.4f}  "
              f"es_dev={es_dev_loss:.4f}  scale={scale_val:.4f}")

        # Save epoch checkpoint
        if cfg["checkpoints"]["save_every_epoch"]:
            ep_path = ckpt_dir / f"epoch_{epoch}"
            ep_path.mkdir(exist_ok=True)
            model.save_pretrained(ep_path)
            tokenizer.save_pretrained(ep_path)
            if isinstance(scale, nn.Parameter):
                torch.save({"scale": scale.item()}, ep_path / "scale.pt")

        # Save best by EN dev loss
        if en_dev_loss < best_dev_loss:
            best_dev_loss = en_dev_loss
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            if isinstance(scale, nn.Parameter):
                torch.save({"scale": scale.item()}, best_dir / "scale.pt")
            print(f"  *** New best EN dev loss: {best_dev_loss:.4f} → saved to {best_dir}")

    # Save training log
    log_path = ckpt_dir / "training_log.jsonl"
    with open(log_path, "w") as f:
        for row in log_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nTraining log saved → {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        choices=["no_shift", "en_only_fixed_2x", "en_only_learned",
                                 "en_plus_es_fixed_2x", "en_plus_es_learned"],
                        help="Which experimental condition to train")
    parser.add_argument("--config", default="src/training/config.yaml")
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA instead of full decoder fine-tuning")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(args, cfg)


if __name__ == "__main__":
    main()
