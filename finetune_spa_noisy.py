"""
Fine-tune NLLB decoder (encoder frozen) on clean→noisy ES pairs.
Registers spa_noisy_Latn as a new language tag, initialized from spa_Latn.
Uses MultiLexNorm ES sentence-level JSONL data.

Usage:
    python finetune_spa_noisy.py \
        --train_path data/multilexnorm/es_train_augmented.jsonl \
        --test_path data/multilexnorm/es_test_augmented.jsonl \
        --output_dir checkpoints/spa_noisy \
        --epochs 10 \
        --batch_size 8 \
        --lr 5e-5

After training, test with:
    python finetune_spa_noisy.py \
        --test_path data/multilexnorm/es_test_augmented.jsonl \
        --output_dir checkpoints/spa_noisy \
        --eval_only
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import gc

# ============================================================
# 1. Register spa_noisy_Latn
# ============================================================

def register_spa_noisy(tokenizer, model, noise_direction=None):
    """
    Add spa_noisy_Latn as a new language token, initialized
    as a copy of spa_Latn's embedding.
    """
    new_lang = "spa_noisy_Latn"

    if new_lang in tokenizer.get_vocab():
        print(f"  {new_lang} already in vocab, skipping registration.")
        return tokenizer.convert_tokens_to_ids(new_lang)

    # Add the new special token
    tokenizer.add_special_tokens({"additional_special_tokens": [new_lang]})
    model.resize_token_embeddings(len(tokenizer))

    # Initialize from spa_Latn
    spa_id = tokenizer.convert_tokens_to_ids("spa_Latn")
    new_id = tokenizer.convert_tokens_to_ids(new_lang)

    with torch.no_grad():
        # Copy into the shared embedding (used by both encoder and decoder)
        if noise_direction is not None:
            model.model.shared.weight[new_id] = model.model.shared.weight[spa_id].clone() + noise_direction.to(model.model.shared.weight.device)
            print(f"  Initialized with noise direction (magnitude {noise_direction.norm().item():.4f})")
        else:
            model.model.shared.weight[new_id] = model.model.shared.weight[spa_id].clone()
        # lm_head (output projection) — if it's tied, this is the same tensor;
        # if not, copy there too
        if not model.config.tie_word_embeddings:
            model.lm_head.weight[new_id] = model.lm_head.weight[spa_id].clone()

    print(f"  Registered {new_lang} (id={new_id}), initialized from spa_Latn (id={spa_id})")
    return new_id

def compute_es_noise_direction(tokenizer, model, device):
    """Compute the ES clean→noisy direction from the 25 handcrafted pairs."""
    es_clean = [
        "Sí, pero de todos modos es una locura, haha",
        "¡Haha, ah, okey! ¡Sí, ya hace un tiempo que salen! ¡Son muy monos!",
        "El mejor snapchat que ví en todo el día jajajajaj",
        "Haha, no te hagas el mono aquí, por favor",
        "Ni siquiera sé por qué me molesto... No puedo creerlo",
        "Mmm... me pareció que la conversación era bastante graciosa, así que...",
        "Dejad de ser tan gays, ninguno de los tres lo es, no tenemos tiempo para esto.",
        "Estoy empezando a pensar que soy demasiado para ti",
        "No lo sé, pero lo que sí sé es que no la llevaré a ningún sitio hahahaha",
        "Ah, has terminado los mensajes, haha, ¡cuánto chateas, ha!",
        "Estoy muy aburrida, y no debería estar aquí, haha",
        "Con tu Fe, debes recibir, así que no te dejes engañar, con Dios todo es posible. #hapibufdedamooz",
        "Revendedores, seguramente, tratando de aplastar a la competencia y quedarse con más.",
        "Después del examen, cuando todos se dicen las respuestas y tú no escribiste nada parecido",
        "Lo sé. Estoy pensando que estoy bien para caminar otra vez hacia afuera y eso.",
        "¿Soy yo, o hace muchísimo calor? ¡Mierda!",
        "El puñetero video de Zayn y Louis fumando, me muero de la risa haha",
        "Oye, GMA, reproducir el video de Zouis no te hace un mejor canar, ¿ok? Adiós -Zel",
        "Kim Kardashian y Kanye West se casaron!",
        "Jajajajaj ¿te acuerdas de cuando castigaron a Harry por decir 'coño' en vivo en la televisión?",
        "¡Voy a tratar de usarlo en las noticias mañana! Haha",
        "Otro tweet ganado en cinco minutos, ponte a estalkear o te lo pierdes.",
        "Me siento como Malcolm, el de en medio hahaha",
        "Me muero de la risa jajajajaja, eso fue un buen chiste",
        "Mi mamá me acaba de preguntar si los chicos de mi escuela se rasuran las axilas... Mmm ¿cómo?",
    ]
    es_noisy = [
        "seee pero igual es super loco lol",
        "haha ah okk ! siii , hace tiempo ya q salen !! son monisimos !!!",
        "mejor snapchat q vi en todo el diaaaaaaa lmfao",
        "lol . no te hagas el mono aqui porfa",
        "ni se xq me molesto ... ayayayya",
        "umm me parecioooooooo queesaaaaaa platika era bastante graciosa asiqqqqqqq",
        "dejad de ser tan pto gays los tres no lo sois no tenemos tiempo pa esto",
        "toy mpezando a pensar q soy demasiado pa ti",
        "nse . pero si q se q no la llevo a ningun sitio njasjajabncj",
        "ah has terminao los msjes lol mae mia cuanto chateas ha ! :p",
        "stoy mega aburrida y no debería star aki lol",
        "con tu fe , debs recivir , asiqqq no te dejes engañar, con dios, todo es posible. #hapibufdedamooz!?",
        "seguro revenderores , jjaj tratand de aplastar a la competencia y quedarse con más .",
        "después del examen cuando todos se diceeeeeen las respuestas y no escrbiste nada parecido",
        "lo se . stoy pensando q toy bn para caminar otra vz afuera y kk",
        "soy yo o hace un puto calor !? mierda !!!!",
        "el pñtero video de zayn y louis fumndo me meo de risa haha",
        "ey gma , pasar el video de zouis no t hace un mejor canal ok bye -zel",
        "kim kardashian & kanye weeest se casaronnnn !",
        "lmfao t acuerdas cuando a harry lo castigaron x decir ' coño ' en vivo en la tele",
        "voy a tratar de usarlo en las notiicas mñn ! haha x",
        "otro tuit ganado eeeeeen 5 mins ponte a stalkear o te lo pierdssss",
        "me 100to cmo malcolm el d en medio loool",
        "me meo de risaaaaa lmfaoo qué buen chisteeeee",
        "mi mama m acaba de preguntar si los chikos de mi scuela se rasuran el sobaco ... perrrdona ????!?!?",
    ]

    tokenizer.src_lang = "spa_Latn"
    diffs = []
    with torch.no_grad():
        for clean, noisy in zip(es_clean, es_noisy):
            enc_c = tokenizer(clean, return_tensors="pt", truncation=True,
                              max_length=128, padding=True).to(device)
            enc_n = tokenizer(noisy, return_tensors="pt", truncation=True,
                              max_length=128, padding=True).to(device)
            h_c = model.model.encoder(**enc_c).last_hidden_state
            h_n = model.model.encoder(**enc_n).last_hidden_state
            mask_c = enc_c["attention_mask"].unsqueeze(-1)
            mask_n = enc_n["attention_mask"].unsqueeze(-1)
            pooled_c = (h_c * mask_c).sum(dim=1) / mask_c.sum(dim=1)
            pooled_n = (h_n * mask_n).sum(dim=1) / mask_n.sum(dim=1)
            diff = pooled_n - pooled_c
            if len(diffs) < 3:
                print(f"    Pair {len(diffs)}: diff magnitude={diff.norm().item():.4f}, first 5 values={diff[0,:5].tolist()}")
            diffs.append(diff.cpu())

    direction = torch.stack(diffs).mean(dim=0).squeeze()
    print(f"  ES noise direction magnitude: {direction.norm().item():.4f}")
    print(f"  Direction shape: {direction.shape}")
    return direction


# ============================================================
# 2. Dataset
# ============================================================

class CleanNoisyDataset(Dataset):
    """Clean→noisy parallel pairs for LoRA fine-tuning."""
    def __init__(self, path, tokenizer, max_length=128, skip_identical=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                clean = item["clean"]
                noisy = item["noisy"]
                if skip_identical and clean == noisy:
                    continue
                self.items.append((clean, noisy))

        print(f"  Loaded {len(self.items)} clean→noisy pairs from {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn_factory(tokenizer, spa_noisy_id, max_length=128):
    """Collate: all targets get spa_noisy_Latn as BOS."""
    def collate_fn(batch):
        sources, targets = zip(*batch)

        tokenizer.src_lang = "spa_Latn"
        inputs = tokenizer(
            list(sources),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        tokenizer.tgt_lang = "spa_Latn"
        labels = tokenizer(
            text_target=list(targets),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        label_ids = labels["input_ids"]
        label_ids[:, 0] = spa_noisy_id
        label_ids[label_ids == tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": label_ids,
        }

    return collate_fn


# ============================================================
# 3. LoRA
# ============================================================

def apply_lora(model, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "encoder" in name and "shared" not in name:
            param.requires_grad = False

    model.print_trainable_parameters()
    return model

# ============================================================
# 4. Training loop
# ============================================================

def train(model, train_loader, val_loader, optimizer, scheduler, device,
          epochs, output_dir, tokenizer, spa_noisy_id, grad_accum=1):

    best_val_loss = float("inf")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += outputs.loss.item()
            n_batches += 1

            if n_batches % 100 == 0:
                print(f"    Epoch {epoch} | Batch {n_batches}/{len(train_loader)} | "
                      f"Loss: {outputs.loss.item():.4f} | Avg: {total_loss/n_batches:.4f}")

        avg_train_loss = total_loss / n_batches

        # Validation
        val_loss = evaluate_loss(model, val_loader, device)

        print(f"  Epoch {epoch}/{epochs} | Train loss: {avg_train_loss:.4f} | Val loss: {val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(output_dir, "best")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"    → Saved best model (val_loss={val_loss:.4f})")

            if device.type == "mps":
                torch.mps.empty_cache()
                gc.collect()

    # Save final
    save_path = os.path.join(output_dir, "final")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  Saved final model to {save_path}")


def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


# ============================================================
# 5. Evaluation
# ============================================================

EVAL_EN_NOISY_PATH = "informal-MT-with-no-parallel-data/data/handwritten/en_pairs.jsonl"
EVAL_ES_NOISY_PATH = "informal-MT-with-no-parallel-data/data/handwritten/es_pairs.jsonl"


def load_eval_pairs():
    """Load the 45 parallel handwritten eval pairs (EN and ES, clean+noisy)."""
    en_clean, en_noisy, es_clean, es_noisy = [], [], [], []
    with open(EVAL_EN_NOISY_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            en_clean.append(item["clean"])
            en_noisy.append(item["noisy"])
    with open(EVAL_ES_NOISY_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            es_clean.append(item["clean"])
            es_noisy.append(item["noisy"])
    assert len(en_clean) == len(es_clean), (
        f"EN ({len(en_clean)}) and ES ({len(es_clean)}) eval files must have same length"
    )
    print(f"  Loaded {len(en_clean)} parallel eval pairs (EN+ES, clean+noisy)")
    return en_clean, en_noisy, es_clean, es_noisy


def _generate(model, tokenizer, device, texts, src_lang, forced_bos_token_id):
    """Generate translations for a list of source texts."""
    outputs = []
    tokenizer.src_lang = src_lang
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=128, padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=128,
            )
        outputs.append(tokenizer.decode(gen[0], skip_special_tokens=True))
    return outputs


def evaluate_all(model, tokenizer, device, spa_noisy_id, output_dir, init_name):
    """
    Comprehensive evaluation on 45 handwritten parallel pairs.

    Scores (BLEU + chrF vs reference noisy ES):
      1. EN→ES baseline:  noisy EN → base NLLB (LoRA OFF, spa_Latn)
      2. EN→ES LoRA:      noisy EN → LoRA ON (spa_noisy_Latn)
      3. ES→ES baseline:  clean ES → base NLLB (LoRA OFF, spa_Latn)
      4. ES→ES LoRA:      clean ES → LoRA ON (spa_noisy_Latn)

    Prints side-by-side examples and writes all predictions to JSONL.
    """
    from sacrebleu.metrics import BLEU, CHRF

    model.eval()
    spa_id = tokenizer.convert_tokens_to_ids("spa_Latn")
    en_clean, en_noisy, es_clean, es_noisy = load_eval_pairs()
    bleu = BLEU()
    chrf = CHRF()

    # ── 1. EN→ES baseline: noisy EN → base NLLB (LoRA OFF, spa_Latn) ────
    model.disable_adapter_layers()
    en_es_baseline = _generate(model, tokenizer, device, en_noisy, "eng_Latn", spa_id)

    # ── 2. EN→ES LoRA: noisy EN → LoRA ON (spa_noisy_Latn) ──────────────
    model.enable_adapter_layers()
    en_es_lora = _generate(model, tokenizer, device, en_noisy, "eng_Latn", spa_noisy_id)

    # ── 3. ES→ES baseline: clean ES → base NLLB (LoRA OFF, spa_Latn) ────
    model.disable_adapter_layers()
    es_es_baseline = _generate(model, tokenizer, device, es_clean, "spa_Latn", spa_id)

    # ── 4. ES→ES LoRA: clean ES → LoRA ON (spa_noisy_Latn) ──────────────
    model.enable_adapter_layers()
    es_es_lora = _generate(model, tokenizer, device, es_clean, "spa_Latn", spa_noisy_id)

    # ── Metrics ──────────────────────────────────────────────────────────
    m = {}
    m["en_es_baseline_bleu"] = bleu.corpus_score(en_es_baseline, [es_noisy]).score
    m["en_es_baseline_chrf"] = chrf.corpus_score(en_es_baseline, [es_noisy]).score
    m["en_es_lora_bleu"]     = bleu.corpus_score(en_es_lora, [es_noisy]).score
    m["en_es_lora_chrf"]     = chrf.corpus_score(en_es_lora, [es_noisy]).score
    m["es_es_baseline_bleu"] = bleu.corpus_score(es_es_baseline, [es_noisy]).score
    m["es_es_baseline_chrf"] = chrf.corpus_score(es_es_baseline, [es_noisy]).score
    m["es_es_lora_bleu"]     = bleu.corpus_score(es_es_lora, [es_noisy]).score
    m["es_es_lora_chrf"]     = chrf.corpus_score(es_es_lora, [es_noisy]).score

    # ── Print results ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  EVALUATION — {init_name}  ({len(en_noisy)} pairs)")
    print(f"{'='*80}")

    print(f"\n  {'Task':<22} {'BLEU':>8} {'chrF':>8}")
    print(f"  {'-'*40}")
    print(f"  {'EN→ES baseline':<22} {m['en_es_baseline_bleu']:>8.2f} {m['en_es_baseline_chrf']:>8.2f}")
    print(f"  {'EN→ES LoRA':<22} {m['en_es_lora_bleu']:>8.2f} {m['en_es_lora_chrf']:>8.2f}")
    print(f"  {'EN→ES Δ':<22} {m['en_es_lora_bleu'] - m['en_es_baseline_bleu']:>+8.2f} {m['en_es_lora_chrf'] - m['en_es_baseline_chrf']:>+8.2f}")
    print(f"  {'-'*40}")
    print(f"  {'ES→ES baseline':<22} {m['es_es_baseline_bleu']:>8.2f} {m['es_es_baseline_chrf']:>8.2f}")
    print(f"  {'ES→ES LoRA':<22} {m['es_es_lora_bleu']:>8.2f} {m['es_es_lora_chrf']:>8.2f}")
    print(f"  {'ES→ES Δ':<22} {m['es_es_lora_bleu'] - m['es_es_baseline_bleu']:>+8.2f} {m['es_es_lora_chrf'] - m['es_es_baseline_chrf']:>+8.2f}")

    # ── Side-by-side examples (first 10 EN→ES + first 5 ES→ES) ──────────
    print(f"\n  --- EN→ES examples (first 10) ---")
    for i in range(min(10, len(en_noisy))):
        print(f"\n  [{i+1:2d}] EN noisy:    {en_noisy[i]}")
        print(f"       Ref ES noisy: {es_noisy[i]}")
        print(f"       → baseline:   {en_es_baseline[i]}")
        print(f"       → LoRA:       {en_es_lora[i]}")

    print(f"\n  --- ES→ES examples (first 10) ---")
    for i in range(min(10, len(es_clean))):
        print(f"\n  [{i+1:2d}] ES clean:    {es_clean[i]}")
        print(f"       Ref ES noisy: {es_noisy[i]}")
        print(f"       → baseline:   {es_es_baseline[i]}")
        print(f"       → LoRA:       {es_es_lora[i]}")

    # ── Write all predictions to JSONL ───────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eval_45_pairs.jsonl")
    with open(out_path, "w") as f:
        for i in range(len(en_noisy)):
            f.write(json.dumps({
                "en_noisy": en_noisy[i],
                "en_clean": en_clean[i],
                "es_noisy_ref": es_noisy[i],
                "es_clean_ref": es_clean[i],
                "en_es_baseline": en_es_baseline[i],
                "en_es_lora": en_es_lora[i],
                "es_es_baseline": es_es_baseline[i],
                "es_es_lora": es_es_lora[i],
            }, ensure_ascii=False) + "\n")
    print(f"\n  Wrote predictions to {out_path}")

    return m


# ============================================================
# 6. Single experiment runner
# ============================================================

def run_experiment(args, device, init_name, noise_direction=None):
    """
    Run one full training experiment with a fresh model.

    Args:
        init_name: label for this run ("direction_init" or "random_init")
        noise_direction: vector to add to spa_Latn embedding, or None for
                         plain copy (spa_Latn only, no offset).
    """
    banner = f"EXPERIMENT: {init_name}"
    print(f"\n{'='*80}")
    print(f"  {banner}")
    print(f"{'='*80}")

    output_dir = os.path.join(args.output_dir, init_name)

    # Fresh model + tokenizer for this experiment
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Register spa_noisy_Latn with the specified init
    print(f"\n  Registering spa_noisy_Latn ({init_name})...")
    spa_noisy_id = register_spa_noisy(tokenizer, model, noise_direction=noise_direction)

    model = model.to(device)

    if args.eval_only:
        load_path = os.path.join(output_dir, "best")
        print(f"  Loading LoRA adapter from {load_path}")
        model = PeftModel.from_pretrained(model, load_path).to(device)
        metrics = evaluate_all(model, tokenizer, device, spa_noisy_id, output_dir, init_name)
        return metrics

    print("\n  Applying LoRA (decoder only)...")
    model = apply_lora(model, lora_r=16, lora_alpha=32)

    print("\n  Loading data...")
    train_dataset = CleanNoisyDataset(args.train_path, tokenizer, args.max_length)
    val_dataset = CleanNoisyDataset(args.test_path, tokenizer, args.max_length)

    collate_fn = collate_fn_factory(tokenizer, spa_noisy_id, args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"  Train: {len(train_dataset)} pairs → {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} pairs → {len(val_loader)} batches")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n  Training for {args.epochs} epochs ({total_steps} steps, {warmup_steps} warmup)...")

    train(model, train_loader, val_loader, optimizer, scheduler, device,
          args.epochs, output_dir, tokenizer, spa_noisy_id,
          grad_accum=args.grad_accum)

    # Evaluation on 45 parallel pairs (EN→ES + ES→ES, with baselines)
    spa_noisy_id = tokenizer.convert_tokens_to_ids("spa_noisy_Latn")
    metrics = evaluate_all(model, tokenizer, device, spa_noisy_id, output_dir, init_name)

    # Free memory before next experiment
    del model, optimizer, scheduler
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return metrics


# ============================================================
# 7. Main — ablation: direction_init vs random_init
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune NLLB decoder for spa_noisy_Latn")
    parser.add_argument("--train_path", type=str,
                        default="data/multilexnorm/es_train_augmented.jsonl")
    parser.add_argument("--test_path", type=str,
                        default="data/multilexnorm/es_test_augmented.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/spa_noisy")
    parser.add_argument("--model_name", type=str,
                        default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true",
                        help="Load best checkpoint and run eval only")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Compute ES noise direction once (needs a temporary model on device)
    print(f"\nLoading {args.model_name} to compute noise direction...")
    tokenizer_tmp = AutoTokenizer.from_pretrained(args.model_name)
    model_tmp = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    es_direction = compute_es_noise_direction(tokenizer_tmp, model_tmp, device)
    del model_tmp, tokenizer_tmp
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Random vector A: same magnitude as noise direction, random orientation
    random_same_mag = torch.randn_like(es_direction)
    random_same_mag = random_same_mag / random_same_mag.norm() * es_direction.norm()

    # Random vector B: uninformed — uses model's embedding init scale, no noise info
    init_std = 0.02  # standard transformer init scale
    random_uninformed = torch.randn_like(es_direction) * init_std
    # magnitude will be ~sqrt(1024)*0.02 ≈ 0.64

    print(f"\n  Noise direction magnitude:      {es_direction.norm().item():.4f}")
    print(f"  Random (same mag) magnitude:    {random_same_mag.norm().item():.4f}")
    print(f"  Random (uninformed) magnitude:  {random_uninformed.norm().item():.4f}")
    cos_dir_samemag = torch.nn.functional.cosine_similarity(
        es_direction.unsqueeze(0), random_same_mag.unsqueeze(0)
    ).item()
    cos_dir_uninf = torch.nn.functional.cosine_similarity(
        es_direction.unsqueeze(0), random_uninformed.unsqueeze(0)
    ).item()
    print(f"  Cosine(direction, random_same_mag):   {cos_dir_samemag:.4f}")
    print(f"  Cosine(direction, random_uninformed): {cos_dir_uninf:.4f}")

    # --- Experiment 1: noise direction init ---
    metrics_dir = run_experiment(args, device, "direction_init", noise_direction=es_direction)

    # --- Experiment 2: random init (same magnitude as noise direction) ---
    metrics_rand = run_experiment(args, device, "random_same_mag", noise_direction=random_same_mag)

    # --- Experiment 3: random init (uninformed — different direction AND magnitude) ---
    metrics_uninf = run_experiment(args, device, "random_uninformed", noise_direction=random_uninformed)

    # --- Comparison ---
    all_metrics = {
        "direction_init": metrics_dir,
        "random_same_mag": metrics_rand,
        "random_uninformed": metrics_uninf,
    }
    print(f"\n{'='*80}")
    print("  ABLATION COMPARISON (45 parallel pairs)")
    print(f"{'='*80}")
    if all(v is not None for v in all_metrics.values()):
        # EN→ES table
        print(f"\n  EN→ES (noisy EN → noisy ES)")
        header = f"  {'':>22}"
        for name in all_metrics:
            header += f" {name:>18}"
        print(header)
        print(f"  {'-'*80}")
        print(f"  {'baseline BLEU':<22}" + "".join(f" {all_metrics[n]['en_es_baseline_bleu']:>18.2f}" for n in all_metrics))
        print(f"  {'baseline chrF':<22}" + "".join(f" {all_metrics[n]['en_es_baseline_chrf']:>18.2f}" for n in all_metrics))
        print(f"  {'LoRA BLEU':<22}" + "".join(f" {all_metrics[n]['en_es_lora_bleu']:>18.2f}" for n in all_metrics))
        print(f"  {'LoRA chrF':<22}" + "".join(f" {all_metrics[n]['en_es_lora_chrf']:>18.2f}" for n in all_metrics))
        print(f"  {'Δ BLEU':<22}" + "".join(f" {all_metrics[n]['en_es_lora_bleu'] - all_metrics[n]['en_es_baseline_bleu']:>+18.2f}" for n in all_metrics))
        print(f"  {'Δ chrF':<22}" + "".join(f" {all_metrics[n]['en_es_lora_chrf'] - all_metrics[n]['en_es_baseline_chrf']:>+18.2f}" for n in all_metrics))

        # ES→ES table
        print(f"\n  ES→ES (clean ES → noisy ES)")
        print(header)
        print(f"  {'-'*80}")
        print(f"  {'baseline BLEU':<22}" + "".join(f" {all_metrics[n]['es_es_baseline_bleu']:>18.2f}" for n in all_metrics))
        print(f"  {'baseline chrF':<22}" + "".join(f" {all_metrics[n]['es_es_baseline_chrf']:>18.2f}" for n in all_metrics))
        print(f"  {'LoRA BLEU':<22}" + "".join(f" {all_metrics[n]['es_es_lora_bleu']:>18.2f}" for n in all_metrics))
        print(f"  {'LoRA chrF':<22}" + "".join(f" {all_metrics[n]['es_es_lora_chrf']:>18.2f}" for n in all_metrics))
        print(f"  {'Δ BLEU':<22}" + "".join(f" {all_metrics[n]['es_es_lora_bleu'] - all_metrics[n]['es_es_baseline_bleu']:>+18.2f}" for n in all_metrics))
        print(f"  {'Δ chrF':<22}" + "".join(f" {all_metrics[n]['es_es_lora_chrf'] - all_metrics[n]['es_es_baseline_chrf']:>+18.2f}" for n in all_metrics))

    print(f"\n  Predictions:")
    for name in all_metrics:
        print(f"    {args.output_dir}/{name}/eval_45_pairs.jsonl")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
