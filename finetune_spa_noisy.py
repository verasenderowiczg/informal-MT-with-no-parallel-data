"""
Fine-tune NLLB for informal text generation with three architecture modes:

  decoder_only  — LoRA on decoder q_proj/v_proj, encoder frozen (ES clean→noisy)
  encoder_only  — LoRA on encoder q_proj/v_proj, decoder frozen (EN clean→noisy)
  combined      — LoRA on both encoder+decoder (mixed EN+ES clean→noisy batches)

Each mode is tested with 3 embedding initialization strategies (direction_init,
random_same_mag, random_uninformed) for ablation.

Usage examples:

    # Decoder-only (original mode)
    python finetune_spa_noisy.py --mode decoder_only \
        --train_path data/multilexnorm/es_combined_train_5k.jsonl \
        --test_path data/multilexnorm/es_test_augmented.jsonl \
        --output_dir checkpoints/decoder_only_5k

    # Encoder-only (cross-lingual transfer)
    python finetune_spa_noisy.py --mode encoder_only \
        --en_train_path data/multilexnorm/en_combined_train_5k.jsonl \
        --en_test_path data/multilexnorm/en_test_augmented.jsonl \
        --output_dir checkpoints/encoder_only_5k

    # Combined (both encoder+decoder)
    python finetune_spa_noisy.py --mode combined \
        --train_path data/multilexnorm/es_combined_train_5k.jsonl \
        --test_path data/multilexnorm/es_test_augmented.jsonl \
        --en_train_path data/multilexnorm/en_combined_train_5k.jsonl \
        --en_test_path data/multilexnorm/en_test_augmented.jsonl \
        --output_dir checkpoints/combined_5k
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

def register_eng_noisy(tokenizer, model, noise_direction=None):
    """
    Add eng_noisy_Latn as a new language token, initialized
    as a copy of eng_Latn's embedding.
    """
    new_lang = "eng_noisy_Latn"

    if new_lang in tokenizer.get_vocab():
        print(f"  {new_lang} already in vocab, skipping registration.")
        return tokenizer.convert_tokens_to_ids(new_lang)

    tokenizer.add_special_tokens({"additional_special_tokens": [new_lang]})
    model.resize_token_embeddings(len(tokenizer))

    eng_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    new_id = tokenizer.convert_tokens_to_ids(new_lang)

    with torch.no_grad():
        if noise_direction is not None:
            model.model.shared.weight[new_id] = model.model.shared.weight[eng_id].clone() + noise_direction.to(model.model.shared.weight.device)
            print(f"  Initialized with noise direction (magnitude {noise_direction.norm().item():.4f})")
        else:
            model.model.shared.weight[new_id] = model.model.shared.weight[eng_id].clone()
        if not model.config.tie_word_embeddings:
            model.lm_head.weight[new_id] = model.lm_head.weight[eng_id].clone()

    print(f"  Registered {new_lang} (id={new_id}), initialized from eng_Latn (id={eng_id})")
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


def compute_en_noise_direction(tokenizer, model, device):
    """Compute the EN clean→noisy direction from the handwritten EN pairs."""
    en_path = "data/handwritten/en_pairs.jsonl"
    en_clean, en_noisy = [], []
    with open(en_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            en_clean.append(item["clean"])
            en_noisy.append(item["noisy"])

    print(f"  Computing EN noise direction from {len(en_clean)} pairs...")
    tokenizer.src_lang = "eng_Latn"
    diffs = []
    with torch.no_grad():
        for clean, noisy in zip(en_clean, en_noisy):
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
    print(f"  EN noise direction magnitude: {direction.norm().item():.4f}")
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


def collate_fn_factory(tokenizer, noisy_bos_id, src_lang, max_length=128):
    """Collate: all targets get the specified noisy BOS token."""
    def collate_fn(batch):
        sources, targets = zip(*batch)

        tokenizer.src_lang = src_lang
        inputs = tokenizer(
            list(sources),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        tokenizer.tgt_lang = src_lang
        labels = tokenizer(
            text_target=list(targets),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        label_ids = labels["input_ids"]
        label_ids[:, 0] = noisy_bos_id
        label_ids[label_ids == tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": label_ids,
        }

    return collate_fn


def encoder_only_collate_factory(tokenizer, eng_noisy_id, eng_id, max_length=128):
    """
    Collate for encoder-only: source is NOISY EN (tagged eng_noisy_Latn),
    target is CLEAN EN (BOS eng_Latn). Teaches the encoder what the new
    'informal English' language means.
    """
    def collate_fn(batch):
        cleans, noisys = zip(*batch)

        tokenizer.src_lang = "eng_noisy_Latn"
        inputs = tokenizer(
            list(noisys),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        tokenizer.tgt_lang = "eng_Latn"
        labels = tokenizer(
            text_target=list(cleans),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        label_ids = labels["input_ids"]
        label_ids[:, 0] = eng_id
        label_ids[label_ids == tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": label_ids,
        }

    return collate_fn


class CombinedDataset(Dataset):
    """Interleave EN and ES clean->noisy pairs for combined training."""
    def __init__(self, en_dataset, es_dataset):
        self.en_items = en_dataset.items
        self.es_items = es_dataset.items
        self.all_items = []
        for clean, noisy in self.en_items:
            self.all_items.append((clean, noisy, "en"))
        for clean, noisy in self.es_items:
            self.all_items.append((clean, noisy, "es"))
        random.shuffle(self.all_items)
        print(f"  Combined dataset: {len(self.en_items)} EN + {len(self.es_items)} ES = {len(self.all_items)} total")

    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, idx):
        return self.all_items[idx]


def combined_collate_fn_factory(tokenizer, eng_noisy_id, eng_id, spa_noisy_id, max_length=128):
    """
    Collate for combined mode:
      EN pairs: noisy EN (eng_noisy_Latn) → clean EN (eng_Latn BOS) — teach encoder
      ES pairs: clean ES (spa_Latn) → noisy ES (spa_noisy_Latn BOS) — teach decoder
    """
    def collate_fn(batch):
        cleans, noisys, langs = zip(*batch)

        all_input_ids = []
        all_attention_masks = []
        all_label_ids = []

        for clean, noisy, lang in zip(cleans, noisys, langs):
            if lang == "en":
                # Encoder learning: noisy EN input → clean EN target
                tokenizer.src_lang = "eng_noisy_Latn"
                inp = tokenizer(noisy, return_tensors="pt", truncation=True, max_length=max_length)
                tokenizer.tgt_lang = "eng_Latn"
                lab = tokenizer(text_target=clean, return_tensors="pt", truncation=True, max_length=max_length)
                bos_id = eng_id
            else:
                # Decoder learning: clean ES input → noisy ES target
                tokenizer.src_lang = "spa_Latn"
                inp = tokenizer(clean, return_tensors="pt", truncation=True, max_length=max_length)
                tokenizer.tgt_lang = "spa_Latn"
                lab = tokenizer(text_target=noisy, return_tensors="pt", truncation=True, max_length=max_length)
                bos_id = spa_noisy_id

            all_input_ids.append(inp["input_ids"].squeeze(0))
            all_attention_masks.append(inp["attention_mask"].squeeze(0))

            lab_ids = lab["input_ids"].squeeze(0)
            lab_ids[0] = bos_id
            all_label_ids.append(lab_ids)

        input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(all_attention_masks, batch_first=True, padding_value=0)
        label_ids = torch.nn.utils.rnn.pad_sequence(all_label_ids, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }

    return collate_fn


# ============================================================
# 3. LoRA
# ============================================================

def save_new_token_embeddings(model, tokenizer, save_path):
    """Save only the new token embeddings (spa_noisy_Latn, eng_noisy_Latn) to a small file."""
    new_tokens = ["spa_noisy_Latn", "eng_noisy_Latn"]
    embeddings = {}
    for token in new_tokens:
        if token in tokenizer.get_vocab():
            token_id = tokenizer.convert_tokens_to_ids(token)
            embeddings[token] = model.get_base_model().model.shared.weight[token_id].detach().cpu()
    if embeddings:
        torch.save(embeddings, os.path.join(save_path, "new_token_embeddings.pt"))


def load_new_token_embeddings(model, tokenizer, ckpt_path):
    """Load saved new token embeddings back into the model."""
    emb_path = os.path.join(ckpt_path, "new_token_embeddings.pt")
    if not os.path.exists(emb_path):
        return
    embeddings = torch.load(emb_path, map_location="cpu")
    with torch.no_grad():
        for token, emb in embeddings.items():
            if token in tokenizer.get_vocab():
                token_id = tokenizer.convert_tokens_to_ids(token)
                model.model.shared.weight[token_id] = emb.to(model.model.shared.weight.device)
    print(f"  Loaded new token embeddings: {list(embeddings.keys())}")


def apply_lora(model, mode="decoder_only", lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """
    Apply LoRA adapters based on architecture mode.

    Modes:
      decoder_only  — LoRA on decoder q_proj/v_proj, encoder frozen
      encoder_only  — LoRA on encoder q_proj/v_proj, decoder frozen
      combined      — LoRA on both encoder AND decoder q_proj/v_proj
    """
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

    if mode == "decoder_only":
        for name, param in model.named_parameters():
            if "encoder" in name and "shared" not in name:
                param.requires_grad = False
    elif mode == "encoder_only":
        for name, param in model.named_parameters():
            if "decoder" in name and "shared" not in name:
                param.requires_grad = False
    # combined: both encoder and decoder LoRA params stay trainable

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
            save_new_token_embeddings(model, tokenizer, save_path)
            print(f"    → Saved best model (val_loss={val_loss:.4f})")

            if device.type == "mps":
                torch.mps.empty_cache()
                gc.collect()

    # Save final
    save_path = os.path.join(output_dir, "final")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    save_new_token_embeddings(model, tokenizer, save_path)
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

EVAL_EN_NOISY_PATH = "data/handwritten/en_pairs.jsonl"
EVAL_ES_NOISY_PATH = "data/handwritten/es_pairs.jsonl"


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


def evaluate_all(model, tokenizer, device, spa_noisy_id, output_dir, init_name,
                  mode="decoder_only", eng_noisy_id=None):
    """
    Comprehensive evaluation on handwritten parallel pairs.

    Eval data: data/handwritten/{en,es}_pairs.jsonl (45-46 pairs)
    These are NEVER part of training data — training uses MultiLexNorm +
    synthetic OpenSubtitles data; eval uses separate handwritten pairs.

    Always computed (all modes):
      1. Baseline:     noisy EN (eng_Latn) → spa_Latn, LoRA OFF
      2. Target task:  noisy EN (eng_noisy_Latn) → spa_noisy_Latn, LoRA ON
      3. ES noisify:   clean ES (spa_Latn) → spa_noisy_Latn, LoRA ON

    Mode-specific extras:
      decoder_only:  noisy EN (eng_Latn) → spa_noisy_Latn, LoRA ON
      encoder_only:  noisy EN (eng_noisy_Latn) → spa_Latn, LoRA ON
      combined:      (target task already covers it)

    All scored against handwritten noisy ES references.
    """
    from sacrebleu.metrics import BLEU, CHRF

    model.eval()
    spa_id = tokenizer.convert_tokens_to_ids("spa_Latn")
    eng_noisy_id_resolved = eng_noisy_id or tokenizer.convert_tokens_to_ids("eng_noisy_Latn")
    en_clean, en_noisy, es_clean, es_noisy = load_eval_pairs()
    bleu = BLEU()
    chrf = CHRF()

    # ══════════════════════════════════════════════════════════════════════
    # UNIVERSAL EVALS (all modes)
    # ══════════════════════════════════════════════════════════════════════

    # 1. Baseline: noisy EN (eng_Latn) → spa_Latn, LoRA OFF
    model.disable_adapter_layers()
    baseline = _generate(model, tokenizer, device, en_noisy, "eng_Latn", spa_id)

    # 2. Target task: noisy EN (eng_noisy_Latn) → spa_noisy_Latn, LoRA ON
    model.enable_adapter_layers()
    target_task = _generate(model, tokenizer, device, en_noisy, "eng_noisy_Latn", spa_noisy_id)

    # 3. ES noisification: clean ES (spa_Latn) → spa_noisy_Latn, LoRA ON
    es_noisify = _generate(model, tokenizer, device, es_clean, "spa_Latn", spa_noisy_id)

    # ══════════════════════════════════════════════════════════════════════
    # MODE-SPECIFIC EXTRAS
    # ══════════════════════════════════════════════════════════════════════

    mode_extra = None
    mode_extra_label = ""

    if mode == "decoder_only":
        # eng_Latn → spa_noisy_Latn (decoder's natural setup, no eng_noisy_Latn)
        mode_extra = _generate(model, tokenizer, device, en_noisy, "eng_Latn", spa_noisy_id)
        mode_extra_label = "EN(eng_Latn)→spa_noisy"

    elif mode == "encoder_only":
        # eng_noisy_Latn → spa_Latn (encoder's natural setup, tests understanding)
        mode_extra = _generate(model, tokenizer, device, en_noisy, "eng_noisy_Latn", spa_id)
        mode_extra_label = "EN(eng_noisy)→spa_Latn"

    # Denoising quality: eng_noisy_Latn → eng_Latn (encoder's training task)
    denoise_output = None
    if mode in ("encoder_only", "combined"):
        eng_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        denoise_output = _generate(model, tokenizer, device, en_noisy, "eng_noisy_Latn", eng_id)

    # ══════════════════════════════════════════════════════════════════════
    # METRICS (all against noisy ES references)
    # ══════════════════════════════════════════════════════════════════════

    m = {}
    m["baseline_bleu"]     = bleu.corpus_score(baseline, [es_noisy]).score
    m["baseline_chrf"]     = chrf.corpus_score(baseline, [es_noisy]).score
    m["target_bleu"]       = bleu.corpus_score(target_task, [es_noisy]).score
    m["target_chrf"]       = chrf.corpus_score(target_task, [es_noisy]).score
    m["es_noisify_bleu"]   = bleu.corpus_score(es_noisify, [es_noisy]).score
    m["es_noisify_chrf"]   = chrf.corpus_score(es_noisify, [es_noisy]).score

    if mode_extra is not None:
        m["mode_extra_bleu"] = bleu.corpus_score(mode_extra, [es_noisy]).score
        m["mode_extra_chrf"] = chrf.corpus_score(mode_extra, [es_noisy]).score

    if denoise_output is not None:
        m["denoise_bleu"] = bleu.corpus_score(denoise_output, [en_clean]).score
        m["denoise_chrf"] = chrf.corpus_score(denoise_output, [en_clean]).score

    # ══════════════════════════════════════════════════════════════════════
    # PRINT RESULTS
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  EVALUATION — {init_name} [{mode}]  ({len(en_noisy)} pairs)")
    print(f"{'='*80}")

    print(f"\n  {'Task':<35} {'BLEU':>8} {'chrF':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Baseline (eng_Latn→spa_Latn)':<35} {m['baseline_bleu']:>8.2f} {m['baseline_chrf']:>8.2f}")
    print(f"  {'Target (eng_noisy→spa_noisy)':<35} {m['target_bleu']:>8.2f} {m['target_chrf']:>8.2f}")
    print(f"  {'Target Δ vs baseline':<35} {m['target_bleu'] - m['baseline_bleu']:>+8.2f} {m['target_chrf'] - m['baseline_chrf']:>+8.2f}")
    print(f"  {'-'*55}")
    print(f"  {'ES noisify (spa_Latn→spa_noisy)':<35} {m['es_noisify_bleu']:>8.2f} {m['es_noisify_chrf']:>8.2f}")

    if mode_extra is not None:
        print(f"  {'-'*55}")
        print(f"  {mode_extra_label:<35} {m['mode_extra_bleu']:>8.2f} {m['mode_extra_chrf']:>8.2f}")

    if denoise_output is not None:
        print(f"  {'-'*55}")
        print(f"  {'Denoise (eng_noisy→eng_Latn vs EN clean)':<35} {m['denoise_bleu']:>8.2f} {m['denoise_chrf']:>8.2f}")

    # ── Side-by-side examples (target task) ──────────────────────────────
    print(f"\n  --- Target task examples (eng_noisy→spa_noisy, first 10) ---")
    for i in range(min(10, len(en_noisy))):
        print(f"\n  [{i+1:2d}] EN noisy:    {en_noisy[i]}")
        print(f"       Ref ES noisy: {es_noisy[i]}")
        print(f"       → baseline:   {baseline[i]}")
        print(f"       → target:     {target_task[i]}")

    print(f"\n  --- ES noisification examples (first 5) ---")
    for i in range(min(5, len(es_clean))):
        print(f"\n  [{i+1:2d}] ES clean:    {es_clean[i]}")
        print(f"       Ref ES noisy: {es_noisy[i]}")
        print(f"       → noisify:    {es_noisify[i]}")

    if mode_extra is not None:
        print(f"\n  --- Mode-specific ({mode_extra_label}, first 5) ---")
        for i in range(min(5, len(en_noisy))):
            print(f"\n  [{i+1:2d}] EN noisy:    {en_noisy[i]}")
            print(f"       Ref ES noisy: {es_noisy[i]}")
            print(f"       → {mode_extra_label}: {mode_extra[i]}")

    # ── Write all predictions to JSONL ───────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eval_predictions.jsonl")
    with open(out_path, "w") as f:
        for i in range(len(en_noisy)):
            record = {
                "en_noisy": en_noisy[i],
                "en_clean": en_clean[i],
                "es_noisy_ref": es_noisy[i],
                "es_clean_ref": es_clean[i],
                "baseline": baseline[i],
                "target_task": target_task[i],
                "es_noisify": es_noisify[i],
            }
            if mode_extra is not None:
                record["mode_extra"] = mode_extra[i]
            if denoise_output is not None:
                record["denoise_output"] = denoise_output[i]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n  Wrote predictions to {out_path}")

    # ── Save metrics to JSON ──────────────────────────────────────────────
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"  Wrote metrics to {metrics_path}")

    return m


# ============================================================
# 6. Single experiment runner
# ============================================================

def run_experiment(args, device, init_name, mode="decoder_only",
                   es_noise_direction=None, en_noise_direction=None):
    """
    Run one full training experiment with a fresh model.

    Args:
        init_name: label for this run ("direction_init", "random_same_mag", etc.)
        mode: "decoder_only", "encoder_only", or "combined"
        es_noise_direction: vector for spa_noisy_Latn init (or None)
        en_noise_direction: vector for eng_noisy_Latn init (or None)
    """
    banner = f"EXPERIMENT: {init_name} [{mode}]"
    print(f"\n{'='*80}")
    print(f"  {banner}")
    print(f"{'='*80}")

    output_dir = os.path.join(args.output_dir, init_name)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Register language tokens — all modes need spa_noisy_Latn for inference target
    spa_noisy_id = None
    eng_noisy_id = None

    print(f"\n  Registering spa_noisy_Latn ({init_name})...")
    spa_noisy_id = register_spa_noisy(tokenizer, model, noise_direction=es_noise_direction)

    if mode in ("encoder_only", "combined"):
        print(f"\n  Registering eng_noisy_Latn ({init_name})...")
        eng_noisy_id = register_eng_noisy(tokenizer, model, noise_direction=en_noise_direction)

    model = model.to(device)

    best_path = os.path.join(output_dir, "best")
    if args.eval_only or os.path.exists(best_path):
        if not args.eval_only:
            print(f"  Checkpoint found at {best_path}, skipping training.")
        print(f"  Loading LoRA adapter from {best_path}")
        model = PeftModel.from_pretrained(model, best_path).to(device)
        metrics = evaluate_all(model, tokenizer, device, spa_noisy_id, output_dir,
                               init_name, mode=mode, eng_noisy_id=eng_noisy_id)
        return metrics

    print(f"\n  Applying LoRA ({mode})...")
    model = apply_lora(model, mode=mode, lora_r=16, lora_alpha=32)

    # Load data based on mode
    print("\n  Loading data...")
    if mode == "decoder_only":
        train_dataset = CleanNoisyDataset(args.train_path, tokenizer, args.max_length)
        val_dataset = CleanNoisyDataset(args.test_path, tokenizer, args.max_length)
        collate_fn = collate_fn_factory(tokenizer, spa_noisy_id, "spa_Latn", args.max_length)

    elif mode == "encoder_only":
        train_dataset = CleanNoisyDataset(args.en_train_path, tokenizer, args.max_length)
        val_dataset = CleanNoisyDataset(args.en_test_path, tokenizer, args.max_length)
        eng_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        collate_fn = encoder_only_collate_factory(tokenizer, eng_noisy_id, eng_id, args.max_length)

    elif mode == "combined":
        en_train = CleanNoisyDataset(args.en_train_path, tokenizer, args.max_length)
        es_train = CleanNoisyDataset(args.train_path, tokenizer, args.max_length)
        train_dataset = CombinedDataset(en_train, es_train)

        en_val = CleanNoisyDataset(args.en_test_path, tokenizer, args.max_length)
        es_val = CleanNoisyDataset(args.test_path, tokenizer, args.max_length)
        val_dataset = CombinedDataset(en_val, es_val)

        eng_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        collate_fn = combined_collate_fn_factory(tokenizer, eng_noisy_id, eng_id, spa_noisy_id, args.max_length)

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
          args.epochs, output_dir, tokenizer,
          spa_noisy_id if spa_noisy_id else eng_noisy_id,
          grad_accum=args.grad_accum)

    # Evaluation
    metrics = evaluate_all(model, tokenizer, device, spa_noisy_id, output_dir,
                           init_name, mode=mode, eng_noisy_id=eng_noisy_id)

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
    parser = argparse.ArgumentParser(description="Fine-tune NLLB for informal text generation")
    parser.add_argument("--mode", type=str, default="decoder_only",
                        choices=["decoder_only", "encoder_only", "combined"],
                        help="Architecture mode: decoder_only, encoder_only, or combined")
    parser.add_argument("--train_path", type=str,
                        default="data/multilexnorm/es_train_augmented.jsonl",
                        help="ES clean→noisy training data")
    parser.add_argument("--test_path", type=str,
                        default="data/multilexnorm/es_test_augmented.jsonl",
                        help="ES clean→noisy validation data")
    parser.add_argument("--en_train_path", type=str,
                        default="data/multilexnorm/en_train_augmented.jsonl",
                        help="EN clean→noisy training data (for encoder_only/combined)")
    parser.add_argument("--en_test_path", type=str,
                        default="data/multilexnorm/en_test_augmented.jsonl",
                        help="EN clean→noisy validation data (for encoder_only/combined)")
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
    parser.add_argument("--init_strategy", type=str, default=None,
                        choices=["direction_init", "random_same_mag", "random_uninformed"],
                        help="Run only this init strategy (default: run all 3)")
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

    print(f"\nMode: {args.mode}")

    # Compute noise directions (needs a temporary model on device)
    print(f"\nLoading {args.model_name} to compute noise directions...")
    tokenizer_tmp = AutoTokenizer.from_pretrained(args.model_name)
    model_tmp = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    es_direction = None
    en_direction = None

    es_direction = compute_es_noise_direction(tokenizer_tmp, model_tmp, device)
    if args.mode in ("encoder_only", "combined"):
        en_direction = compute_en_noise_direction(tokenizer_tmp, model_tmp, device)

    del model_tmp, tokenizer_tmp
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Build init vectors for the 3 ablation conditions
    def make_random_vectors(direction):
        random_same_mag = torch.randn_like(direction)
        random_same_mag = random_same_mag / random_same_mag.norm() * direction.norm()
        init_std = 0.02
        random_uninformed = torch.randn_like(direction) * init_std
        return random_same_mag, random_uninformed

    es_random_same, es_random_uninf = (None, None)
    en_random_same, en_random_uninf = (None, None)

    if es_direction is not None:
        es_random_same, es_random_uninf = make_random_vectors(es_direction)
        print(f"\n  ES noise direction magnitude:      {es_direction.norm().item():.4f}")
        print(f"  ES random (same mag) magnitude:    {es_random_same.norm().item():.4f}")
        print(f"  ES random (uninformed) magnitude:  {es_random_uninf.norm().item():.4f}")

    if en_direction is not None:
        en_random_same, en_random_uninf = make_random_vectors(en_direction)
        print(f"\n  EN noise direction magnitude:      {en_direction.norm().item():.4f}")
        print(f"  EN random (same mag) magnitude:    {en_random_same.norm().item():.4f}")
        print(f"  EN random (uninformed) magnitude:  {en_random_uninf.norm().item():.4f}")

    # 3 init conditions with matching strategies for both languages
    inits = [
        ("direction_init",    es_direction,    en_direction),
        ("random_same_mag",   es_random_same,  en_random_same),
        ("random_uninformed", es_random_uninf, en_random_uninf),
    ]

    if args.init_strategy:
        inits = [(n, e, en) for n, e, en in inits if n == args.init_strategy]

    all_metrics = {}
    for init_name, es_dir, en_dir in inits:
        metrics = run_experiment(
            args, device, init_name, mode=args.mode,
            es_noise_direction=es_dir, en_noise_direction=en_dir,
        )
        all_metrics[init_name] = metrics

    # --- Comparison table ---
    print(f"\n{'='*80}")
    print(f"  ABLATION COMPARISON [{args.mode}] (handwritten eval pairs)")
    print(f"{'='*80}")
    if all(v is not None for v in all_metrics.values()):
        header = f"  {'':>35}"
        for name in all_metrics:
            header += f" {name:>18}"

        print(f"\n  Target task: eng_noisy_Latn → spa_noisy_Latn")
        print(header)
        print(f"  {'-'*90}")
        print(f"  {'baseline BLEU':<35}" + "".join(f" {all_metrics[n]['baseline_bleu']:>18.2f}" for n in all_metrics))
        print(f"  {'baseline chrF':<35}" + "".join(f" {all_metrics[n]['baseline_chrf']:>18.2f}" for n in all_metrics))
        print(f"  {'target BLEU':<35}" + "".join(f" {all_metrics[n]['target_bleu']:>18.2f}" for n in all_metrics))
        print(f"  {'target chrF':<35}" + "".join(f" {all_metrics[n]['target_chrf']:>18.2f}" for n in all_metrics))
        print(f"  {'Δ BLEU':<35}" + "".join(f" {all_metrics[n]['target_bleu'] - all_metrics[n]['baseline_bleu']:>+18.2f}" for n in all_metrics))
        print(f"  {'Δ chrF':<35}" + "".join(f" {all_metrics[n]['target_chrf'] - all_metrics[n]['baseline_chrf']:>+18.2f}" for n in all_metrics))

        print(f"\n  ES noisification: spa_Latn → spa_noisy_Latn")
        print(header)
        print(f"  {'-'*90}")
        print(f"  {'es_noisify BLEU':<35}" + "".join(f" {all_metrics[n]['es_noisify_bleu']:>18.2f}" for n in all_metrics))
        print(f"  {'es_noisify chrF':<35}" + "".join(f" {all_metrics[n]['es_noisify_chrf']:>18.2f}" for n in all_metrics))

        if "mode_extra_bleu" in list(all_metrics.values())[0]:
            print(f"\n  Mode-specific extra")
            print(header)
            print(f"  {'-'*90}")
            print(f"  {'extra BLEU':<35}" + "".join(f" {all_metrics[n]['mode_extra_bleu']:>18.2f}" for n in all_metrics))
            print(f"  {'extra chrF':<35}" + "".join(f" {all_metrics[n]['mode_extra_chrf']:>18.2f}" for n in all_metrics))

    print(f"\n  Predictions:")
    for name in all_metrics:
        print(f"    {args.output_dir}/{name}/eval_predictions.jsonl")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
