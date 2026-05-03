"""
PyTorch dataset for shifted-encoder decoder fine-tuning.

Each example:
  - encoder_input_ids / encoder_attention_mask  (clean sentence, for the encoder)
  - labels                                       (noisy sentence, for the decoder)

The shift is applied at training time inside the model forward pass,
not here in the dataset.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class NoisyNormDataset(Dataset):
    def __init__(
        self,
        paths: list[str | Path],
        tokenizer: PreTrainedTokenizerBase,
        src_lang: str,
        tgt_lang: str,
        max_length: int = 128,
    ):
        """
        paths:    one or more JSONL files with {"clean": ..., "noisy": ...}
        src_lang: NLLB language code for the clean (encoder) side
        tgt_lang: NLLB language code for the noisy (decoder target) side
                  Typically same as src_lang (EN→noisy-EN, ES→noisy-ES).
        """
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        self.pairs: list[dict] = []

        for path in paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    self.pairs.append(json.loads(line))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        clean = pair["clean"]
        noisy = pair["noisy"]

        self.tokenizer.src_lang = self.src_lang
        enc = self.tokenizer(
            clean,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        # For the decoder target we use the same tokenizer but force the target
        # language token via the forced_bos approach: prepend the tgt lang token.
        with self.tokenizer.as_target_tokenizer():
            dec = self.tokenizer(
                noisy,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

        labels = dec["input_ids"]
        # Replace padding token id with -100 so it's ignored in cross-entropy loss.
        labels = [t if t != self.tokenizer.pad_token_id else -100 for t in labels]

        return {
            "input_ids":      torch.tensor(enc["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"],  dtype=torch.long),
            "labels":         torch.tensor(labels,                 dtype=torch.long),
        }


def collate_fn(batch, pad_token_id: int):
    """Pad input_ids, attention_mask, and labels to the longest in the batch."""
    max_enc = max(x["input_ids"].size(0)      for x in batch)
    max_dec = max(x["labels"].size(0)         for x in batch)

    input_ids      = torch.full((len(batch), max_enc), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_enc,               dtype=torch.long)
    labels         = torch.full((len(batch), max_dec), -100,         dtype=torch.long)

    for i, x in enumerate(batch):
        el = x["input_ids"].size(0)
        dl = x["labels"].size(0)
        input_ids[i, :el]      = x["input_ids"]
        attention_mask[i, :el] = x["attention_mask"]
        labels[i, :dl]         = x["labels"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def make_collate_fn(pad_token_id: int):
    return lambda batch: collate_fn(batch, pad_token_id)
