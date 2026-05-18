# Bypassing the Parallel Data Constraint: Adapting NLLB-200 for Informal EN–ES Translation with Monolingual Data Alone

**Asier Azpiri Iriarte, Sofía Barajas Pascual, Vera Senderowicz Guerra** — UPV/EHU

---

## Overview

Translating informal, social-media-style text remains a challenge for NMT systems trained on clean parallel corpora. The standard remedy of in-domain fine-tuning is unavailable for most language pairs, including English–Spanish, where informal parallel data do not exist.

This repository contains the code and data for our study, which explores whether NLLB-200 can be adapted to produce informal Spanish from informal English **without parallel data**, using only synthetically generated monolingual clean↔noisy pairs. Key findings:

- **Decoder-only LoRA** fine-tuning is the most promising architecture for informal production, yielding +6.1 BLEU on the cross-lingual task.
- **Encoder-side adaptation does not compose cross-lingually** despite strong monolingual performance — a composition gap not explained by capacity alone.
- **The clean↔noisy direction is geometrically present** in multilingual embedding space (cosine ~0.86 for XLM-R/NLLB) but too variable across sentence pairs to serve as a useful initialization prior.
- **A cascade pipeline** (denoise → translate → noisify) produces fully translated informal output that no single-model system achieves (+3.0 BLEU, +4.6 chrF over baseline).
- **Remaining gap**: neither system reliably produces target-side abbreviations, and source-side pragmatic markers (interjections) are erased during denoising and unrecoverable downstream.

---

## Released Data

We release two datasets of synthetically generated monolingual clean↔noisy sentence pairs, one per language. These are intended to be combined with the existing [MultiLexNorm](https://arxiv.org/abs/2202.12550) dataset (not redistributed here).

| File | Language | Pairs | Description |
|------|----------|-------|-------------|
| [`data/multilexnorm/en_synthetic_noisy_10k.jsonl`](data/multilexnorm/en_synthetic_noisy_10k.jsonl) | English | 9,825 | Synthetic noisy↔clean pairs generated from OpenSubtitles via rule-based transforms |
| [`data/multilexnorm/es_synthetic_noisy_10k.jsonl`](data/multilexnorm/es_synthetic_noisy_10k.jsonl) | Spanish | 9,373 | Synthetic noisy↔clean pairs generated from OpenSubtitles via rule-based transforms |

Each file is in JSONL format, one JSON object per line:

```json
{"noisy": "xq no vienes mñna?? jajaja", "clean": "¿por qué no vienes mañana?"}
```

We also release the small **handwritten evaluation sets** used in the paper — sentence pairs authored manually to cover a variety of informal phenomena:

| File | Language | Pairs |
|------|----------|-------|
| [`data/handwritten/en_pairs.jsonl`](data/handwritten/en_pairs.jsonl) | English | 45 |
| [`data/handwritten/es_pairs.jsonl`](data/handwritten/es_pairs.jsonl) | Spanish | 44 |

These evaluation pairs were **never used in training**.

### Noise types covered

The synthetic data applies stochastic rule-based transforms targeting the following informal phenomena:

| Phenomenon | EN | ES |
|---|:---:|:---:|
| Accent / diacritic dropping | — | ✓ |
| Vowel elongation (`cooool`) | ✓ | ✓ |
| Lowercasing | ✓ | ✓ |
| Repeated punctuation (`!!!`, `???`) | ✓ | ✓ |
| Consonant dropping (`-g` in *-ing*) | ✓ | — |
| Intervocalic *d* dropping (*habla**d**o* → *hablao*) | — | ✓ |
| Word-level abbreviations (*xq*, *q*, *x*, *tb*) | — | ✓ |
| Slang / interjection injection | ✓ | ✓ |

---

## Repository Structure

```
.
├── paper.tex                          # Paper source (ACL format)
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── handwritten/                   # Human-authored evaluation pairs
│   │   ├── en_pairs.jsonl
│   │   └── es_pairs.jsonl
│   └── multilexnorm/                  # Training data
│       ├── en_synthetic_noisy_10k.jsonl   # Released: synthetic EN pairs
│       ├── es_synthetic_noisy_10k.jsonl   # Released: synthetic ES pairs
│       ├── en_train.jsonl             # Derived from MultiLexNorm (not redistributed)
│       ├── en_test.jsonl
│       ├── en_train_augmented.jsonl
│       ├── en_test_augmented.jsonl
│       ├── es_train.jsonl
│       ├── es_test.jsonl
│       ├── es_train_augmented.jsonl
│       ├── es_test_augmented.jsonl
│       └── es_combined_train.jsonl    # Real ES + synthetic ES combined
│
├── src/
│   ├── data_prep/
│   │   ├── parse_multilexnorm.py      # Convert MultiLexNorm .norm files → JSONL
│   │   ├── augment.py                 # Add elongation & punctuation augmentation
│   │   ├── generate_noisy_data_es.py  # Generate synthetic ES noisy↔clean pairs
│   │   └── generate_noisy_data_en.py  # Generate synthetic EN noisy↔clean pairs
│   │
│   ├── training/
│   │   ├── finetune.py                # Main fine-tuning script (all architecture modes)
│   │   ├── train_decoder.py           # Decoder fine-tuning with shifted encoder states
│   │   ├── dataset.py                 # PyTorch Dataset for clean↔noisy pairs
│   │   └── config.yaml                # Hyperparameter configuration
│   │
│   ├── embedding_analysis/
│   │   ├── compute_direction.py       # Compute mean noise direction vectors
│   │   └── compare_models.py          # Compare directions across XLM-R, NLLB, CANINE
│   │
│   ├── evaluation/
│   │   ├── evaluate.py                # Full evaluation (WA, ERR, chrF, BLEU, COMET)
│   │   ├── mt_test.py                 # Machine translation evaluation
│   │   └── noise_type_analysis.py     # Per-noise-type breakdown
│   │
│   └── utils/
│       └── direction.py               # Load/save/apply noise direction vectors
│
├── notebooks/
│   ├── 01_feasibility.ipynb           # Initial feasibility exploration
│   └── 02_noise_direction_analysis.ipynb  # Noise direction analysis
│
└── results/
    └── embedding_comparison.csv       # Output of compare_models.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

The main model used is [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M), downloaded automatically by HuggingFace Transformers on first use.

---

## Reproducing the Experiments

### 1. Obtain MultiLexNorm data

Download the [MultiLexNorm dataset](https://github.com/text-normalization/multilexnorm2021) and place the `.norm` files under `data/multilexnorm/en/` and `data/multilexnorm/es/`. Then parse them into JSONL:

```bash
python src/data_prep/parse_multilexnorm.py
python src/data_prep/augment.py
```

This produces `en_train.jsonl`, `en_test.jsonl`, `es_train.jsonl`, `es_test.jsonl` and their augmented variants.

### 2. Generate synthetic noisy data

The released synthetic data was generated from [OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles/) sentences using rule-based transforms:

```bash
# Spanish (produces es_synthetic_noisy_10k.jsonl)
python src/data_prep/generate_noisy_data_es.py

# English (produces en_synthetic_noisy_10k.jsonl)
python src/data_prep/generate_noisy_data_en.py
```

If you just want to use the synthetic data without regenerating it, the released files are ready to use directly.

### 3. Fine-tune

The main training script supports three architecture modes via command-line flags:

```bash
# Decoder-only (frozen encoder, LoRA on decoder)
python src/training/finetune.py --mode decoder_only

# Encoder-only (LoRA on encoder, frozen decoder)
python src/training/finetune.py --mode encoder_only

# Combined (LoRA on both)
python src/training/finetune.py --mode combined
```

Hyperparameters (LoRA rank, learning rate, epochs, etc.) are in [`src/training/config.yaml`](src/training/config.yaml).

The script registers two new language tokens with NLLB-200: `eng_noisy_Latn` and `spa_noisy_Latn`.

### 4. Embedding analysis

To reproduce the geometric analysis of the clean↔noisy direction:

```bash
python src/embedding_analysis/compare_models.py
# Output: results/embedding_comparison.csv
```

### 5. Evaluate

```bash
python src/evaluation/evaluate.py --model_dir <path_to_checkpoint> --test_file data/handwritten/es_pairs.jsonl
```

---

## Citation

If you use the data or code from this work, please cite:

```bibtex
@misc{azpiri2025informal,
  title={Bypassing the Parallel Data Constraint: Adapting {NLLB}-200 for Informal {EN}--{ES} Translation with Monolingual Data Alone},
  author={Azpiri Iriarte, Asier and Barajas Pascual, Sof{\'i}a and Senderowicz Guerra, Vera},
  year={2025},
  institution={UPV/EHU}
}
```

---

## License

The code in this repository is released under the [MIT License](LICENSE).

The released synthetic datasets (`en_synthetic_noisy_10k.jsonl`, `es_synthetic_noisy_10k.jsonl`) were generated from [OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles/) source sentences. Please refer to the OpenSubtitles terms of use.

The handwritten evaluation pairs are original work and released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
