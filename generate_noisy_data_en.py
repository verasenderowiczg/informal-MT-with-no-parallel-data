"""
Generate synthetic noisy English training data for eng_noisy_Latn fine-tuning.

Mirrors generate_noisy_data.py but adapted for English noise patterns:
  - No accent dropping (English doesn't use accents)
  - No d-dropping (Spanish-specific)
  - Abbreviations from EN MultiLexNorm (u, ur, bc, pls, dat, etc.)
  - Vowel elongation (sooo, yesss, nooo)
  - Punctuation repetition
  - g-dropping in -ing (running → runnin)
  - English interjections (lol, lmao, omg, bruh, ngl, fr)

Usage:
    python generate_noisy_data_en.py \
        --opensubs_path en_sample.txt \
        --multilexnorm_train data/multilexnorm/en/train.norm \
        --multilexnorm_test data/multilexnorm/en/test.norm \
        --output_path data/multilexnorm/en_synthetic_noisy_5k.jsonl \
        --n_samples 2640

    cat data/multilexnorm/en_train_augmented.jsonl data/multilexnorm/en_synthetic_noisy_5k.jsonl > data/multilexnorm/en_combined_train_5k.jsonl
"""

import argparse
import json
import random
import re
from collections import defaultdict


# ============================================================
# 1. Build abbreviation dictionary from MultiLexNorm
# ============================================================

def load_multilexnorm_mappings(*norm_paths):
    """
    Parse MultiLexNorm .norm files.
    Format: noisy_word\tclean_word per line, blank lines or *\t* separate sentences.
    Returns: dict of clean_word_lower -> [list of noisy variants]
    """
    mappings = defaultdict(set)

    for path in norm_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line == "*\t*":
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                noisy, clean = parts[0].strip(), parts[1].strip()
                if noisy.lower() == clean.lower():
                    continue
                if not any(c.isalpha() for c in clean):
                    continue
                mappings[clean.lower()].add(noisy)

    mappings = {k: list(v) for k, v in mappings.items()}

    print(f"  Loaded {len(mappings)} unique clean->noisy word mappings")
    examples = list(mappings.items())[:10]
    for clean, noisy_list in examples:
        print(f"    {clean} -> {noisy_list[:3]}")

    return mappings


# ============================================================
# 2. Load source sentences
# ============================================================

def load_source_sentences(path, mappings=None, n_candidates=20000,
                          min_words=4, max_words=20):
    """
    Load English sentences, splitting into abbreviation-rich and regular pools.
    When mappings is provided, sentences containing abbreviatable words are
    tracked separately so generate_dataset() can oversample them.
    """
    abbrev_rich = []
    regular = []
    mapping_keys = set(mappings.keys()) if mappings else set()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("<") or line.startswith("(") or line.startswith("["):
                continue
            if "&" in line and ";" in line:
                continue
            if line == line.upper() and len(line) > 3:
                continue
            if re.match(r"^\d+$", line) or "-->" in line:
                continue

            words = line.split()
            if len(words) < min_words or len(words) > max_words:
                continue

            if mapping_keys:
                word_keys = {w.lower().rstrip(".,;:!?") for w in words}
                n_abbrev = len(word_keys & mapping_keys)
                if n_abbrev >= 1:
                    abbrev_rich.append(line)
                else:
                    regular.append(line)
            else:
                regular.append(line)

            if len(abbrev_rich) + len(regular) >= n_candidates:
                break

    print(f"  Loaded {len(abbrev_rich)} abbreviation-rich + "
          f"{len(regular)} regular = {len(abbrev_rich) + len(regular)} total")
    return abbrev_rich, regular


# ============================================================
# 3. Noisification transforms (English-specific)
# ============================================================

def elongate_vowels(word):
    """Elongate the last vowel in a word: 'please' -> 'pleaase'."""
    vowel_positions = [i for i, c in enumerate(word.lower()) if c in "aeiou"]
    if not vowel_positions:
        return word
    pos = vowel_positions[-1]
    repeat = random.randint(2, 3)
    return word[:pos] + word[pos] * repeat + word[pos + 1:]


def repeat_punctuation(text, prob=0.3):
    """Repeat sentence-final punctuation: '?' -> '???', '!' -> '!!!'"""
    if random.random() > prob:
        return text

    match = re.search(r"([?!.]+)\s*$", text)
    if not match:
        return text

    punct = match.group(1)
    if "?" in punct or "!" in punct:
        base_char = "?" if "?" in punct else "!"
        repeat = random.randint(2, 5)
        new_punct = base_char * repeat
        return text[:match.start()] + new_punct
    return text


def drop_g_in_ing(word, prob=0.4):
    """running -> runnin, going -> goin"""
    if random.random() > prob:
        return word
    if word.lower().endswith("ing") and len(word) > 4:
        return word[:-1]
    if word.lower().endswith("ing,") or word.lower().endswith("ing."):
        return word[:-2] + word[-1]
    return word


def substitute_abbreviation(word, mappings):
    """Replace a word with its noisy abbreviation if available."""
    key = word.lower().rstrip(".,;:!?")
    if key in mappings:
        noisy = random.choice(mappings[key])
        trailing = word[len(key):] if len(word) > len(key) else ""
        return noisy + trailing
    return word


def noisify_sentence(clean_text, mappings, intensity="medium"):
    """Apply stochastic English noisification transforms."""
    if intensity == "light":
        abbrev_prob = 0.2
        punct_prob = 0.15
        g_drop_prob = 0.2
        lowercase_prob = 0.6
    elif intensity == "heavy":
        abbrev_prob = 0.8
        punct_prob = 0.5
        g_drop_prob = 0.6
        lowercase_prob = 0.95
    else:  # medium
        abbrev_prob = 0.4
        punct_prob = 0.3
        g_drop_prob = 0.4
        lowercase_prob = 0.8

    words = clean_text.split()
    result = []

    for word in words:
        w = word

        if random.random() < abbrev_prob:
            w = substitute_abbreviation(w, mappings)

        w = drop_g_in_ing(w, prob=g_drop_prob)

        result.append(w)

    # Vowel elongation -- at most 1 word, 20% of sentences, natural positions
    if random.random() < 0.2:
        eligible = [
            i for i, w in enumerate(result)
            if len(w) > 3
            and any(c in "aeiou" for c in w.lower())
            and (
                i == len(result) - 1
                or result[i].endswith(",")
                or result[i].endswith(".")
                or (i + 1 < len(result) and result[i + 1].startswith(","))
            )
        ]
        if eligible:
            idx = random.choice(eligible)
            result[idx] = elongate_vowels(result[idx])

    text = " ".join(result)

    text = repeat_punctuation(text, prob=punct_prob)

    if random.random() < lowercase_prob:
        text = text.lower()

    # Drop trailing period
    if random.random() < 0.8:
        text = re.sub(r"\.\s*$", "", text)

    # Random interjection
    if random.random() < 0.15:
        interjections = ["lol", "lmao", "omg", "bruh", "ngl", "fr", "smh", "tbh"]
        text = text.rstrip() + " " + random.choice(interjections)

    # Clean up trailing comma
    text = re.sub(r",\s*$", "", text)

    return text


# ============================================================
# 4. Generate dataset
# ============================================================

def _edit_ratio(clean, noisy):
    """Fraction of characters that differ between clean and noisy."""
    if not clean:
        return 0.0
    diffs = sum(1 for a, b in zip(clean, noisy) if a != b)
    diffs += abs(len(clean) - len(noisy))
    return diffs / max(len(clean), len(noisy))


def generate_dataset(abbrev_rich, regular, mappings, n_samples=3000,
                     seed=42, abbrev_ratio=0.65):
    """
    Generate n_samples noisy/clean pairs, drawing abbrev_ratio from the
    abbreviation-rich pool and the rest from the regular pool.
    Abbrev-rich sentences use heavier intensity to ensure abbreviations
    actually get applied. Near-no-ops (< 5% edit) are rejected.
    """
    random.seed(seed)

    n_abbrev = int(n_samples * abbrev_ratio)
    n_regular = n_samples - n_abbrev

    selected_abbrev = [(s, True) for s in random.choices(abbrev_rich, k=n_abbrev)] if abbrev_rich else []
    selected_regular = [(s, False) for s in random.choices(regular, k=n_regular)] if regular else []
    selected = selected_abbrev + selected_regular
    random.shuffle(selected)

    pairs = []
    intensities_regular = ["light", "medium", "medium", "heavy"]
    intensities_abbrev = ["medium", "heavy", "heavy", "heavy"]

    for clean, is_abbrev in selected:
        intensity = random.choice(intensities_abbrev if is_abbrev else intensities_regular)
        noisy = noisify_sentence(clean, mappings, intensity=intensity)

        if noisy == clean:
            continue
        if _edit_ratio(clean, noisy) < 0.05:
            continue

        pairs.append({"clean": clean, "noisy": noisy})

    print(f"  Generated {len(pairs)} noisy/clean pairs "
          f"({len(selected_abbrev)} abbrev-rich, {len(selected_regular)} regular)")
    return pairs


# ============================================================
# 5. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic noisy English data")
    parser.add_argument("--opensubs_path", type=str, required=True,
                        help="Path to English plain text corpus")
    parser.add_argument("--multilexnorm_train", type=str, required=True,
                        help="Path to MultiLexNorm EN train.norm")
    parser.add_argument("--multilexnorm_test", type=str, required=True,
                        help="Path to MultiLexNorm EN test.norm")
    parser.add_argument("--output_path", type=str, default="en_synthetic_noisy.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--n_samples", type=int, default=3000,
                        help="Number of synthetic pairs to generate")
    parser.add_argument("--n_candidates", type=int, default=20000,
                        help="Number of source lines to consider")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading MultiLexNorm EN word mappings...")
    mappings = load_multilexnorm_mappings(args.multilexnorm_train, args.multilexnorm_test)

    print(f"\nLoading source sentences from {args.opensubs_path}...")
    abbrev_rich, regular = load_source_sentences(
        args.opensubs_path, mappings=mappings, n_candidates=args.n_candidates)

    print(f"\nGenerating {args.n_samples} synthetic noisy/clean pairs...")
    pairs = generate_dataset(abbrev_rich, regular, mappings,
                             n_samples=args.n_samples, seed=args.seed)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten to {args.output_path}")
    print(f"\nSample outputs:")
    for pair in pairs[:10]:
        print(f"  CLEAN: {pair['clean']}")
        print(f"  NOISY: {pair['noisy']}")
        print()


if __name__ == "__main__":
    main()
