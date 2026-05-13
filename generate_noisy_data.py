"""
Generate synthetic noisy Spanish training data for spa_noisy_Latn fine-tuning.

Sources:
  - OpenSubtitles Spanish monolingual text (dialogue, informal register)
  - MultiLexNorm ES train.norm + test.norm (word-level noisy↔clean mappings)

Transforms applied stochastically:
  1. Abbreviation substitution (from MultiLexNorm mappings)
  2. Accent dropping (á→a, é→e, etc.)
  3. Vowel elongation (locaaa, siiii)
  4. Punctuation repetition (? → ???, ! → !!!)
  5. Lowercase everything
  6. d-dropping in -ado/-ido (cansado → cansao)

Usage:
    # Step 1: Download OpenSubtitles Spanish (one-time, ~300MB compressed)
    wget -O es.txt.gz "https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/es.txt.gz"
    gunzip es.txt.gz

    # Step 2: Run this script
    python generate_noisy_data.py \
        --opensubs_path es.txt \
        --multilexnorm_train data/multilexnorm/es/train.norm \
        --multilexnorm_test data/multilexnorm/es/test.norm \
        --output_path data/multilexnorm/es_synthetic_noisy.jsonl \
        --n_samples 3000

    # Step 3: Combine with real data for training
    cat data/multilexnorm/es_train_augmented.jsonl data/multilexnorm/es_synthetic_noisy.jsonl > data/multilexnorm/es_combined_train.jsonl
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
    Format: noisy_word\\tclean_word per line, blank lines or *\\t* separate sentences.
    Returns: dict of clean_word_lower → [list of noisy variants]
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
                # Skip identical pairs
                if noisy.lower() == clean.lower():
                    continue
                # Skip pure punctuation
                if not any(c.isalpha() for c in clean):
                    continue
                mappings[clean.lower()].add(noisy)

    # Convert sets to lists for random.choice
    mappings = {k: list(v) for k, v in mappings.items()}

    print(f"  Loaded {len(mappings)} unique clean→noisy word mappings")
    # Show some examples
    examples = list(mappings.items())[:10]
    for clean, noisy_list in examples:
        print(f"    {clean} → {noisy_list[:3]}")

    return mappings


# ============================================================
# 2. Load OpenSubtitles sentences
# ============================================================

def load_opensubs_sentences(path, n_candidates=20000, min_words=4, max_words=20):
    """
    Load Spanish subtitle lines, filtering for good candidates:
    - Not too short (>= min_words) — avoid fragments like "Sí." or "¡Vamos!"
    - Not too long (<= max_words) — keep tweet-length
    - No XML/HTML tags
    - No lines that are just character names or sound descriptions
    """
    candidates = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Skip XML-like content
            if line.startswith("<") or line.startswith("(") or line.startswith("["):
                continue

            # Skip lines with HTML entities
            if "&" in line and ";" in line:
                continue

            # Skip lines that are ALL CAPS (usually character names or descriptions)
            if line == line.upper() and len(line) > 3:
                continue

            # Skip lines with timestamps or subtitle numbering
            if re.match(r"^\d+$", line) or "-->" in line:
                continue

            words = line.split()
            if len(words) < min_words or len(words) > max_words:
                continue

            # Must contain at least some Spanish characters or common words
            candidates.append(line)

            if len(candidates) >= n_candidates:
                break

    print(f"  Loaded {len(candidates)} candidate sentences from OpenSubtitles")
    return candidates


# ============================================================
# 3. Noisification transforms
# ============================================================

ACCENT_MAP = {
    "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
    "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
    "ü": "u", "Ü": "U",
}

def drop_accents(text, prob=0.8):
    """Drop accents from vowels with given probability."""
    result = []
    for char in text:
        if char in ACCENT_MAP and random.random() < prob:
            result.append(ACCENT_MAP[char])
        else:
            result.append(char)
    return "".join(result)


def elongate_vowels(word):
    """Elongate the last vowel in a word: 'loco' → 'locoooo'."""
    vowel_positions = [i for i, c in enumerate(word.lower()) if c in "aeiou"]
    if not vowel_positions:
        return word
    pos = vowel_positions[-1]  # last vowel, more natural
    repeat = random.randint(3, 5)
    return word[:pos] + word[pos] * repeat + word[pos + 1:]


def repeat_punctuation(text, prob=0.3):
    """Repeat sentence-final punctuation: '?' → '???', '!' → '!!!'"""
    if random.random() > prob:
        return text

    # Find trailing punctuation
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


def drop_d_in_ado(word, prob=0.4):
    """cansado → cansao, comido → comío"""
    if random.random() > prob:
        return word
    # -ado → -ao
    if word.lower().endswith("ado"):
        return word[:-3] + word[-3].replace("a", "a") + "o"  # keep case
    if word.lower().endswith("ado,") or word.lower().endswith("ado."):
        return word[:-4] + "ao" + word[-1]
    # -ido → -ío (less common but exists)
    if word.lower().endswith("ido"):
        return word[:-3] + "ío"
    return word


def substitute_abbreviation(word, mappings):
    """Replace a word with its noisy abbreviation if available."""
    key = word.lower().rstrip(".,;:!?")
    if key in mappings:
        noisy = random.choice(mappings[key])
        # Preserve trailing punctuation
        trailing = word[len(key):] if len(word) > len(key) else ""
        return noisy + trailing
    return word


def noisify_sentence(clean_text, mappings, intensity="medium"):
    """
    Apply stochastic noisification transforms to a clean sentence.
    intensity: "light", "medium", "heavy" — controls how many transforms apply.
    """
    if intensity == "light":
        abbrev_prob = 0.2
        accent_prob = 0.5
        elong_prob = 0.0
        punct_prob = 0.15
        d_drop_prob = 0.2
        lowercase_prob = 0.6
    elif intensity == "heavy":
        abbrev_prob = 0.6
        accent_prob = 0.95
        elong_prob = 0.0
        punct_prob = 0.5
        d_drop_prob = 0.6
        lowercase_prob = 0.95
    else:  # medium
        abbrev_prob = 0.4
        accent_prob = 0.8
        elong_prob = 0.0
        punct_prob = 0.3
        d_drop_prob = 0.4
        lowercase_prob = 0.8

    words = clean_text.split()
    result = []

    for word in words:
        w = word

        # 1. Abbreviation substitution
        if random.random() < abbrev_prob:
            w = substitute_abbreviation(w, mappings)

        # 2. d-dropping in -ado/-ido
        w = drop_d_in_ado(w, prob=d_drop_prob)

        result.append(w)

    # 3. Vowel elongation — at most 1 word per sentence, 20% of sentences
    # Only elongate words at end of sentence or just before a comma/period
    if random.random() < 0.2:
        eligible = [
            i for i, w in enumerate(result)
            if len(w) > 3
            and any(c in "aeiouáéíóú" for c in w.lower())
            and (
                i == len(result) - 1                        # last word
                or result[i].endswith(",")                  # before comma
                or result[i].endswith(".")                  # before period
                or (i + 1 < len(result) and result[i + 1].startswith(","))  # next token is comma
            )
        ]
        if eligible:
            idx = random.choice(eligible)
            result[idx] = elongate_vowels(result[idx])

    text = " ".join(result)

    # 4. Drop accents
    text = drop_accents(text, prob=accent_prob)

    # 5. Punctuation repetition
    text = repeat_punctuation(text, prob=punct_prob)

    # 6. Lowercase
    if random.random() < lowercase_prob:
        text = text.lower()

    # 7. Drop inverted punctuation marks (¿ ¡) — very common in informal Spanish
    if random.random() < 0.7:
        text = text.replace("¿", "").replace("¡", "")

    return text


# ============================================================
# 4. Generate dataset
# ============================================================

def generate_dataset(sentences, mappings, n_samples=3000, seed=42):
    """
    Generate n_samples noisy/clean pairs from the source sentences.
    Uses mixed intensities for variety.
    """
    random.seed(seed)

    # Sample with replacement if needed
    if len(sentences) < n_samples:
        selected = random.choices(sentences, k=n_samples)
    else:
        selected = random.sample(sentences, n_samples)

    pairs = []
    intensities = ["light", "medium", "medium", "heavy"]  # bias toward medium

    for clean in selected:
        intensity = random.choice(intensities)
        noisy = noisify_sentence(clean, mappings, intensity=intensity)

        # Only keep if something actually changed
        if noisy != clean:
            pairs.append({"clean": clean, "noisy": noisy})

    print(f"  Generated {len(pairs)} noisy/clean pairs")
    return pairs


# ============================================================
# 5. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic noisy Spanish data")
    parser.add_argument("--opensubs_path", type=str, required=True,
                        help="Path to OpenSubtitles Spanish plain text (es.txt)")
    parser.add_argument("--multilexnorm_train", type=str, required=True,
                        help="Path to MultiLexNorm ES train.norm")
    parser.add_argument("--multilexnorm_test", type=str, required=True,
                        help="Path to MultiLexNorm ES test.norm (or dev.norm)")
    parser.add_argument("--output_path", type=str, default="es_synthetic_noisy.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--n_samples", type=int, default=3000,
                        help="Number of synthetic pairs to generate")
    parser.add_argument("--n_candidates", type=int, default=20000,
                        help="Number of OpenSubtitles lines to consider")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load abbreviation mappings
    print("Loading MultiLexNorm word mappings...")
    mappings = load_multilexnorm_mappings(args.multilexnorm_train, args.multilexnorm_test)

    # Load source sentences
    print(f"\nLoading OpenSubtitles sentences from {args.opensubs_path}...")
    sentences = load_opensubs_sentences(args.opensubs_path, n_candidates=args.n_candidates)

    # Generate noisy pairs
    print(f"\nGenerating {args.n_samples} synthetic noisy/clean pairs...")
    pairs = generate_dataset(sentences, mappings, n_samples=args.n_samples, seed=args.seed)

    # Write output
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