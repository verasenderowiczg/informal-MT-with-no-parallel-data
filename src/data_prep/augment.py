"""
Task 1.2: Data augmentation for informality features missing from MultiLexNorm.

MultiLexNorm covers abbreviations and slang but misses:
  - Vowel elongation:       "incredible" → "incredibleee"
  - Punctuation repetition: "?" → "??", "!" → "!!!", or removal

Applied to the noisy side only; clean side is unchanged.
Rates: ~15-20% of sentences per transformation (independently sampled).
"""

import argparse
import json
import random
import re
from pathlib import Path


ELONGATION_RATE = 0.175
PUNCT_RATE = 0.175
ELONGATION_MIN = 2
ELONGATION_MAX = 4


def elongate_word(word: str, rng: random.Random) -> str:
    """Repeat the final vowel of a word 2-4 extra times."""
    vowels = set("aeiouáéíóúàèìòùAEIOUÁÉÍÓÚÀÈÌÒÙ")
    for i in range(len(word) - 1, -1, -1):
        if word[i] in vowels:
            repeat = rng.randint(ELONGATION_MIN, ELONGATION_MAX)
            return word[:i+1] + word[i] * repeat + word[i+1:]
    return word  # no vowel found (e.g., acronym)


def augment_elongation(sentence: str, rng: random.Random) -> str:
    """Randomly elongate the final vowel of 1-2 words in the sentence."""
    words = sentence.split()
    if not words:
        return sentence

    n_words = rng.randint(1, min(2, len(words)))
    candidates = list(range(len(words)))
    chosen = rng.sample(candidates, n_words)

    for idx in chosen:
        words[idx] = elongate_word(words[idx], rng)

    return " ".join(words)


def augment_punctuation(sentence: str, rng: random.Random) -> str:
    """
    Randomly modify sentence-final punctuation:
      - "?" → "??" or "???" (60%) or remove (20%)
      - "!" → "!!" or "!!!" (60%) or remove (20%)
      - Otherwise: sometimes append "??" or "!!" (20%)
    """
    sentence = sentence.rstrip()
    if not sentence:
        return sentence

    last = sentence[-1]
    roll = rng.random()

    if last == "?":
        if roll < 0.6:
            return sentence + "?" * rng.randint(1, 2)
        elif roll < 0.8:
            return sentence[:-1]
        return sentence
    elif last == "!":
        if roll < 0.6:
            return sentence + "!" * rng.randint(1, 2)
        elif roll < 0.8:
            return sentence[:-1]
        return sentence
    else:
        if roll < 0.2:
            return sentence + rng.choice(["??", "!!", "!?"])
        return sentence


def augment_pair(pair: dict, rng: random.Random) -> dict:
    noisy = pair["noisy"]

    if rng.random() < ELONGATION_RATE:
        noisy = augment_elongation(noisy, rng)

    if rng.random() < PUNCT_RATE:
        noisy = augment_punctuation(noisy, rng)

    return {"noisy": noisy, "clean": pair["clean"]}


def augment_file(in_path: Path, out_path: Path, seed: int) -> None:
    rng = random.Random(seed)

    with open(in_path, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]

    augmented = [augment_pair(p, rng) for p in pairs]

    changed = sum(1 for orig, aug in zip(pairs, augmented) if orig["noisy"] != aug["noisy"])

    with open(out_path, "w", encoding="utf-8") as f:
        for pair in augmented:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"  {in_path.name} → {out_path.name}  "
          f"({changed}/{len(pairs)} sentences modified, {100*changed/len(pairs):.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/multilexnorm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("Augmenting noisy sides of MultiLexNorm splits...")

    for name in ("en_train", "en_test", "es_train", "es_test"):
        in_path = data_dir / f"{name}.jsonl"
        out_path = data_dir / f"{name}_augmented.jsonl"
        if not in_path.exists():
            print(f"  WARNING: {in_path} not found, skipping.")
            continue
        augment_file(in_path, out_path, seed=args.seed)


if __name__ == "__main__":
    main()
