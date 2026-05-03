"""
Parse MultiLexNorm .norm files into sentence-level clean/noisy JSONL pairs.

Format: tab-separated, one token per line, blank lines = sentence boundary.
  col1 = noisy token, col2 = clean normalization.

Edge cases:
  - Split: col2 contains a space → one noisy token maps to multiple clean tokens
    e.g.  "wanna\twant to"
  - Merge: col2 is empty → this noisy token was absorbed into the previous token's norm
    e.g.  "screen\tscreenshot"  +  "shot\t"   →  clean="screenshot", noisy="screen shot"

Output: {"noisy": "...", "clean": "..."} one JSON object per line.
"""

import argparse
import json
from pathlib import Path


def parse_norm_file(path: Path) -> list[dict]:
    pairs = []
    noisy_tokens: list[str] = []
    clean_tokens: list[str] = []

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line == "":
                if noisy_tokens:
                    pairs.append({
                        "noisy": " ".join(noisy_tokens),
                        "clean": " ".join(clean_tokens),
                    })
                noisy_tokens = []
                clean_tokens = []
                continue

            parts = line.split("\t")
            noisy_tok = parts[0]
            clean_tok = parts[1] if len(parts) > 1 else ""

            noisy_tokens.append(noisy_tok)

            if clean_tok == "":
                # Merge continuation: absorbed into the previous clean token — skip.
                pass
            else:
                # Handles both normal 1:1 and splits (clean_tok may contain spaces).
                clean_tokens.extend(clean_tok.split())

    # Flush final sentence if file doesn't end with a blank line.
    if noisy_tokens:
        pairs.append({
            "noisy": " ".join(noisy_tokens),
            "clean": " ".join(clean_tokens),
        })

    return pairs


def print_stats(pairs: list[dict], label: str) -> None:
    n = len(pairs)
    noisy_lens = [len(p["noisy"].split()) for p in pairs]
    clean_lens = [len(p["clean"].split()) for p in pairs]

    differ = sum(1 for p in pairs if p["noisy"].lower() != p["clean"].lower())
    token_total = sum(noisy_lens)
    token_differ = sum(
        sum(1 for nt, ct in zip(p["noisy"].split(), p["clean"].split()) if nt.lower() != ct.lower())
        for p in pairs
    )

    print(f"\n{label}")
    print(f"  Sentence pairs:          {n}")
    print(f"  Avg noisy length (tok):  {sum(noisy_lens)/n:.1f}")
    print(f"  Avg clean length (tok):  {sum(clean_lens)/n:.1f}")
    print(f"  Sentences with any diff: {differ} ({100*differ/n:.1f}%)")
    print(f"  Token-level diff rate:   {100*token_differ/token_total:.1f}%  ({token_differ}/{token_total})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/multilexnorm/raw/data",
                        help="Path to MultiLexNorm data/ directory")
    parser.add_argument("--out_dir", default="data/multilexnorm",
                        help="Output directory for JSONL files")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "en_train": raw_dir / "en" / "train.norm",
        "en_test":  raw_dir / "en" / "test.norm",
        "es_train": raw_dir / "es" / "train.norm",
        "es_test":  raw_dir / "es" / "test.norm",
    }

    for name, path in splits.items():
        if not path.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue

        pairs = parse_norm_file(path)
        out_path = out_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        print_stats(pairs, f"{name} → {out_path}")


if __name__ == "__main__":
    main()
