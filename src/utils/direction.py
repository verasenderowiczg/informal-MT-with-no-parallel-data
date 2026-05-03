"""
Shared utilities for loading and applying noise directions.

A "noise direction" is a mean-pooled hidden-state difference vector
computed from clean/noisy sentence pairs via NLLB's encoder.
Shape: [1, hidden_dim]  (broadcast-ready for [batch, seq_len, hidden_dim])
"""

from pathlib import Path
import torch


def load_direction(path: str | Path) -> torch.Tensor:
    """Load a saved noise direction tensor. Returns shape [1, hidden_dim]."""
    d = torch.load(path, map_location="cpu")
    if d.dim() == 1:
        d = d.unsqueeze(0)
    return d


def save_direction(direction: torch.Tensor, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(direction.cpu(), path)


def apply_direction(
    hidden_states: torch.Tensor,
    direction: torch.Tensor,
    scale: float | torch.Tensor,
) -> torch.Tensor:
    """
    Shift encoder hidden states by scale * direction.

    hidden_states: [batch, seq_len, hidden_dim]
    direction:     [1, hidden_dim]  or  [hidden_dim]
    scale:         scalar float or 0-dim tensor

    Returns shifted tensor, same shape as hidden_states.
    """
    d = direction.to(hidden_states.device)
    if d.dim() == 1:
        d = d.unsqueeze(0)
    # Broadcast across batch and seq_len dimensions.
    return hidden_states + scale * d.unsqueeze(0)


@torch.no_grad()
def compute_mean_direction(
    pairs: list[tuple[str, str]],
    tokenizer,
    encoder,
    device: torch.device,
    src_lang: str,
    max_length: int = 128,
) -> torch.Tensor:
    """
    Compute mean-pooled noise direction from a list of (clean, noisy) string pairs.

    Returns direction tensor of shape [1, hidden_dim].
    """
    tokenizer.src_lang = src_lang
    diffs = []

    for clean, noisy in pairs:
        def encode(text):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)
            hidden = encoder(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        pooled_clean = encode(clean)
        pooled_noisy = encode(noisy)
        diffs.append((pooled_noisy - pooled_clean).cpu())

    return torch.stack(diffs).mean(dim=0)  # [1, hidden_dim]
