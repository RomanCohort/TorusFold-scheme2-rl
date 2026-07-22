"""tokenizer.py — circRNA sequence → Scheme10 token tensor.

Token map (matches :class:`Scheme10Model` + scheme8, see
``scheme10_circ_equivariant_gnn.py`` forward docstring / line ~628):

    A → 0   U → 1   G → 2   C → 3   N(pad) → 4

DNA ``T`` is silently folded to ``U``; lowercase is upper-cased. Unknown
letters (``R``, ``Y``, ``W`` …) are mapped to the pad token ``4`` rather than
rejected — the server's API layer rejects non-ACGU(N) up front, so this is a
defensive second line. Sequences shorter than ``min_len`` are right-padded
with ``N`` (token 4); Scheme10's ``forward`` derives valid lengths from the
pad token, so padding does not corrupt the prediction.
"""

from __future__ import annotations

import torch

NUC_TO_ID = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4}
PAD_ID = 4
DNA_T_TO_U = str.maketrans({"T": "U", "t": "u"})


def clean_sequence(sequence: str) -> str:
    """Normalise a raw sequence: uppercase, T→U, collapse whitespace.

    Does NOT reject unknown letters — :func:`tokenize` maps them to pad.
    The API layer is the single source of truth for rejection.
    """
    seq = sequence.translate(DNA_T_TO_U)
    seq = "".join(seq.split()).upper()
    return seq


def tokenize(
    sequence: str,
    *,
    pad_to: int | None = None,
    min_len: int = 8,
) -> torch.Tensor:
    """Tokenise a circRNA sequence into a ``(1, L)`` long tensor.

    Args:
        sequence: raw ACGU(T) string; ``T``→``U``, whitespace stripped.
        pad_to: if given, right-pad with ``N`` up to this length (no truncation).
            Scheme10 expects a single sequence per batch, so batch dim is 1.
        min_len: minimum length; shorter sequences are padded to this (Scheme10's
            ring-angle / edge-category helpers need at least a handful of residues).

    Returns:
        LongTensor of shape ``(1, L)`` with values in ``0..4``.
    """
    seq = clean_sequence(sequence)
    if len(seq) == 0:
        raise ValueError("empty sequence after cleaning")

    ids = [NUC_TO_ID.get(ch, PAD_ID) for ch in seq]

    target_len = len(ids)
    if pad_to is not None:
        target_len = max(target_len, pad_to)
    target_len = max(target_len, min_len)

    if target_len > len(ids):
        ids.extend([PAD_ID] * (target_len - len(ids)))

    return torch.tensor([ids], dtype=torch.long)
