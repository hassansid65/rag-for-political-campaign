"""Micro-benchmark: reranker cost vs. model / candidate count / sequence length."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("LOG_LEVEL", "ERROR")

import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

QUERY = "How much do I get under Amma Vodi and am I eligible with three children?"
PASSAGE = (
    "Amma Vodi will continue at Rs. 15,000 per year per mother, and we will remove "
    "the current one-child restriction. From the 2025-26 academic year the benefit "
    "applies to every child enrolled from Class 1 to Class 12, capped at three "
    "children per household. Eligibility requires 75 percent attendance and a "
    "household income below Rs. 2.5 lakh per annum. Every government school will "
    "have functioning toilets, drinking water, a boundary wall and a digital "
    "classroom by June 2026. In Visakhapatnam, NTR and Guntur districts, 340 "
    "schools will be upgraded to English-medium CBSE-affiliated model schools. "
) * 3


def bench(model_name: str, max_length: int, n_pairs: int, passage_chars: int) -> float:
    model = CrossEncoder(model_name, max_length=max_length, device="cpu")
    pairs = [(QUERY, PASSAGE[:passage_chars]) for _ in range(n_pairs)]
    model.predict(pairs[:2], show_progress_bar=False)          # warm
    best = min(
        (
            _time_once(model, pairs)
            for _ in range(3)
        )
    )
    return best


def _time_once(model, pairs) -> float:
    start = time.perf_counter()
    model.predict(pairs, batch_size=min(16, len(pairs)), show_progress_bar=False)
    return (time.perf_counter() - start) * 1000


def main() -> None:
    cores = os.cpu_count() or 4
    print(f"CPU cores: {cores}  ·  torch threads: {torch.get_num_threads()}")
    print()

    configs = [
        # model,                              max_len, pairs, passage_chars
        ("BAAI/bge-reranker-base",            512, 20, 2048),   # current default
        ("BAAI/bge-reranker-base",            256, 20,  700),
        ("BAAI/bge-reranker-base",            256, 12,  700),
        ("BAAI/bge-reranker-base",            192, 12,  600),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", 256, 20,  700),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", 256, 12,  700),
    ]

    print(f"{'model':<42} {'max_len':>7} {'pairs':>6} {'chars':>6} {'ms':>9} {'ms/pair':>8}")
    print("-" * 84)
    for model_name, max_length, pairs, chars in configs:
        try:
            elapsed = bench(model_name, max_length, pairs, chars)
        except Exception as exc:  # noqa: BLE001
            print(f"{model_name:<42} FAILED: {exc}")
            continue
        print(
            f"{model_name:<42} {max_length:>7} {pairs:>6} {chars:>6} "
            f"{elapsed:>9.1f} {elapsed / pairs:>8.1f}"
        )

    print()
    print("Now with torch threads pinned to physical cores:")
    torch.set_num_threads(max(1, cores // 2))
    print(f"  torch threads = {torch.get_num_threads()}")
    for model_name, max_length, pairs, chars in configs[2:4]:
        elapsed = bench(model_name, max_length, pairs, chars)
        print(
            f"  {model_name:<40} {max_length:>4} {pairs:>3}p {chars:>5}c "
            f"→ {elapsed:>8.1f}ms ({elapsed / pairs:.1f}/pair)"
        )


if __name__ == "__main__":
    main()
