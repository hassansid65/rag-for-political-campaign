"""Verify the cascade actually cascades, and where its time goes."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("LOG_LEVEL", "INFO")

from core.config import settings  # noqa: E402
from core.latency import METRICS  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402

setup_logging("INFO")

from retrieval.reranker import get_reranker  # noqa: E402

QUERY = "How much do I get under Amma Vodi?"
PASSAGES = [
    "Amma Vodi will continue at Rs. 15,000 per year per mother, and we will remove "
    "the current one-child restriction. Eligibility requires 75 percent attendance.",
    "Rythu Bharosa Plus pays Rs. 18,000 per farmer family per year in three "
    "instalments timed to Kharif sowing, Rabi sowing and harvest.",
    "The Dr. YSR Aarogyasri coverage limit rises from Rs. 5 lakh to Rs. 10 lakh per "
    "family per year, across 4,100 procedures at empanelled hospitals.",
] * 6   # 18 candidates, mirroring a real fused candidate set


def main() -> None:
    print(f"config: mode={settings.rerank_mode} cap={settings.rerank_candidate_cap} "
          f"keep={settings.rerank_cascade_keep} max_len={settings.reranker_max_length}")
    print(f"        precise={settings.reranker_model}")
    print(f"        fast   ={settings.reranker_fast_model}")
    print()

    reranker = get_reranker()
    t0 = time.perf_counter()
    reranker.load()
    print(f"load(): {time.perf_counter() - t0:.2f}s")
    print(f"health: {reranker.health()}")
    print()

    # Real chunks are ~700 chars, not ~150. Sequence length dominates
    # cross-encoder cost, so benchmark at the size we actually index.
    filler = (
        " Every government school will have functioning toilets, drinking water, a "
        "boundary wall and a digital classroom by June 2026. In Visakhapatnam, NTR "
        "and Guntur districts, 340 schools will be upgraded to English-medium "
        "CBSE-affiliated model schools. Full fee reimbursement applies below "
        "Rs. 2.5 lakh per annum, paid directly to the institution within 60 days."
    )
    realistic = [(p + filler)[:700] for p in PASSAGES]
    print(f"passage length: {len(realistic[0])} chars\n")

    candidates = realistic[: settings.rerank_candidate_cap]
    for mode in ("fast", "cascade", "single"):
        # Two runs: first may still pay lazy init for that mode's tiers.
        for run in range(2):
            t0 = time.perf_counter()
            scores = reranker.score(QUERY, candidates, mode=mode)
            elapsed = (time.perf_counter() - t0) * 1000
            if run == 1:
                top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
                print(
                    f"{mode:<8} {len(candidates)} pairs → {elapsed:7.1f}ms "
                    f"({elapsed / len(candidates):5.1f}/pair)  "
                    f"top3_scores={[round(scores[i], 3) for i in top]}"
                )

    print()
    print("per-tier metrics:")
    for key, stats in sorted(METRICS.snapshot().items()):
        if "rerank" in key:
            print(f"  {key:<52} p50={stats['p50']:8.1f}ms count={stats['count']}")
    print(f"counters: {METRICS.counters()}")


if __name__ == "__main__":
    main()
