"""
Why is reranking 4-5x slower under uvicorn than in a standalone benchmark?

Hypothesis: cross-encoder inference runs via `loop.run_in_executor(None, ...)`,
i.e. in a ThreadPoolExecutor worker. OpenMP disables *nested* parallelism by
default, so a torch op launched from a secondary thread runs single-threaded even
though `torch.set_num_threads(6)` was honoured on the main thread.

This script times the same batch three ways to isolate that.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("LOG_LEVEL", "ERROR")

PASSAGE = (
    "Amma Vodi will continue at Rs. 15,000 per year per mother, and we will remove "
    "the current one-child restriction. From the 2025-26 academic year the benefit "
    "applies to every child enrolled from Class 1 to Class 12, capped at three "
    "children per household. Eligibility requires 75 percent attendance and a "
    "household income below Rs. 2.5 lakh per annum. Every government school will "
    "have functioning toilets and a digital classroom by June 2026."
)
QUERY = "How much do I get under Amma Vodi?"


def main() -> None:
    import torch

    from retrieval.reranker import get_reranker

    reranker = get_reranker()
    reranker.load()
    passages = [PASSAGE for _ in range(16)]

    def run() -> float:
        start = time.perf_counter()
        reranker.score(QUERY, passages, mode="cascade")
        return (time.perf_counter() - start) * 1000

    print(f"OMP_NUM_THREADS env   : {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    print(f"torch.get_num_threads : {torch.get_num_threads()}")
    print(f"torch interop threads : {torch.get_num_interop_threads()}")
    print(f"cpu_count             : {os.cpu_count()}")
    print()

    run()  # warm

    main_thread = min(run() for _ in range(3))
    print(f"1. main thread                       : {main_thread:8.1f}ms")

    with ThreadPoolExecutor(max_workers=4) as pool:
        pool_thread = min(pool.submit(run).result() for _ in range(3))
    print(f"2. ThreadPoolExecutor worker         : {pool_thread:8.1f}ms")

    async def via_loop() -> float:
        loop = asyncio.get_running_loop()
        results = []
        for _ in range(3):
            results.append(await loop.run_in_executor(None, run))
        return min(results)

    loop_thread = asyncio.run(via_loop())
    print(f"3. asyncio default executor          : {loop_thread:8.1f}ms")

    print()
    ratio = pool_thread / main_thread if main_thread else 0
    print(f"executor / main ratio: {ratio:.2f}x")
    if ratio > 1.8:
        print(
            "CONFIRMED: torch loses intra-op parallelism in worker threads.\n"
            "Fix: set OMP_NUM_THREADS before torch is imported."
        )
    else:
        print("Not reproduced — the slowdown is elsewhere.")


if __name__ == "__main__":
    main()
