"""
A single process-wide lock for HuggingFace model construction.

Loading two transformer models concurrently in different threads raises:

    NotImplementedError: Cannot copy out of meta tensor; no data!
    Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() …

The cause is that `accelerate.init_empty_weights()` — which `transformers` enters
during `from_pretrained` — swaps torch's **global** default device to `meta` and
restores it on exit. That override is process-global, not thread-local. So if
thread A constructs `SentenceTransformer` while thread B is inside that context,
A's parameters get allocated on the meta device and A's subsequent `.to("cpu")`
has no data to copy.

We hit this loading BGE-small and the two cross-encoders concurrently at startup.
Serializing construction costs a few seconds once, at boot, and removes a failure
that otherwise surfaces as a 500 on the first upload.

Inference is unaffected — this guards construction only, so concurrent
`encode()` / `predict()` calls still run in parallel.
"""

from __future__ import annotations

import threading

MODEL_LOAD_LOCK = threading.RLock()
"""Held only while a transformer model is being constructed."""
