"""
Host resource probing, used to pick a reranking tier that actually fits.

Why this exists: BGE-reranker-base is a 278M-parameter fp32 model, ~1.1 GB
resident. On a box with headroom it reranks 4 pairs in ~420 ms. On a box with
~2 GB free it thrashes and the same batch takes ~2.2 s — a 5x regression that
looks like a code bug but is the allocator paging. We measured exactly that
during development (12-core laptop, 2 GB free, Defender active).

Rather than shipping a default that is fast on the author's machine and slow on
the reviewer's, `RERANK_MODE=auto` measures available memory at startup and picks
a tier that fits, then logs the decision so the behaviour is never a mystery.

No psutil dependency — the three platform probes below are short and avoid adding
a package for twenty lines of work.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Headroom required before loading BGE-reranker-base alongside everything else.
#
# Raised from 2.5 on measured evidence: at 3.3 GB reported free, loading the
# precise tier on top of torch + the embedder + MiniLM + the HTTP/TLS stack
# produced a hard access violation (exit 0xC0000005), not a graceful OOM — the
# allocation fails inside native code and takes the process with it. "Reported
# free" also overstates what a single process can actually get, since it counts
# memory the OS will not hand over. 4 GB is the smallest value that held.
PRECISE_TIER_GB = 4.0


@dataclass
class HostResources:
    total_gb: float
    available_gb: float
    cpu_count: int
    has_cuda: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "total_gb": round(self.total_gb, 2),
            "available_gb": round(self.available_gb, 2),
            "cpu_count": self.cpu_count,
            "cuda": self.has_cuda,
        }


def _available_memory_gb() -> tuple[float, float]:
    """(total_gb, available_gb). Returns (0, 0) when it cannot be determined."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return 0.0, 0.0
            gib = 1024**3
            return status.ullTotalPhys / gib, status.ullAvailPhys / gib

        if sys.platform.startswith("linux"):
            values: dict[str, int] = {}
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    key, _, rest = line.partition(":")
                    parts = rest.split()
                    if parts:
                        values[key] = int(parts[0])  # kB
            total = values.get("MemTotal", 0) / (1024**2)
            # MemAvailable is the kernel's own estimate and is far more accurate
            # than MemFree, which excludes reclaimable page cache.
            available = values.get("MemAvailable", values.get("MemFree", 0)) / (1024**2)
            return total, available

        # macOS / BSD: total is easy, available needs vm_stat. Report total for
        # both — cgroup-less Darwin dev machines are rarely the constrained case.
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        total = (page_size * pages) / (1024**3)
        return total, total * 0.5
    except Exception as exc:  # noqa: BLE001 — probing must never break startup
        logger.debug("Memory probe failed: %s", exc)
        return 0.0, 0.0


def cuda_info() -> tuple[bool, str, float]:
    """(available, device_name, vram_gb). Never raises."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "", 0.0
        props = torch.cuda.get_device_properties(0)
        return True, torch.cuda.get_device_name(0), props.total_memory / (1024**3)
    except Exception:  # noqa: BLE001
        return False, "", 0.0


# Peak VRAM for BGE-small + both cross-encoders, fp32, with batch activations.
_MODELS_VRAM_GB = 2.0


def resolve_device(configured: str) -> str:
    """Map a configured device to a concrete one.

    `auto` prefers CUDA when a device is present with enough VRAM for the model
    set. This is worth doing rather than defaulting to CPU: the cross-encoders are
    the dominant latency cost in retrieval and they are 20-50x faster on even a
    modest laptop GPU. If VRAM is too small we stay on CPU rather than risk an
    OOM mid-request, which would fail the turn instead of merely slowing it.
    """
    configured = (configured or "auto").strip().lower()
    if configured != "auto":
        return configured

    available, name, vram = cuda_info()
    if not available:
        return "cpu"
    if vram < _MODELS_VRAM_GB:
        logger.info(
            "CUDA device %s has only %.1f GB VRAM (< %.1f GB needed); using CPU",
            name, vram, _MODELS_VRAM_GB,
        )
        return "cpu"
    logger.info("Using CUDA device: %s (%.1f GB VRAM)", name, vram)
    return "cuda"


def _has_cuda() -> bool:
    """True only if the reranker will *actually run* on CUDA.

    `torch.cuda.is_available()` alone is the wrong question: a machine can have a
    GPU while `RERANKER_DEVICE=cpu`, in which case the precise tier still runs on
    CPU and still needs its ~1.1 GB of host RAM. Keying this on hardware presence
    rather than resolved placement picked `cascade` on a 2 GB-free host during
    development — exactly the case this guards against.
    """
    try:
        from core.config import settings

        return resolve_device(settings.reranker_device).startswith("cuda")
    except Exception:  # noqa: BLE001
        return False


def probe() -> HostResources:
    total, available = _available_memory_gb()
    return HostResources(
        total_gb=total,
        available_gb=available,
        cpu_count=os.cpu_count() or 1,
        has_cuda=_has_cuda(),
    )


def resolve_rerank_mode(configured: str) -> tuple[str, str]:
    """Map a configured mode to an effective one. Returns (mode, reason)."""
    configured = (configured or "auto").lower()
    if configured != "auto":
        return configured, f"RERANK_MODE={configured} (explicit)"

    host = probe()

    # A GPU makes the precise tier essentially free; always take the quality.
    if host.has_cuda:
        return "cascade", f"auto → cascade (CUDA available, {host.cpu_count} cores)"

    if host.available_gb <= 0:
        return "cascade", "auto → cascade (memory probe unavailable, assuming headroom)"

    if host.available_gb < PRECISE_TIER_GB:
        return (
            "fast",
            f"auto → fast ({host.available_gb:.1f} GB available < {PRECISE_TIER_GB} GB "
            f"needed for BGE-reranker-base; loading it here would page-thrash and "
            f"cost ~5x. Set RERANK_MODE=cascade to force it.)",
        )

    return (
        "cascade",
        f"auto → cascade ({host.available_gb:.1f} GB available, {host.cpu_count} cores)",
    )
