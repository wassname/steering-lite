"""Suppress third-party progress bars and advisory warnings.

Call `quiet_external_logs()` once at the top of an entrypoint script to keep
benchmark logs free of HF/transformers noise. Pueue already preserves the full
log per job; the goal here is signal-to-noise, not capture.
"""
from __future__ import annotations

import os
import warnings


def quiet_external_logs() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    warnings.filterwarnings("ignore", message=".*torch_dtype.*")
    warnings.filterwarnings("ignore", message=".*generation flags.*")
    try:
        import datasets
        datasets.disable_progress_bars()
    except Exception:
        pass
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        if hasattr(hf_logging, "disable_progress_bar"):
            hf_logging.disable_progress_bar()
    except Exception:
        pass
