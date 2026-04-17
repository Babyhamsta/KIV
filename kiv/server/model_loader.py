"""HuggingFace model loading with KIV middleware installation.

KIV drives PyTorch attention functions and operates on safetensors
weights; the ollama GGUF store is incompatible. This module always
resolves a model through the HuggingFace hub.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Bundle returned by :func:`load_model`.

    ``display_name`` is what the ollama API will echo back in responses
    (``name``, ``model`` fields). Defaults to the repo id.
    """

    model: Any
    tokenizer: Any
    middleware: Any
    display_name: str
    repo_id: str


def load_model(
    repo_id: str,
    *,
    quantize: str | None = None,
    device_map: str = "auto",
    display_name: str | None = None,
    dtype: str = "auto",
    kiv_config: Any = None,
    trust_remote_code: bool = False,
    attn_implementation: str = "eager",
) -> LoadedModel:
    """Load a HuggingFace model and install the KIV middleware.

    Args:
        repo_id: HuggingFace repo id (``org/name``) or local path.
        quantize: ``"4bit"`` or ``"8bit"`` via bitsandbytes. ``None`` for
            native precision.
        device_map: passed through to ``from_pretrained``.
        display_name: name surfaced via /api/tags. Defaults to repo_id.
        dtype: ``"auto"``, ``"float16"``, ``"bfloat16"``, or ``"float32"``.
        kiv_config: optional :class:`kiv.KIVConfig`. Defaults used otherwise.
        trust_remote_code: forwarded to ``from_pretrained``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..config import KIVConfig
    from ..middleware import KIVMiddleware

    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,
    }

    torch_dtype = _resolve_dtype(dtype)
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype

    if quantize is not None:
        kwargs["quantization_config"] = _build_quant_config(quantize, torch_dtype)

    logger.info("Loading model %s (quantize=%s, dtype=%s)", repo_id, quantize, dtype)
    model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=trust_remote_code
    )

    middleware = KIVMiddleware(model, kiv_config or KIVConfig())
    middleware.install()

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        middleware=middleware,
        display_name=display_name or repo_id,
        repo_id=repo_id,
    )


def _resolve_dtype(dtype: str) -> torch.dtype | None:
    mapping = {
        "auto": None,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(
            f"Unknown dtype '{dtype}'. Expected one of: {sorted(mapping)}."
        )
    return mapping[dtype]


def _build_quant_config(quantize: str, torch_dtype: torch.dtype | None) -> Any:
    """Build a BitsAndBytesConfig for 4-bit or 8-bit quantization.

    4-bit quantization uses NF4 with the model's compute dtype, which is
    the configuration KIV is validated against for 12 GB consumer GPUs.
    When the caller passes no explicit dtype, the compute dtype is
    chosen based on the active CUDA device's bf16 support: Ampere and
    newer get bfloat16, older hardware (Turing, Pascal) gets float16 so
    the 4-bit kernels do not fall through to slow emulated paths.
    """
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Quantization requires bitsandbytes. Install with: "
            "pip install 'kiv[quantization]'"
        ) from exc

    compute_dtype = torch_dtype or _default_bnb_compute_dtype()

    if quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )
    if quantize == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unknown quantize value '{quantize}'. Use '4bit' or '8bit'.")


def _default_bnb_compute_dtype() -> torch.dtype:
    """Pick a safe default compute dtype for bitsandbytes quantization."""
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except (AttributeError, RuntimeError):  # pragma: no cover - defensive
            pass
        logger.warning(
            "CUDA device does not report bf16 support; falling back to "
            "float16 for bitsandbytes compute dtype. Pass --dtype bfloat16 "
            "to override."
        )
        return torch.float16
    return torch.float16
