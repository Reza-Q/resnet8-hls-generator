"""Load a PyTorch checkpoint and normalise it into a plain float state dict.

Supported inputs:

* a plain ``state_dict`` (``{name: tensor}``),
* a dict that wraps the state dict under one of :data:`STATE_DICT_KEYS`,
* a fully pickled ``nn.Module``,
* the state dict of a statically quantized model (packed linear parameters and
  quantized tensors are dequantized, and identity batch-norm parameters are
  injected because the checkpoint no longer contains them).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

STATE_DICT_KEYS = ("model_state", "state_dict", "model_state_dict", "net", "model", "params")
"""Keys under which a training script may have stored the actual state dict."""

_PACKED_LINEAR_SUFFIX = "._packed_params._packed_params"


def is_quantized_state_dict(sd: Mapping[str, Any]) -> bool:
    """Return True if any entry is a quantized tensor."""
    return any(isinstance(v, torch.Tensor) and v.is_quantized for v in sd.values())


def has_packed_linear(sd: Mapping[str, Any]) -> bool:
    """Return True if the state dict contains packed (quantized) linear parameters."""
    return any("_packed_params" in k for k in sd)


def unpack_linear(sd: Mapping[str, Any]) -> dict[str, Any]:
    """Expand ``nnq.Linear`` packed parameters into ``<name>.weight`` / ``<name>.bias``."""
    out: dict[str, Any] = {}
    for key, value in sd.items():
        if _PACKED_LINEAR_SUFFIX in key:
            base = key.replace(_PACKED_LINEAR_SUFFIX, "")
            try:
                weight, bias = value
                out[f"{base}.weight"] = weight.dequantize() if weight.is_quantized else weight
                if bias is not None:
                    out[f"{base}.bias"] = bias
            except Exception as exc:  # noqa: BLE001 - keep going, but tell the user
                logger.warning("[quant] WARNING: could not unpack %s: %s", key, exc)
        elif "_packed_params" not in key:
            out[key] = value
    return out


def dequantize_state_dict(sd: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Dequantize tensors and drop quantization bookkeeping entries."""
    skip_suffixes = (".scale", ".zero_point")
    skip_substrings = ("activation_post_process", "skip_add")
    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if any(key.endswith(s) for s in skip_suffixes):
            continue
        if any(s in key for s in skip_substrings):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu()
        out[key] = tensor.dequantize() if tensor.is_quantized else tensor
    return out


def inject_identity_batchnorm(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Add identity batch-norm parameters where a quantized model has folded them away."""
    out = dict(sd)

    def _inject(prefix: str, channels: int) -> None:
        defaults = (
            (".weight", torch.ones(channels)),
            (".bias", torch.zeros(channels)),
            (".running_mean", torch.zeros(channels)),
            (".running_var", torch.ones(channels)),
        )
        for suffix, value in defaults:
            if prefix + suffix not in out:
                out[prefix + suffix] = value

    if "conv1.weight" in out and "bn1.weight" not in out:
        _inject("bn1", out["conv1.weight"].shape[0])
    stage = 1
    while f"layer{stage}.0.conv1.weight" in out:
        block = 0
        while f"layer{stage}.{block}.conv1.weight" in out:
            prefix = f"layer{stage}.{block}"
            _inject(f"{prefix}.bn1", out[f"{prefix}.conv1.weight"].shape[0])
            _inject(f"{prefix}.bn2", out[f"{prefix}.conv2.weight"].shape[0])
            projection = f"{prefix}.shortcut.0.weight"
            if projection in out:
                _inject(f"{prefix}.shortcut.1", out[projection].shape[0])
            block += 1
        stage += 1
    return out


def extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    """Find the state dict inside whatever object ``torch.load`` returned."""
    if isinstance(checkpoint, dict):
        matched = next((k for k in STATE_DICT_KEYS if k in checkpoint), None)
        if matched is not None:
            logger.info("Loaded from key: '%s'", matched)
            state = checkpoint[matched]
        elif any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            logger.info("Loaded state dict directly")
            state = checkpoint
        else:
            raise KeyError(f"Cannot find state dict. Keys: {list(checkpoint.keys())}")
    else:
        logger.info("Loaded from nn.Module")
        state = checkpoint
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    return state


def load_state_dict_numpy(path: str | Path, *, weights_only: bool = False) -> dict[str, np.ndarray]:
    """Load a checkpoint and return its parameters as float numpy arrays.

    Security note:
        ``torch.load`` uses pickle. With ``weights_only=False`` (needed for
        quantized checkpoints and pickled modules) a malicious file can execute
        arbitrary code, so only load checkpoints you trust.

    Args:
        path: Checkpoint file.
        weights_only: Restrict unpickling to tensors and plain containers.

    Returns:
        Mapping from parameter name to numpy array.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If no state dict can be located in the file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=weights_only)
    state = extract_state_dict(checkpoint)

    if has_packed_linear(state):
        state = unpack_linear(state)
        logger.info("[quant] Unpacked nnq.Linear _packed_params")
    if is_quantized_state_dict(state):
        state = dequantize_state_dict(state)
        state = inject_identity_batchnorm(state)
        logger.info("[quant] Dequantized + identity BN injected")
    else:
        state = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}

    return {k: v.detach().cpu().numpy() for k, v in state.items()}


def describe_state_dict(sd_np: Mapping[str, np.ndarray]) -> str:
    """Return a text table of parameter names and shapes (bookkeeping entries hidden)."""
    lines = ["", f"{'Key':<58} Shape", "-" * 75]
    for key, value in sd_np.items():
        if "num_batches_tracked" not in key:
            lines.append(f"{key:<58} {value.shape}")
    return "\n".join(lines)
