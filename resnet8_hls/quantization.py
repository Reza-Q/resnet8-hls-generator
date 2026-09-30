"""Fixed-point helpers that mirror Vitis HLS ``ap_fixed`` behaviour."""

from __future__ import annotations

import io
import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from .config import FixedPoint, Precision

LOW_SQNR_DB = 20.0
"""Layers whose SQNR falls below this value are flagged in the report."""


def quantize_hls(arr: np.ndarray, total_bits: int, int_bits: int) -> np.ndarray:
    """Quantize to ``ap_fixed<W,I,AP_TRN,AP_SAT>``: floor, then saturate.

    This matches the arithmetic of the generated hardware exactly.

    Args:
        arr: Values to quantize.
        total_bits: Word length W.
        int_bits: Integer bits I.

    Returns:
        A float64 array holding the representable values.
    """
    scale = 2.0 ** (total_bits - int_bits)
    q_max = 2 ** (total_bits - 1) - 1
    q_min = -(2 ** (total_bits - 1))
    q = np.floor(np.asarray(arr, dtype=np.float64) * scale)
    return (np.clip(q, q_min, q_max) / scale).astype(np.float64)


def quantize_round(arr: np.ndarray, total_bits: int, int_bits: int) -> np.ndarray:
    """Round-to-nearest plus saturation, used as the reference for SQNR evaluation.

    Returns:
        A float32 array holding the representable values.
    """
    scale = 2.0 ** (total_bits - int_bits)
    q_max = 2 ** (total_bits - 1) - 1
    q_min = -(2 ** (total_bits - 1))
    q = np.round(np.asarray(arr, dtype=np.float64) * scale)
    return (np.clip(q, q_min, q_max) / scale).astype(np.float32)


def quantize_fused_weights(
    fused: Mapping[str, Sequence[np.ndarray]], precision: Precision
) -> dict[str, tuple[np.ndarray, ...]]:
    """Post-training quantization (PTQ): snap BN-fused parameters to the HLS grid.

    Every parameter tuple alternates weight, bias, weight, bias, ... Weights (even
    positions) use the ``data`` format and biases (odd positions) use the ``acc`` format.
    Quantization is floor plus saturation, exactly like ``ap_fixed<W,I,AP_TRN,AP_SAT>``.

    Returns:
        A new mapping with float32 arrays; the input is left untouched.
    """
    quantized: dict[str, tuple[np.ndarray, ...]] = {}
    for name, arrays in fused.items():
        snapped = []
        for position, arr in enumerate(arrays):
            fmt = precision.data if position % 2 == 0 else precision.acc
            snapped.append(quantize_hls(arr, fmt.total_bits, fmt.int_bits).astype(np.float32))
        quantized[name] = tuple(snapped)
    return quantized


def format_cpp_array(arr: np.ndarray, depth: int = 0) -> str:
    """Render a numpy array as a C++ brace initializer with 10 decimal places.

    Args:
        arr: 1-D or N-D array of numbers.
        depth: Indentation level (four spaces per level) of the outermost brace.
    """
    buf = io.StringIO()

    def _write(a: np.ndarray, level: int) -> None:
        indent = "    " * level
        if a.ndim == 1:
            buf.write(indent + "{")
            buf.write(", ".join(f"{float(v):.10f}f" for v in a))
            buf.write("}")
        else:
            buf.write(indent + "{\n")
            for i, sub in enumerate(a):
                _write(sub, level + 1)
                buf.write(",\n" if i < len(a) - 1 else "\n")
            buf.write(indent + "}")

    _write(arr, depth)
    return buf.getvalue()


def quantization_report(
    fused: Mapping[str, Iterable[np.ndarray]],
    layers: Iterable[str],
    fmt: FixedPoint,
) -> dict[str, dict[str, float]]:
    """Compute per-layer quantization error and SQNR on BN-fused parameters.

    Args:
        fused: Mapping from HLS layer name to its parameter arrays.
        layers: Layer names to inspect, in network order.
        fmt: Fixed-point format the parameters are quantized to.

    Returns:
        ``{layer: {"max_err": ..., "sqnr_db": ...}}`` for every layer that has
        non-trivial parameters. The worst case over the layer's arrays is reported.
    """
    total_bits, int_bits = fmt
    report: dict[str, dict[str, float]] = {}
    for name in layers:
        if name == "gap":
            continue
        arrays = list(fused.get(name, ()))
        if not arrays:
            continue
        errors: list[float] = []
        sqnrs: list[float] = []
        for arr in arrays:
            if arr.size == 0 or float(np.abs(arr).max()) < 1e-9:
                continue
            ref = arr.astype(np.float64)
            quant = quantize_round(arr, total_bits, int_bits).astype(np.float64)
            signal = float(np.mean(ref**2))
            noise = float(np.mean((ref - quant) ** 2))
            errors.append(float(np.abs(ref - quant).max()))
            sqnrs.append(10 * math.log10(signal / noise) if noise > 1e-30 else 99.0)
        if not errors:
            continue
        report[name] = {"max_err": max(errors), "sqnr_db": min(sqnrs)}
    return report


def format_quantization_report(report: Mapping[str, Mapping[str, float]], fmt: FixedPoint) -> str:
    """Render :func:`quantization_report` output as a fixed-width text table."""
    total_bits, int_bits = fmt
    lines = [
        "",
        f"{'Layer':<22} {'bits':>5} {'int_b':>6} {'SQNR_dB':>9} {'max_err':>10}",
        "-" * 58,
    ]
    for name, stats in report.items():
        flag = "  <-- LOW SQNR" if stats["sqnr_db"] < LOW_SQNR_DB else ""
        lines.append(
            f"  {name:<20} {total_bits:>5} {int_bits:>6} "
            f"{stats['sqnr_db']:>9.1f} {stats['max_err']:>10.6f}{flag}"
        )
    lines += ["", "Quantization check done."]
    return "\n".join(lines)
