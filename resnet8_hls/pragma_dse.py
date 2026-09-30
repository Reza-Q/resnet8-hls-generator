"""Design-space exploration: eight lean Vitis HLS pragma configurations (P0 to P7).

Every configuration keeps the design small (no array partitioning by default and at
most a tiny unroll factor) so that all eight projects fit comfortably on a small
device such as the Zynq-7020.

Conventions used by the pragma parameters:

* ``unroll_*``: ``0`` means complete unroll, ``1`` means no unroll (sequential loop)
  and ``n > 1`` means unroll factor ``n``.
* ``partition_*_factor``: ``0`` means complete partition, ``1`` means no partition and
  ``n > 1`` means cyclic partition factor ``n``.

The DSP / LUT / latency numbers are cheap analytical estimates used to compare
configurations with each other. They are *not* a substitute for the Vitis HLS
synthesis reports.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .model import IMG_CHANNELS, INPUT_SIZE, NetworkSpec

logger = logging.getLogger(__name__)

KERNEL_SIZE = 3
"""Spatial size of the convolution kernels."""


@dataclass(frozen=True)
class ConvPragmas:
    """Pragma settings of a 3x3 convolution (the stem or a residual block)."""

    ii: int
    """Initiation interval of the pipelined output-pixel loop."""
    unroll_ic: int = 1
    """Unroll setting of the input-channel loop."""
    unroll_kh: int = 1
    """Unroll setting of the kernel-height loop."""
    unroll_kw: int = 1
    """Unroll setting of the kernel-width loop."""
    partition_w_factor: int = 1
    """Partition setting of the weight array (dimension 4, the kernel width)."""
    partition_mid_factor: int = 1
    """Partition setting of the intermediate buffer (dimension 1, the channels)."""


@dataclass(frozen=True)
class FCPragmas:
    """Pragma settings of the fully connected layer."""

    ii: int
    """Initiation interval of the pipelined output loop."""
    unroll_i: int = 1
    """Unroll setting of the input loop."""


@dataclass(frozen=True)
class PragmaConfig:
    """One point of the design space, plus its estimated cost."""

    label: str
    description: str
    conv1: ConvPragmas
    resblock: ConvPragmas
    fc: FCPragmas
    use_dataflow: bool = False
    est_dsp: int = 0
    est_lut_k: float = 0.0
    """Estimated LUT count in thousands."""
    est_latency_k: float = 0.0
    """Estimated latency in thousands of clock cycles."""
    warning_flags: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def divisors_desc(n: int, limit: int) -> list[int]:
    """Divisors of ``n`` that do not exceed ``limit``, largest first."""
    return sorted((d for d in range(1, n + 1) if n % d == 0 and d <= limit), reverse=True)


def safe_unroll(loop_bound: int, requested: int) -> int:
    """Turn a requested unroll factor into a legal unroll setting for a loop.

    Returns ``0`` (complete unroll) when the request covers the whole loop, ``1`` for a
    sequential loop, and otherwise the largest divisor of ``loop_bound`` that does not
    exceed ``requested``.
    """
    if requested == 0 or requested >= loop_bound:
        return 0
    if requested <= 1:
        return 1
    best = divisors_desc(loop_bound, requested)
    return best[0] if best else 1


def _lanes(loop_bound: int, unroll: int) -> int:
    """Number of parallel lanes created by an unroll setting."""
    if unroll == 0:
        return loop_bound
    return unroll if unroll > 1 else 1


def parallel_macs(c_in: int, unroll_ic: int, unroll_kh: int, unroll_kw: int) -> int:
    """Number of multiply-accumulate units a 3x3 convolution needs in parallel."""
    return _lanes(c_in, unroll_ic) * _lanes(KERNEL_SIZE, unroll_kh) * _lanes(KERNEL_SIZE, unroll_kw)


def conv_pragmas(c_in: int, ii: int, unroll_ic: int, unroll_kh: int, unroll_kw: int) -> ConvPragmas:
    """Build a convolution entry: no array partitioning by default."""
    return ConvPragmas(
        ii=max(1, ii),
        unroll_ic=safe_unroll(c_in, unroll_ic),
        unroll_kh=safe_unroll(KERNEL_SIZE, unroll_kh),
        unroll_kw=safe_unroll(KERNEL_SIZE, unroll_kw),
        partition_w_factor=1,
        partition_mid_factor=1,
    )


def fc_pragmas(fc_in: int, ii: int, unroll_i: int) -> FCPragmas:
    """Build a fully connected layer entry."""
    return FCPragmas(ii=max(1, ii), unroll_i=safe_unroll(fc_in, unroll_i))


# --------------------------------------------------------------------------- #
# Cost estimates
# --------------------------------------------------------------------------- #
def _conv_macs(c_in: int, pragmas: ConvPragmas) -> int:
    return parallel_macs(c_in, pragmas.unroll_ic, pragmas.unroll_kh, pragmas.unroll_kw)


def estimate_dsp(spec: NetworkSpec, cfg: PragmaConfig) -> int:
    """Estimate the number of DSP slices (one per parallel MAC)."""
    dsp = _conv_macs(IMG_CHANNELS, cfg.conv1)
    for block in spec.blocks:
        dsp += _conv_macs(block.c_in, cfg.resblock)
        dsp += _conv_macs(block.c_mid, cfg.resblock)
    dsp += _lanes(spec.fc_in, cfg.fc.unroll_i)
    return dsp


def estimate_lut_k(spec: NetworkSpec, cfg: PragmaConfig) -> float:
    """Estimate the LUT count in thousands (a fixed base plus a per-MAC cost)."""
    lut = 1.2 + _conv_macs(IMG_CHANNELS, cfg.conv1) * 0.020
    for block in spec.blocks:
        lut += (
            _conv_macs(block.c_in, cfg.resblock) + _conv_macs(block.c_mid, cfg.resblock)
        ) * 0.020
    lut += _lanes(spec.fc_in, cfg.fc.unroll_i) * 0.018
    return round(lut, 1)


def _passes(macs: int, total: int) -> int:
    """How many sequential passes are needed to cover ``total`` MACs with ``macs`` units."""
    return 1 if macs >= total else total // max(1, macs)


def estimate_latency_k(spec: NetworkSpec, cfg: PragmaConfig) -> float:
    """Estimate the latency in thousands of clock cycles."""
    latencies: list[int] = []

    macs = _conv_macs(IMG_CHANNELS, cfg.conv1)
    passes = _passes(macs, IMG_CHANNELS * KERNEL_SIZE**2)
    latencies.append(spec.stem_channels * INPUT_SIZE * INPUT_SIZE * cfg.conv1.ii * passes)

    for block in spec.blocks:
        for c_loop in (block.c_in, block.c_mid):
            passes = _passes(_conv_macs(c_loop, cfg.resblock), c_loop * KERNEL_SIZE**2)
            latencies.append(block.c_mid * block.h_out * block.h_out * cfg.resblock.ii * passes)

    lanes = _lanes(spec.fc_in, cfg.fc.unroll_i)
    fc_passes = 1 if lanes >= spec.fc_in else spec.fc_in // max(1, lanes)
    latencies.append(spec.fc_out * fc_passes * cfg.fc.ii)
    return round(sum(latencies) / 1000, 1)


def check_warnings(cfg: PragmaConfig) -> tuple[str, ...]:
    """Flag configurations that are likely to be limited by memory ports."""
    warnings = []
    if cfg.resblock.unroll_ic > 1 and cfg.resblock.partition_w_factor == 1:
        warnings.append("W-MEMPORT [resblock] IC unroll active but no weight partition")
    return tuple(warnings)


# --------------------------------------------------------------------------- #
# The pragma set
# --------------------------------------------------------------------------- #
def generate_pragma_set(spec: NetworkSpec) -> list[PragmaConfig]:
    """Create the eight configurations P0-P7 and attach their cost estimates.

    Args:
        spec: Shape information of the network (see :func:`resnet8_hls.model.describe_network`).
    """
    c_max = max((block.c_in for block in spec.blocks), default=1)
    fc_in = spec.fc_in
    logger.info(
        "[PRAGMA_DSE] c_stem=%d  c_max=%d  fc_in=%d  fc_out=%d",
        spec.stem_channels,
        c_max,
        fc_in,
        spec.fc_out,
    )

    stem = IMG_CHANNELS
    configs: list[PragmaConfig] = []

    # P0: II=8 everywhere, no unroll, no partition -- absolute minimum resources.
    configs.append(
        PragmaConfig(
            label="P0_absolute_min_ii8",
            description="II=8 all layers, no unroll, no partition",
            conv1=conv_pragmas(stem, 8, 1, 1, 1),
            resblock=conv_pragmas(c_max, 8, 1, 1, 1),
            fc=fc_pragmas(fc_in, 8, 1),
        )
    )

    # P1: II=4, no unroll, no partition.
    configs.append(
        PragmaConfig(
            label="P1_lean_ii4",
            description="II=4 all layers, no unroll, no partition",
            conv1=conv_pragmas(stem, 4, 1, 1, 1),
            resblock=conv_pragmas(c_max, 4, 1, 1, 1),
            fc=fc_pragmas(fc_in, 4, 1),
        )
    )

    # P2: II=2, no unroll, no partition.
    configs.append(
        PragmaConfig(
            label="P2_lean_ii2",
            description="II=2 all layers, no unroll, no partition",
            conv1=conv_pragmas(stem, 2, 1, 1, 1),
            resblock=conv_pragmas(c_max, 2, 1, 1, 1),
            fc=fc_pragmas(fc_in, 2, 1),
        )
    )

    # P3: stem II=2 (c_in=3, cheap), residual blocks and FC at II=4.
    configs.append(
        PragmaConfig(
            label="P3_stem_ii2_rest_ii4",
            description="Stem II=2 (c_in=3 cheap), resblock+FC II=4",
            conv1=conv_pragmas(stem, 2, 1, 1, 1),
            resblock=conv_pragmas(c_max, 4, 1, 1, 1),
            fc=fc_pragmas(fc_in, 4, 1),
        )
    )

    # P4: conv II=4, FC II=2 -- FC latency focus.
    configs.append(
        PragmaConfig(
            label="P4_conv_ii4_fc_ii2",
            description="Conv II=4, FC II=2, no unroll -- FC-latency focus",
            conv1=conv_pragmas(stem, 4, 1, 1, 1),
            resblock=conv_pragmas(c_max, 4, 1, 1, 1),
            fc=fc_pragmas(fc_in, 2, 1),
        )
    )

    # P5: conv II=4, FC unroll=2 -- minimal FC parallelism.
    configs.append(
        PragmaConfig(
            label="P5_conv_seq_fc_x2",
            description="Conv II=4, FC unroll=2 -- FC-biased minimal parallelism",
            conv1=conv_pragmas(stem, 4, 1, 1, 1),
            resblock=conv_pragmas(c_max, 4, 1, 1, 1),
            fc=fc_pragmas(fc_in, 2, safe_unroll(fc_in, 2)),
        )
    )

    # P6: load-proportional II -- stem II=8, residual blocks II=4, FC II=2.
    configs.append(
        PragmaConfig(
            label="P6_load_proportional_ii",
            description="II=8/4/2 for stem/resblock/FC -- load-proportional",
            conv1=conv_pragmas(stem, 8, 1, 1, 1),
            resblock=conv_pragmas(c_max, 4, 1, 1, 1),
            fc=fc_pragmas(fc_in, 2, 1),
        )
    )

    # P7: residual-block IC unroll=2 (guarded by c_max > 2), everything else II=4.
    ic_unroll = safe_unroll(c_max, 2) if c_max > 2 else 1
    resblock_p7 = conv_pragmas(c_max, 4, ic_unroll, 1, 1)
    if ic_unroll > 1:
        # Complete partition of the kernel-width dimension (size 3, always safe).
        resblock_p7 = replace(resblock_p7, partition_w_factor=0)
    configs.append(
        PragmaConfig(
            label="P7_resblock_ic2_rest_seq",
            description="Resblock IC unroll=2 (guarded c_max>2), rest II=4",
            conv1=conv_pragmas(stem, 4, 1, 1, 1),
            resblock=resblock_p7,
            fc=fc_pragmas(fc_in, 4, 1),
        )
    )

    return [
        replace(
            cfg,
            est_dsp=estimate_dsp(spec, cfg),
            est_lut_k=estimate_lut_k(spec, cfg),
            est_latency_k=estimate_latency_k(spec, cfg),
            warning_flags=check_warnings(cfg),
        )
        for cfg in configs
    ]


def format_pragma_summary(configs: Sequence[PragmaConfig]) -> str:
    """Render the pragma set as a fixed-width text table."""
    width = 84
    lines = [
        "",
        "=" * width,
        f"  Pragma Set -- {len(configs)} configurations  (est. x2.5 actual LUT)",
        "=" * width,
        f"  {'#':<3} {'Label':<35} {'DSP':>5} {'LUT(k)':>7} {'Lat(kcy)':>9} {'DF':>4} {'W':>4}",
        f"  {'-' * 78}",
    ]
    for idx, cfg in enumerate(configs):
        warn = ("!" * len(cfg.warning_flags)) if cfg.warning_flags else "OK"
        dataflow = "Y" if cfg.use_dataflow else "N"
        lines.append(
            f"  {idx:<3} {cfg.label:<35} {cfg.est_dsp:>5} "
            f"{cfg.est_lut_k:>7.1f} {cfg.est_latency_k:>9.1f} {dataflow:>4} {warn:>4}"
        )
    lines += [f"  {'-' * 78}", ""]
    return "\n".join(lines)
