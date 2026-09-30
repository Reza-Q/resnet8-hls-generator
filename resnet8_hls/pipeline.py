"""End-to-end pipeline: checkpoint -> quantization -> HLS projects -> Tcl script.

Stages (mirroring the original notebook, cell by cell):

1. Load the checkpoint and fuse batch norm into the convolutions.
2. Report the per-layer quantization error (SQNR) of the fused parameters.
3. Quantize: quantization-aware training (QAT) or post-training quantization (PTQ).
4. Explore the pragma design space (eight configurations, P0 to P7).
5. Generate one HLS source folder per configuration; design 0 also gets a testbench.
6. Write the Vitis HLS batch script ``create_project.tcl``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .checkpoint import describe_state_dict, load_state_dict_numpy
from .codegen import generate_design
from .config import PipelineConfig, Precision
from .data import EVAL_BATCH_SIZE, load_cifar10, make_loader
from .model import (
    FusedWeights,
    build_model_from_numpy,
    describe_network,
    get_fused_weights,
    hls_layer_names,
)
from .pragma_dse import PragmaConfig, format_pragma_summary, generate_pragma_set
from .qat import export_qat_weights, train_qat, uniform_bit_config
from .quantization import (
    format_quantization_report,
    quantization_report,
    quantize_fused_weights,
)
from .tcl_export import write_tcl
from .testbench import write_testbench

logger = logging.getLogger(__name__)

PRAGMA_DIR_TEMPLATE = "pragma_{index}"
TCL_FILENAME = "create_project.tcl"


@dataclass(frozen=True)
class PipelineResult:
    """What :func:`run_pipeline` produced."""

    output_dir: Path
    """Root folder holding the ``pragma_*`` folders and the Tcl script."""
    tcl_path: Path
    pragma_configs: tuple[PragmaConfig, ...]
    state_dict: dict[str, np.ndarray]
    """Float state dict after quantization (QAT weights, or the original for PTQ)."""
    fused_weights: FusedWeights
    """BN-fused, quantized parameters that were written to the C++ sources."""
    quantization_report: dict[str, dict[str, float]]
    """Per-layer SQNR / max error of the original fused parameters."""
    best_val_accuracy: float | None
    """Best QAT validation accuracy in percent (``None`` when PTQ was used)."""
    reference_accuracy: float
    """Float32 reference accuracy on the testbench samples in percent."""


def _describe_bit_widths(precision: Precision, layers: list[str]) -> str:
    lines = [
        f"DATA_BITS : {precision.data.cpp_type()}",
        f"ACC_BITS  : {precision.acc.cpp_type()}",
        "",
    ]
    for layer in layers:
        lines.append(
            f"  {layer:<22} data=ap_fixed<{precision.data.total_bits},{precision.data.int_bits}>"
            f"  acc=ap_fixed<{precision.acc.total_bits},{precision.acc.int_bits}>"
        )
    return "\n".join(lines)


def run_pipeline(
    config: PipelineConfig,
    *,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
) -> PipelineResult:
    """Run the complete flow described in the module docstring.

    Args:
        config: All settings (see :class:`resnet8_hls.config.PipelineConfig`).
        train_dataset: Training data for QAT. Defaults to CIFAR-10 loaded from
            ``config.data_dir``. Must yield normalized ``(image, label)`` pairs.
        test_dataset: Validation / testbench data. Defaults to the CIFAR-10 test split.

    Returns:
        A :class:`PipelineResult` describing the generated design.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If the test dataset has fewer than ``config.num_test_samples`` samples.
    """
    precision = config.precision
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Checkpoint and batch-norm fusion.
    sd_np = load_state_dict_numpy(config.checkpoint, weights_only=config.weights_only)
    logger.info(describe_state_dict(sd_np))
    layers = hls_layer_names(sd_np)
    logger.info("HLS layers: %s", layers)
    fused = get_fused_weights(sd_np)

    # 2. Bit widths and quantization check.
    logger.info(_describe_bit_widths(precision, layers))
    report = quantization_report(fused, layers, precision.data)
    logger.info(format_quantization_report(report, precision.data))

    if test_dataset is None:
        test_dataset = load_cifar10(config.data_dir, train=False, download=config.download)
    if config.num_test_samples > len(test_dataset):
        raise ValueError(
            f"num_test_samples={config.num_test_samples} exceeds the test set size "
            f"({len(test_dataset)})"
        )

    # 3. Quantization: QAT or PTQ.
    best_val: float | None = None
    if config.enable_qat:
        if train_dataset is None:
            train_dataset = load_cifar10(config.data_dir, train=True, download=config.download)
        if config.seed is not None:
            torch.manual_seed(config.seed)
        qat_model, best_val = train_qat(
            build_model_from_numpy(sd_np),
            precision,
            make_loader(train_dataset, config.qat.batch_size, shuffle=True, drop_last=True),
            make_loader(test_dataset, EVAL_BATCH_SIZE, shuffle=False),
            config.qat,
            bit_cfg=uniform_bit_config(layers, precision),
        )
        sd_np = export_qat_weights(qat_model)
        fused = get_fused_weights(sd_np)
        logger.info("[QAT] best_val=%.2f%%  state dict and fused weights updated.", best_val)
    else:
        logger.info("[PTQ] Snapping BN-fused weights to HLS grid (floor+clamp).")
        fused = quantize_fused_weights(fused, precision)
        logger.info(
            "[PTQ] Done.  data=ap_fixed<%d,%d>  acc=ap_fixed<%d,%d>",
            *precision.data,
            *precision.acc,
        )

    # 4. Pragma design-space exploration.
    spec = describe_network(sd_np)
    pragma_set = generate_pragma_set(spec)
    logger.info(format_pragma_summary(pragma_set))

    # 5. Code generation. The reference model uses the (post-QAT) float state dict.
    reference_model = build_model_from_numpy(sd_np)
    reference_accuracy = float("nan")
    for idx, pragmas in enumerate(pragma_set):
        design_dir = config.output_dir / PRAGMA_DIR_TEMPLATE.format(index=idx)
        logger.info("\n%s\n[GEN] pragma_%d  (%s)", "=" * 60, idx, pragmas.label)
        generate_design(design_dir, spec, fused, pragmas, precision)
        if idx == 0:  # Only the first design gets a testbench (csim / cosim target).
            reference_accuracy = write_testbench(
                design_dir,
                test_dataset,
                reference_model,
                precision.data,
                config.num_test_samples,
                spec.fc_out,
            )
            logger.info(
                "  [testbench] %d samples  float32_ref=%.1f%%",
                config.num_test_samples,
                reference_accuracy,
            )
        logger.info("  [common/top] written  dataflow=%s", pragmas.use_dataflow)
    logger.info(
        "\n[GEN_ALL] Done -- all %d pragma folders written under %s/",
        len(pragma_set),
        config.output_dir,
    )

    # 6. Vitis HLS batch script.
    tcl_path = write_tcl(
        config.output_dir / TCL_FILENAME,
        pragma_set,
        [block.name for block in spec.blocks],
        precision,
        config.part,
        config.clock_ns,
    )
    logger.info("[TCL] %s written  (%d projects)", tcl_path, len(pragma_set))
    logger.info("      P0: csim + csynth + cosim  |  P1-P%d: csynth only", len(pragma_set) - 1)
    logger.info("\nRun from %s/:\n  vitis_hls -f %s", config.output_dir, TCL_FILENAME)

    return PipelineResult(
        output_dir=config.output_dir,
        tcl_path=tcl_path,
        pragma_configs=tuple(pragma_set),
        state_dict=sd_np,
        fused_weights=fused,
        quantization_report=report,
        best_val_accuracy=best_val,
        reference_accuracy=reference_accuracy,
    )
