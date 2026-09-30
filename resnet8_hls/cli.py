"""Command-line interface: ``resnet8-hls`` or ``python -m resnet8_hls``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import (
    DEFAULT_ACC_BITS,
    DEFAULT_CLOCK_NS,
    DEFAULT_PART,
    FixedPoint,
    PipelineConfig,
    QATConfig,
)
from .pipeline import run_pipeline

DEFAULT_OUTPUT_ROOT = Path("hls_output")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    defaults = QATConfig()
    parser = argparse.ArgumentParser(
        prog="resnet8-hls",
        description=(
            "Generate quantized Vitis HLS C++ projects (eight pragma configurations) "
            "from a trained PyTorch CIFAR-10 ResNet checkpoint."
        ),
    )
    parser.add_argument("checkpoint", type=Path, help="path to the trained .pth checkpoint")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help=f"output folder (default: {DEFAULT_OUTPUT_ROOT}/<checkpoint name>)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="CIFAR-10 folder (default: data)"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="do not download CIFAR-10 if it is missing",
    )

    quant = parser.add_argument_group("quantization")
    quant.add_argument(
        "--acc-bits",
        type=int,
        nargs=2,
        metavar=("TOTAL", "INT"),
        default=DEFAULT_ACC_BITS,
        help="accumulator/bias format ap_fixed<TOTAL,INT> (default: 8 4)",
    )
    quant.add_argument(
        "--data-bits",
        type=int,
        nargs=2,
        metavar=("TOTAL", "INT"),
        help="weight/activation format ap_fixed<TOTAL,INT> (default: same as --acc-bits)",
    )
    quant.add_argument(
        "--no-qat",
        action="store_true",
        help="use post-training quantization instead of quantization-aware training",
    )

    qat = parser.add_argument_group("quantization-aware training")
    qat.add_argument(
        "--epochs", type=int, default=defaults.epochs, help="training epochs (default: %(default)s)"
    )
    qat.add_argument(
        "--lr", type=float, default=defaults.lr, help="initial learning rate (default: %(default)s)"
    )
    qat.add_argument(
        "--lr-min",
        type=float,
        default=defaults.lr_min,
        help="final learning rate of the cosine schedule (default: %(default)s)",
    )
    qat.add_argument(
        "--batch-size",
        type=int,
        default=defaults.batch_size,
        help="batch size (default: %(default)s)",
    )
    qat.add_argument(
        "--weight-decay",
        type=float,
        default=defaults.weight_decay,
        help="SGD weight decay (default: %(default)s)",
    )
    qat.add_argument(
        "--momentum",
        type=float,
        default=defaults.momentum,
        help="SGD momentum (default: %(default)s)",
    )
    qat.add_argument(
        "--grad-clip",
        type=float,
        default=defaults.grad_clip,
        help="gradient-norm clipping threshold (default: %(default)s)",
    )
    qat.add_argument("--seed", type=int, default=0, help="random seed for QAT (default: 0)")

    hls = parser.add_argument_group("HLS output")
    hls.add_argument("--part", default=DEFAULT_PART, help="Xilinx part (default: %(default)s)")
    hls.add_argument(
        "--clock-ns",
        type=float,
        default=DEFAULT_CLOCK_NS,
        help="target clock period in ns (default: %(default)s)",
    )
    hls.add_argument(
        "--num-test-samples",
        type=int,
        default=100,
        help="CIFAR-10 test images exported to the C++ testbench (default: 100)",
    )

    parser.add_argument(
        "--weights-only",
        action="store_true",
        help=(
            "load the checkpoint with torch.load(weights_only=True) "
            "(safer, but plain state dicts only)"
        ),
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only print warnings and errors")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Translate parsed arguments into a :class:`PipelineConfig`."""
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / args.checkpoint.stem
    return PipelineConfig(
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        data_dir=args.data_dir,
        acc_bits=FixedPoint(*args.acc_bits),
        data_bits=FixedPoint(*args.data_bits) if args.data_bits else None,
        enable_qat=not args.no_qat,
        qat=QATConfig(
            epochs=args.epochs,
            lr=args.lr,
            lr_min=args.lr_min,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            momentum=args.momentum,
        ),
        num_test_samples=args.num_test_samples,
        part=args.part,
        clock_ns=args.clock_ns,
        seed=args.seed,
        weights_only=args.weights_only,
        download=not args.no_download,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    try:
        run_pipeline(config_from_args(args))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
