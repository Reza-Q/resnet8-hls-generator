"""Configuration objects shared by all pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

DEFAULT_ACC_BITS = (8, 4)
DEFAULT_PART = "xc7z020clg400-2"
DEFAULT_CLOCK_NS = 10


class FixedPoint(NamedTuple):
    """A signed fixed-point format, i.e. ``ap_fixed<total_bits, int_bits>``.

    Attributes:
        total_bits: Word length W, sign bit included.
        int_bits: Number of integer bits I, sign bit included (Vitis HLS convention).
    """

    total_bits: int
    int_bits: int

    @property
    def frac_bits(self) -> int:
        """Number of fractional bits."""
        return self.total_bits - self.int_bits

    def cpp_type(self) -> str:
        """Return the Vitis HLS type using truncation and saturation."""
        return f"ap_fixed<{self.total_bits},{self.int_bits},AP_TRN,AP_SAT>"


@dataclass(frozen=True)
class Precision:
    """Fixed-point formats used by a layer (or by the whole network).

    Attributes:
        data: Format of weights and activations (``data_t`` in the generated code).
        acc: Format of accumulators and biases (``acc_t`` in the generated code).
    """

    data: FixedPoint
    acc: FixedPoint


@dataclass(frozen=True)
class QATConfig:
    """Hyper-parameters of the quantization-aware training (QAT) stage."""

    epochs: int = 10
    lr: float = 1e-3
    lr_min: float = 1e-5
    batch_size: int = 128
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    momentum: float = 0.9


@dataclass
class PipelineConfig:
    """Everything :func:`resnet8_hls.pipeline.run_pipeline` needs to run.

    Attributes:
        checkpoint: Path to the trained PyTorch checkpoint.
        output_dir: Directory that receives ``pragma_*`` folders and ``create_project.tcl``.
        data_dir: Directory used to store / look up CIFAR-10.
        acc_bits: Accumulator / bias format.
        data_bits: Weight / activation format. ``None`` means "same as ``acc_bits``".
        enable_qat: Use QAT when true, post-training quantization (PTQ) otherwise.
        qat: QAT hyper-parameters.
        num_test_samples: Number of CIFAR-10 test images written to the C++ testbench.
        part: Xilinx part number used in the Tcl script.
        clock_ns: Target clock period in nanoseconds.
        seed: Seed for the QAT stage (``None`` leaves the global RNG untouched).
        weights_only: Pass ``weights_only=True`` to ``torch.load`` (plain state dicts only).
        download: Allow torchvision to download CIFAR-10 when it is missing.
    """

    checkpoint: Path
    output_dir: Path
    data_dir: Path = Path("data")
    acc_bits: FixedPoint = FixedPoint(*DEFAULT_ACC_BITS)
    data_bits: FixedPoint | None = None
    enable_qat: bool = True
    qat: QATConfig = field(default_factory=QATConfig)
    num_test_samples: int = 100
    part: str = DEFAULT_PART
    clock_ns: float = DEFAULT_CLOCK_NS
    seed: int | None = 0
    weights_only: bool = False
    download: bool = True

    def __post_init__(self) -> None:
        self.checkpoint = Path(self.checkpoint)
        self.output_dir = Path(self.output_dir)
        self.data_dir = Path(self.data_dir)
        for label, fmt in (("acc_bits", self.acc_bits), ("data_bits", self.data_bits)):
            if fmt is not None and not 0 < fmt.int_bits <= fmt.total_bits:
                raise ValueError(
                    f"{label} must satisfy 0 < int_bits <= total_bits, got {tuple(fmt)}"
                )
        if self.num_test_samples < 1:
            raise ValueError("num_test_samples must be at least 1")
        if self.clock_ns <= 0:
            raise ValueError("clock_ns must be positive")

    @property
    def precision(self) -> Precision:
        """Resolve the data/accumulator formats (data defaults to the accumulator format)."""
        data = self.data_bits if self.data_bits is not None else self.acc_bits
        return Precision(data=data, acc=self.acc_bits)
