"""Generate quantized Vitis HLS C++ projects from a trained PyTorch CIFAR-10 ResNet.

Quick start::

    from resnet8_hls import PipelineConfig, run_pipeline

    run_pipeline(PipelineConfig(checkpoint="models/model.pth", output_dir="hls_output/model"))
"""

from .config import FixedPoint, PipelineConfig, Precision, QATConfig
from .pipeline import PipelineResult, run_pipeline

__version__ = "0.1.0"

__all__ = [
    "FixedPoint",
    "PipelineConfig",
    "PipelineResult",
    "Precision",
    "QATConfig",
    "__version__",
    "run_pipeline",
]
