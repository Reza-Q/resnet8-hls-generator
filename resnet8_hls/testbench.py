"""C++ testbench generation (``testbench/test_samples.h`` and ``tb_resnet8.cpp``).

The testbench feeds CIFAR-10 test images, quantized exactly like the hardware input, to
the top-level HLS function and compares the predicted class with the label and the
logits with a float32 PyTorch reference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .codegen import render_template, write_text
from .config import FixedPoint
from .model import IMG_CHANNELS, INPUT_SIZE
from .quantization import quantize_hls

logger = logging.getLogger(__name__)

TESTBENCH_DIRNAME = "testbench"
SAMPLES_FILENAME = "test_samples.h"
TB_SOURCE_FILENAME = "tb_resnet8.cpp"


def _write_sample(out: TextIO, index: int, image: np.ndarray) -> None:
    out.write(f"    /* sample {index} */\n    {{\n")
    for channel in range(IMG_CHANNELS):
        out.write(f"        /* ch {channel} */\n        {{\n")
        for row in range(INPUT_SIZE):
            values = ", ".join(f"{image[channel, row, col]:.10f}f" for col in range(INPUT_SIZE))
            out.write(f"            {{{values}}},\n")
        out.write("        },\n")
    out.write("    },\n")


def write_test_samples(
    path: Path,
    dataset: Dataset,
    model: nn.Module,
    data_format: FixedPoint,
    num_samples: int,
    num_classes: int,
) -> float:
    """Write ``test_samples.h`` with inputs, labels and float32 reference logits.

    Args:
        path: Header file to create.
        dataset: Dataset yielding normalized ``(image, label)`` pairs.
        model: Float reference model (in eval mode).
        data_format: Format the inputs are quantized to (floor + saturate).
        num_samples: Number of samples taken from the start of ``dataset``.
        num_classes: Number of network outputs.

    Returns:
        The float32 reference accuracy on these samples, in percent.
    """
    total_bits, int_bits = data_format
    labels: list[int] = []
    logits: list[np.ndarray] = []
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(
            f"#pragma once\n#define TB_N       {num_samples}\n"
            f"#define TB_C_IN    {IMG_CHANNELS}\n#define TB_H       {INPUT_SIZE}\n"
            f"#define TB_W       {INPUT_SIZE}\n#define TB_C_OUT   {num_classes}\n\n"
        )
        out.write(
            f"static const float tb_input[{num_samples}][{IMG_CHANNELS}]"
            f"[{INPUT_SIZE}][{INPUT_SIZE}] = {{\n"
        )
        for index in range(num_samples):
            image, label = dataset[index]
            # Quantize with floor + saturate so the input matches the hardware exactly.
            quantized = quantize_hls(np.asarray(image, dtype=np.float64), total_bits, int_bits)
            _write_sample(out, index, quantized)
            with torch.no_grad():
                logits.append(model(torch.as_tensor(image).unsqueeze(0))[0].numpy())
            labels.append(int(label))
        out.write("};\n\n")

        out.write(f"static const int tb_label[{num_samples}] = {{")
        out.write(", ".join(str(label) for label in labels))
        out.write("};\n\n")

        out.write(f"static const float tb_ref_logits[{num_samples}][{num_classes}] = {{\n")
        for sample_logits in logits:
            out.write("    {" + ", ".join(f"{v:.10f}f" for v in sample_logits) + "},\n")
        out.write("};\n")

    correct = sum(int(np.argmax(lg)) == label for lg, label in zip(logits, labels, strict=True))
    return correct / num_samples * 100


def write_testbench(
    design_dir: Path,
    dataset: Dataset,
    model: nn.Module,
    data_format: FixedPoint,
    num_samples: int,
    num_classes: int,
) -> float:
    """Create the ``testbench/`` folder and ``tb_resnet8.cpp`` inside ``design_dir``.

    Returns:
        The float32 reference accuracy on the exported samples, in percent.
    """
    testbench_dir = design_dir / TESTBENCH_DIRNAME
    testbench_dir.mkdir(parents=True, exist_ok=True)
    accuracy = write_test_samples(
        testbench_dir / SAMPLES_FILENAME, dataset, model, data_format, num_samples, num_classes
    )
    write_text(design_dir / TB_SOURCE_FILENAME, render_template("tb_resnet8.cpp.j2"))
    return accuracy
