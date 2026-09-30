"""Vitis HLS C++ code generation.

All C++ text lives in Jinja2 templates under ``resnet8_hls/templates/``. Every
``generate_*`` function receives the output directory explicitly and returns the paths
it wrote; nothing here relies on module-level mutable state.

Generated layout of one design (one ``pragma_<n>`` folder)::

    common.hpp                 data_t typedef
    conv1.hpp / conv1.cpp      stem convolution (+ ReLU)
    layerS_blockB.hpp / .cpp   residual blocks
    gap_layer.hpp / .cpp       global average pooling
    fc_layer.hpp / .cpp        fully connected classifier
    resnet8.hpp / resnet8.cpp  top-level function with AXI-stream I/O
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from jinja2 import Environment, PackageLoader, StrictUndefined

from .config import Precision
from .model import IMG_CHANNELS, INPUT_SIZE, FusedWeights, NetworkSpec
from .pragma_dse import ConvPragmas, FCPragmas, PragmaConfig
from .quantization import format_cpp_array

logger = logging.getLogger(__name__)

TEMPLATE_PACKAGE = "resnet8_hls"
TEMPLATE_DIR = "templates"


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return Environment(
        loader=PackageLoader(TEMPLATE_PACKAGE, TEMPLATE_DIR),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )


def render_template(name: str, **context: object) -> str:
    """Render the template ``name`` (a file in the templates directory)."""
    return _environment().get_template(name).render(**context)


def write_text(path: Path, text: str) -> None:
    """Write ``text`` as UTF-8 with Unix line endings, regardless of the platform."""
    path.write_text(text, encoding="utf-8", newline="\n")


def _clip_unroll(unroll: int, loop_bound: int) -> int:
    """Clamp an unroll factor to the loop bound (``0`` and ``1`` keep their meaning)."""
    return min(unroll, loop_bound) if unroll not in (0, 1) else unroll


def _render_pair(
    output_dir: Path, header: str, source: str, base: str, context: dict
) -> list[Path]:
    """Render ``<header>`` / ``<source>`` templates into ``<base>.hpp`` / ``<base>.cpp``."""
    hpp = output_dir / f"{base}.hpp"
    cpp = output_dir / f"{base}.cpp"
    write_text(hpp, render_template(header, **context))
    write_text(cpp, render_template(source, **context))
    return [hpp, cpp]


# --------------------------------------------------------------------------- #
# Individual layers
# --------------------------------------------------------------------------- #
def generate_conv(
    output_dir: Path,
    name: str,
    weights: np.ndarray,
    bias: np.ndarray,
    h_in: int,
    w_in: int,
    pragmas: ConvPragmas,
    precision: Precision,
    *,
    stride: int = 1,
    relu: bool = False,
) -> list[Path]:
    """Generate a stand-alone 3x3 convolution layer (used for the stem).

    Args:
        output_dir: Folder that receives ``<name>.hpp`` and ``<name>.cpp``.
        name: HLS function name.
        weights: BN-fused weights with shape ``[c_out, c_in, 3, 3]``.
        bias: BN-fused bias with shape ``[c_out]``.
        h_in: Input height.
        w_in: Input width.
        pragmas: Pragma settings for this layer.
        precision: Data / accumulator formats.
        stride: Convolution stride.
        relu: Append a ReLU to the layer.

    Returns:
        The paths of the written files.
    """
    c_out, c_in = weights.shape[:2]
    data_w, data_i = precision.data
    acc_w, acc_i = precision.acc
    context = {
        "func": name,
        "NAME": name.upper(),
        "c_in": c_in,
        "c_out": c_out,
        "h_in": h_in,
        "w_in": w_in,
        "h_out": h_in // stride,
        "w_out": w_in // stride,
        "stride": stride,
        "relu": relu,
        "data_w": data_w,
        "data_i": data_i,
        "acc_w": acc_w,
        "acc_i": acc_i,
        "conv_ii": pragmas.ii,
        "unroll_kh": pragmas.unroll_kh,
        "unroll_kw": pragmas.unroll_kw,
        "unroll_ic": _clip_unroll(pragmas.unroll_ic, c_in),
        "partition_w_factor": pragmas.partition_w_factor,
        "weights": format_cpp_array(weights),
        "bias": format_cpp_array(bias),
    }
    written = _render_pair(output_dir, "conv.hpp.j2", "conv.cpp.j2", name, context)
    logger.info(
        "  [conv]     %-30s  [%d,%d,3,3]  stride=%d  data=ap_fixed<%d,%d>  acc=ap_fixed<%d,%d>",
        name,
        c_out,
        c_in,
        stride,
        data_w,
        data_i,
        acc_w,
        acc_i,
    )
    return written


def generate_resblock(
    output_dir: Path,
    name: str,
    arrays: tuple[np.ndarray, ...],
    h_in: int,
    w_in: int,
    pragmas: ConvPragmas,
    precision: Precision,
    *,
    stride: int = 1,
) -> list[Path]:
    """Generate one residual block.

    Args:
        output_dir: Folder that receives ``<name>.hpp`` and ``<name>.cpp``.
        name: HLS function name, e.g. ``layer2_block0``.
        arrays: ``(w1, b1, w2, b2)`` or ``(w1, b1, w2, b2, w_proj, b_proj)``; the latter
            selects a 1x1 projection shortcut instead of an identity shortcut.
        h_in: Input height.
        w_in: Input width.
        pragmas: Pragma settings shared by all residual blocks.
        precision: Data / accumulator formats.
        stride: Stride of the first convolution (and of the projection).

    Returns:
        The paths of the written files.
    """
    fw1, fb1, fw2, fb2 = arrays[:4]
    has_proj = len(arrays) == 6
    c_in, c_mid, c_out = fw1.shape[1], fw1.shape[0], fw2.shape[0]
    data_w, data_i = precision.data
    acc_w, acc_i = precision.acc
    weights_proj = bias_proj = ""
    if has_proj:
        weights_proj = format_cpp_array(arrays[4])
        bias_proj = format_cpp_array(arrays[5])
    context = {
        "func": name,
        "NAME": name.upper(),
        "c_in": c_in,
        "c_mid": c_mid,
        "c_out": c_out,
        "h_in": h_in,
        "w_in": w_in,
        "h_out": h_in // stride,
        "w_out": w_in // stride,
        "stride": stride,
        "has_proj": has_proj,
        "data_w": data_w,
        "data_i": data_i,
        "acc_w": acc_w,
        "acc_i": acc_i,
        "conv_ii": pragmas.ii,
        "unroll_kh": pragmas.unroll_kh,
        "unroll_kw": pragmas.unroll_kw,
        "unroll_ic1": _clip_unroll(pragmas.unroll_ic, c_in),
        "unroll_ic2": _clip_unroll(pragmas.unroll_ic, c_mid),
        "partition_w_factor": pragmas.partition_w_factor,
        "partition_mid_factor": pragmas.partition_mid_factor,
        "w1": format_cpp_array(fw1),
        "b1": format_cpp_array(fb1),
        "w2": format_cpp_array(fw2),
        "b2": format_cpp_array(fb2),
        "wp": weights_proj,
        "bp": bias_proj,
    }
    written = _render_pair(output_dir, "resblock.hpp.j2", "resblock.cpp.j2", name, context)
    logger.info(
        "  [resblock] %-30s  c_in=%3d  c_mid=%3d  c_out=%3d  stride=%d  %s",
        name,
        c_in,
        c_mid,
        c_out,
        stride,
        "[proj]" if has_proj else "[id]  ",
    )
    return written


def generate_gap(
    output_dir: Path, channels: int, height: int, width: int, precision: Precision
) -> list[Path]:
    """Generate the global-average-pooling layer (``gap_layer.hpp`` / ``gap_layer.cpp``)."""
    acc_w, acc_i = precision.acc
    context = {"c": channels, "h": height, "w": width, "acc_w": acc_w, "acc_i": acc_i}
    written = _render_pair(output_dir, "gap_layer.hpp.j2", "gap_layer.cpp.j2", "gap_layer", context)
    logger.info("  [gap]      gap_layer  [%d,%d,%d] -> [%d]", channels, height, width, channels)
    return written


def generate_fc(
    output_dir: Path,
    weights: np.ndarray,
    bias: np.ndarray,
    pragmas: FCPragmas,
    precision: Precision,
) -> list[Path]:
    """Generate the fully connected layer (``fc_layer.hpp`` / ``fc_layer.cpp``).

    Args:
        output_dir: Folder that receives the files.
        weights: Weight matrix with shape ``[fc_out, fc_in]``.
        bias: Bias vector with shape ``[fc_out]``.
        pragmas: Pragma settings of the layer.
        precision: Data / accumulator formats.
    """
    fc_out, fc_in = weights.shape
    data_w, data_i = precision.data
    acc_w, acc_i = precision.acc
    context = {
        "fc_in": fc_in,
        "fc_out": fc_out,
        "data_w": data_w,
        "data_i": data_i,
        "acc_w": acc_w,
        "acc_i": acc_i,
        "fc_ii": pragmas.ii,
        "unroll_i": pragmas.unroll_i,
        "weights": format_cpp_array(weights),
        "bias": format_cpp_array(bias),
    }
    written = _render_pair(output_dir, "fc_layer.hpp.j2", "fc_layer.cpp.j2", "fc_layer", context)
    logger.info(
        "  [fc]       fc_layer  [%d,%d]  data=ap_fixed<%d,%d>", fc_out, fc_in, data_w, data_i
    )
    return written


# --------------------------------------------------------------------------- #
# Shared headers and the top-level function
# --------------------------------------------------------------------------- #
def generate_common(output_dir: Path, precision: Precision) -> list[Path]:
    """Write ``common.hpp`` with the ``data_t`` typedef."""
    path = output_dir / "common.hpp"
    write_text(path, render_template("common.hpp.j2", data_type=precision.data.cpp_type()))
    return [path]


def generate_top(
    output_dir: Path, spec: NetworkSpec, precision: Precision, *, use_dataflow: bool
) -> list[Path]:
    """Write the top-level ``resnet8.hpp`` / ``resnet8.cpp`` with AXI-stream input and output."""
    hpp = output_dir / "resnet8.hpp"
    write_text(
        hpp,
        render_template(
            "resnet8.hpp.j2",
            acc_type=precision.acc.cpp_type(),
            input_size=INPUT_SIZE,
            img_channels=IMG_CHANNELS,
            fc_out=spec.fc_out,
        ),
    )

    includes = ["resnet8.hpp", "hls_stream.h", "conv1.hpp"]
    includes += [f"{block.name}.hpp" for block in spec.blocks]
    includes += ["gap_layer.hpp", "fc_layer.hpp"]

    buffers = [f"data_t b_conv1[{spec.stem_channels}][{INPUT_SIZE}][{INPUT_SIZE}];"]
    calls = ["conv1(input, b_conv1);"]
    previous = "b_conv1"
    for block in spec.blocks:
        buffers.append(f"data_t b_{block.name}[{block.c_out}][{block.h_out}][{block.h_out}];")
        calls.append(f"{block.name}({previous}, b_{block.name});")
        previous = f"b_{block.name}"
    buffers.append(f"data_t b_gap[{spec.final_channels}];")
    buffers.append(f"data_t b_fc[{spec.fc_out}];")
    calls.append(f"gap_layer({previous}, b_gap);")
    calls.append("fc_layer(b_gap, b_fc);")

    cpp = output_dir / "resnet8.cpp"
    write_text(
        cpp,
        render_template(
            "resnet8.cpp.j2",
            includes=includes,
            buffers=buffers,
            calls=calls,
            use_dataflow=use_dataflow,
        ),
    )
    return [hpp, cpp]


# --------------------------------------------------------------------------- #
# One complete design
# --------------------------------------------------------------------------- #
def generate_design(
    output_dir: Path,
    spec: NetworkSpec,
    fused: FusedWeights,
    pragmas: PragmaConfig,
    precision: Precision,
) -> list[Path]:
    """Generate every source file of one HLS design into ``output_dir``.

    Args:
        output_dir: Target folder (created if missing).
        spec: Network shape information.
        fused: BN-fused (and quantized) parameters of every HLS layer.
        pragmas: The pragma configuration to apply.
        precision: Data / accumulator formats.

    Returns:
        The paths of all written files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    stem_w, stem_b = fused["conv1"]
    written += generate_conv(
        output_dir,
        "conv1",
        stem_w,
        stem_b,
        INPUT_SIZE,
        INPUT_SIZE,
        pragmas.conv1,
        precision,
        stride=1,
        relu=True,
    )
    for block in spec.blocks:
        written += generate_resblock(
            output_dir,
            block.name,
            fused[block.name],
            block.h_in,
            block.h_in,
            pragmas.resblock,
            precision,
            stride=block.stride,
        )
    written += generate_gap(
        output_dir, spec.final_channels, spec.final_spatial, spec.final_spatial, precision
    )
    fc_w, fc_b = fused["fc"]
    written += generate_fc(output_dir, fc_w, fc_b, pragmas.fc, precision)
    written += generate_common(output_dir, precision)
    written += generate_top(output_dir, spec, precision, use_dataflow=pragmas.use_dataflow)
    return written
