"""Vitis HLS batch script (``create_project.tcl``) that builds every design."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .codegen import write_text
from .config import Precision
from .pragma_dse import PragmaConfig
from .testbench import SAMPLES_FILENAME, TB_SOURCE_FILENAME, TESTBENCH_DIRNAME

PROJECT_PREFIX = "resnet8_P"
"""Vitis HLS project name is ``resnet8_P<index>``."""


def render_tcl(
    configs: Sequence[PragmaConfig],
    block_names: Sequence[str],
    precision: Precision,
    part: str,
    clock_ns: float,
) -> str:
    """Render the Tcl script text.

    Design 0 runs C simulation, synthesis and co-simulation (it owns the testbench); all
    other designs run C synthesis only.

    Args:
        configs: The pragma configurations, one HLS project each.
        block_names: HLS names of the residual blocks (their sources are added to each project).
        precision: Data / accumulator formats (documented in the script header).
        part: Xilinx part number.
        clock_ns: Target clock period in nanoseconds.
    """
    clock = f"{clock_ns:g}"
    sources = ["conv1", *block_names, "gap_layer", "fc_layer", "resnet8"]
    count = len(configs)
    lines = [
        f"# create_project.tcl -- Vitis HLS batch: {count} pragma configs",
        f"# data_t : {precision.data.cpp_type()}",
        f"# acc_t  : {precision.acc.cpp_type()}",
        f"# part   : {part}   clock: {clock} ns",
        f"# P0: csim + csynth + cosim  |  P1-P{count - 1}: csynth only",
        "# Run from this directory:   vitis_hls -f create_project.tcl",
        "",
        "set ROOT [file dirname [info script]]",
        "",
    ]
    for idx, cfg in enumerate(configs):
        is_first = idx == 0
        flow = "csim+csynth+cosim" if is_first else "csynth only"
        lines += [
            f"# --- P{idx}: {cfg.label} ({flow}) ---",
            f"open_project {PROJECT_PREFIX}{idx}",
            "set_top resnet8",
            f"set SRC $ROOT/pragma_{idx}",
            "",
            "add_files $SRC/common.hpp",
            "add_files $SRC/resnet8.hpp",
        ]
        for source in sources:
            lines.append(f"add_files $SRC/{source}.hpp")
            lines.append(f"add_files $SRC/{source}.cpp")
        if is_first:
            lines += [
                f"add_files -tb $SRC/{TB_SOURCE_FILENAME}",
                f"add_files -tb $SRC/{TESTBENCH_DIRNAME}/{SAMPLES_FILENAME}",
            ]
        lines += [
            "",
            'open_solution "solution1" -flow_target vivado',
            f"set_part {{{part}}}",
            f"create_clock -period {clock} -name default",
            "",
            "config_compile -pipeline_style flp -unsafe_math_optimizations=false",
            "config_schedule -relax_ii_for_timing=true",
        ]
        lines += ["csim_design", "csynth_design", "cosim_design"] if is_first else ["csynth_design"]
        lines += ["close_project", ""]
    return "\n".join(lines)


def write_tcl(
    path: Path,
    configs: Sequence[PragmaConfig],
    block_names: Sequence[str],
    precision: Precision,
    part: str,
    clock_ns: float,
) -> Path:
    """Write the Tcl script to ``path`` and return the path."""
    write_text(path, render_tcl(configs, block_names, precision, part, clock_ns))
    return path
