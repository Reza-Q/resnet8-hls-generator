import re

import numpy as np
import torch

from resnet8_hls.config import FixedPoint, Precision
from resnet8_hls.model import describe_network
from resnet8_hls.pragma_dse import generate_pragma_set
from resnet8_hls.tcl_export import render_tcl
from resnet8_hls.testbench import write_test_samples, write_testbench

PRECISION = Precision(data=FixedPoint(8, 4), acc=FixedPoint(8, 4))


def test_test_samples_header_matches_the_dataset_and_the_reference_model(
    tmp_path, tiny_model, image_dataset
):
    path = tmp_path / "test_samples.h"
    accuracy = write_test_samples(path, image_dataset, tiny_model, PRECISION.data, 12, 10)
    text = path.read_text()

    assert "#define TB_N       12" in text
    assert "static const float tb_input[12][3][32][32] = {" in text
    assert text.count("/* sample ") == 12
    assert text.endswith("};\n")

    labels = re.search(r"tb_label\[12\] = \{([^}]*)\}", text).group(1)
    assert [int(v) for v in labels.split(",")] == image_dataset.labels[:12]

    logits_block = text.split("tb_ref_logits[12][10] = {")[1]
    logits = np.array(re.findall(r"-?\d+\.\d+(?=f)", logits_block), dtype=np.float64).reshape(
        12, 10
    )
    with torch.no_grad():
        expected = tiny_model(image_dataset.images[:12]).numpy()
    np.testing.assert_allclose(logits, expected, atol=1e-6)

    predictions = expected.argmax(axis=1)
    assert accuracy == (predictions == np.array(image_dataset.labels[:12])).mean() * 100


def test_test_inputs_are_quantized_to_the_hls_grid(tmp_path, tiny_model, image_dataset):
    path = tmp_path / "test_samples.h"
    write_test_samples(path, image_dataset, tiny_model, PRECISION.data, 2, 10)
    input_block = path.read_text().split("tb_label")[0].split("= {", 1)[1]
    values = np.array(re.findall(r"-?\d+\.\d+(?=f)", input_block), dtype=np.float64)
    assert values.size == 2 * 3 * 32 * 32
    np.testing.assert_allclose(values * 16, np.round(values * 16), atol=1e-6)
    assert values.max() <= 7.9375 and values.min() >= -8.0


def test_write_testbench_creates_the_source_and_the_header(tmp_path, tiny_model, image_dataset):
    write_testbench(tmp_path, image_dataset, tiny_model, PRECISION.data, 4, 10)
    assert (tmp_path / "testbench" / "test_samples.h").is_file()
    source = (tmp_path / "tb_resnet8.cpp").read_text()
    assert '#include "testbench/test_samples.h"' in source
    assert 'printf("\\n-- Summary --\\n");' in source
    assert "{{" not in source


def _tcl(tiny_state_dict, **kwargs):
    spec = describe_network(tiny_state_dict)
    configs = generate_pragma_set(spec)
    blocks = [b.name for b in spec.blocks]
    return render_tcl(
        configs,
        blocks,
        PRECISION,
        kwargs.get("part", "xc7z020clg400-2"),
        kwargs.get("clock_ns", 10),
    )


def test_tcl_runs_the_full_flow_only_for_the_first_design(tiny_state_dict):
    tcl = _tcl(tiny_state_dict)
    assert tcl.count("open_project ") == 8
    assert "open_project resnet8_P0" in tcl and "open_project resnet8_P7" in tcl
    assert tcl.count("csim_design") == 1
    assert tcl.count("cosim_design") == 1
    assert tcl.count("csynth_design") == 8
    assert tcl.count("add_files -tb") == 2
    assert tcl.endswith("close_project\n")


def test_tcl_lists_every_source_file(tiny_state_dict):
    tcl = _tcl(tiny_state_dict)
    for name in ("conv1", "layer1_block0", "layer3_block1", "gap_layer", "fc_layer", "resnet8"):
        assert f"add_files $SRC/{name}.cpp" in tcl
        assert f"add_files $SRC/{name}.hpp" in tcl


def test_tcl_part_and_clock_are_configurable(tiny_state_dict):
    default = _tcl(tiny_state_dict)
    assert "set_part {xc7z020clg400-2}" in default
    assert "create_clock -period 10 -name default" in default
    custom = _tcl(tiny_state_dict, part="xc7z045ffg900-2", clock_ns=8.5)
    assert "set_part {xc7z045ffg900-2}" in custom
    assert "create_clock -period 8.5 -name default" in custom
    assert "# part   : xc7z045ffg900-2   clock: 8.5 ns" in custom
