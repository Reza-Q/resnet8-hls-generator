import dataclasses

import pytest

from resnet8_hls.codegen import generate_design
from resnet8_hls.config import FixedPoint, Precision
from resnet8_hls.model import describe_network, get_fused_weights
from resnet8_hls.pragma_dse import generate_pragma_set
from resnet8_hls.quantization import quantize_fused_weights

PRECISION = Precision(data=FixedPoint(8, 4), acc=FixedPoint(8, 4))
BLOCKS = ["layer1_block0", "layer1_block1", "layer2_block0", "layer3_block0", "layer3_block1"]


@pytest.fixture
def design_inputs(tiny_state_dict):
    spec = describe_network(tiny_state_dict)
    fused = quantize_fused_weights(get_fused_weights(tiny_state_dict), PRECISION)
    return spec, fused, generate_pragma_set(spec)


def test_a_design_contains_every_expected_file(tmp_path, design_inputs):
    spec, fused, pragma_set = design_inputs
    written = generate_design(tmp_path / "d", spec, fused, pragma_set[0], PRECISION)
    names = {p.name for p in written}
    expected = {"common.hpp", "resnet8.hpp", "resnet8.cpp", "conv1.hpp", "conv1.cpp"}
    expected |= {f"{b}.{ext}" for b in BLOCKS for ext in ("hpp", "cpp")}
    expected |= {f"{n}.{ext}" for n in ("gap_layer", "fc_layer") for ext in ("hpp", "cpp")}
    assert names == expected
    assert {p.name for p in (tmp_path / "d").iterdir()} == expected


def test_output_has_no_template_leftovers_and_uses_unix_line_endings(tmp_path, design_inputs):
    spec, fused, pragma_set = design_inputs
    for path in generate_design(tmp_path, spec, fused, pragma_set[7], PRECISION):
        data = path.read_bytes()
        text = data.decode("utf-8")
        for marker in ("{{", "{%", "{#", "%}"):
            assert marker not in text, f"{path.name} contains {marker!r}"
        assert b"\r" not in data, path.name
        assert data.endswith(b"\n"), path.name


def test_data_type_and_top_level_files(tmp_path, design_inputs):
    spec, fused, pragma_set = design_inputs
    generate_design(tmp_path, spec, fused, pragma_set[0], PRECISION)
    assert "typedef ap_fixed<8,4,AP_TRN,AP_SAT> data_t;" in (tmp_path / "common.hpp").read_text()
    header = (tmp_path / "resnet8.hpp").read_text()
    assert "#define NET_C_OUT 10" in header
    assert "typedef ap_fixed<8,4,AP_TRN,AP_SAT> acc_t;" in header

    top = (tmp_path / "resnet8.cpp").read_text()
    lines = [line.strip() for line in top.splitlines()]
    start = lines.index("read_input(input_stream, input);") + 1
    calls = lines[start : lines.index("write_output(b_fc, output_stream);")]
    assert calls == [
        "conv1(input, b_conv1);",
        "layer1_block0(b_conv1, b_layer1_block0);",
        "layer1_block1(b_layer1_block0, b_layer1_block1);",
        "layer2_block0(b_layer1_block1, b_layer2_block0);",
        "layer3_block0(b_layer2_block0, b_layer3_block0);",
        "layer3_block1(b_layer3_block0, b_layer3_block1);",
        "gap_layer(b_layer3_block1, b_gap);",
        "fc_layer(b_gap, b_fc);",
    ]
    assert "data_t b_layer2_block0[8][16][16];" in top
    assert "data_t b_gap[16];" in top and "data_t b_fc[10];" in top
    assert "#pragma HLS DATAFLOW" not in top


def test_dataflow_pragma_is_optional(tmp_path, design_inputs):
    spec, fused, pragma_set = design_inputs
    with_df = dataclasses.replace(pragma_set[0], use_dataflow=True)
    generate_design(tmp_path, spec, fused, with_df, PRECISION)
    assert "#pragma HLS DATAFLOW" in (tmp_path / "resnet8.cpp").read_text()


def test_identity_and_projection_shortcuts(tmp_path, design_inputs):
    spec, fused, pragma_set = design_inputs
    generate_design(tmp_path, spec, fused, pragma_set[0], PRECISION)
    identity = (tmp_path / "layer1_block0.cpp").read_text()
    projection = (tmp_path / "layer2_block0.cpp").read_text()
    assert "LOOP_PROJ" not in identity
    assert "input[oc][oh][ow]" in identity
    assert "LOOP_PROJ" in projection
    assert "layer2_block0_wp[8][4]" in projection
    assert "input[ic][oh * 2][ow * 2]" in projection


def test_pragmas_and_declarations_follow_the_configuration(tmp_path, design_inputs):
    spec, fused, pragma_set = design_inputs
    p0, p7 = tmp_path / "p0", tmp_path / "p7"
    generate_design(p0, spec, fused, pragma_set[0], PRECISION)
    generate_design(p7, spec, fused, pragma_set[7], PRECISION)

    stem = (p0 / "conv1.cpp").read_text()
    assert "#pragma HLS PIPELINE II=8" in stem
    assert "static const CONV1_w_t conv1_w[4][3][3][3] =" in stem
    assert "(acc > (CONV1_acc_t)0)" in stem  # the stem applies ReLU
    assert "ARRAY_PARTITION" not in (p0 / "layer3_block0.cpp").read_text()

    block = (p7 / "layer3_block0.cpp").read_text()
    assert "#pragma HLS PIPELINE II=4" in block
    assert "ARRAY_PARTITION variable=layer3_block0_w1 dim=4 complete" in block
    assert "#pragma HLS UNROLL factor=2" in block
    assert "static const LAYER3_BLOCK0_local_t layer3_block0_w1[16][8][3][3] =" in block


def test_designs_are_written_only_to_their_own_directory(tmp_path, design_inputs):
    spec, fused, pragma_set = design_inputs
    generate_design(tmp_path / "a", spec, fused, pragma_set[0], PRECISION)
    generate_design(tmp_path / "b", spec, fused, pragma_set[2], PRECISION)
    assert "II=8" in (tmp_path / "a" / "conv1.cpp").read_text()
    assert "II=2" in (tmp_path / "b" / "conv1.cpp").read_text()


def test_precision_is_reflected_in_the_generated_types(tmp_path, tiny_state_dict):
    precision = Precision(data=FixedPoint(12, 6), acc=FixedPoint(20, 9))
    spec = describe_network(tiny_state_dict)
    fused = quantize_fused_weights(get_fused_weights(tiny_state_dict), precision)
    generate_design(tmp_path, spec, fused, generate_pragma_set(spec)[0], precision)
    assert "ap_fixed<12,6,AP_TRN,AP_SAT> data_t" in (tmp_path / "common.hpp").read_text()
    assert "ap_fixed<20,9,AP_TRN,AP_SAT> acc_t" in (tmp_path / "resnet8.hpp").read_text()
    assert "typedef ap_fixed<12,6> CONV1_w_t;" in (tmp_path / "conv1.cpp").read_text()
    assert (
        "typedef ap_fixed<20,9,AP_TRN,AP_SAT> gap_acc_t;"
        in (tmp_path / "gap_layer.cpp").read_text()
    )
