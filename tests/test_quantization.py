import numpy as np

from resnet8_hls.config import FixedPoint, Precision
from resnet8_hls.quantization import (
    format_cpp_array,
    format_quantization_report,
    quantization_report,
    quantize_fused_weights,
    quantize_hls,
    quantize_round,
)


def test_quantize_hls_floors_toward_negative_infinity():
    # ap_fixed<8,4>: resolution 1/16, range [-8, 7.9375]
    result = quantize_hls(np.array([0.99, -0.01, 0.0625]), 8, 4)
    np.testing.assert_array_equal(result, [0.9375, -0.0625, 0.0625])
    assert result.dtype == np.float64


def test_quantize_hls_saturates():
    result = quantize_hls(np.array([100.0, -100.0]), 8, 4)
    np.testing.assert_array_equal(result, [7.9375, -8.0])


def test_quantize_round_rounds_to_nearest_and_returns_float32():
    result = quantize_round(np.array([0.99, -0.01]), 8, 4)
    np.testing.assert_array_equal(result, [1.0, 0.0])
    assert result.dtype == np.float32


def test_quantize_hls_is_idempotent_and_on_grid():
    values = np.random.default_rng(0).normal(scale=3.0, size=1000)
    once = quantize_hls(values, 8, 4)
    np.testing.assert_array_equal(quantize_hls(once, 8, 4), once)
    np.testing.assert_array_equal(once * 16, np.round(once * 16))


def test_format_cpp_array_one_dimensional():
    assert format_cpp_array(np.array([1.0, -0.5])) == "{1.0000000000f, -0.5000000000f}"


def test_format_cpp_array_two_dimensional_layout():
    text = format_cpp_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert text == ("{\n    {1.0000000000f, 2.0000000000f},\n    {3.0000000000f, 4.0000000000f}\n}")


def test_format_cpp_array_respects_depth():
    assert format_cpp_array(np.array([1.0]), depth=2).startswith(" " * 8 + "{")


def test_quantization_report_flags_low_sqnr_and_skips_gap_and_zeros():
    fused = {
        "conv1": (np.full((4,), 0.01), np.zeros(4)),  # 0.01 rounds to 0 -> SQNR 0 dB
        "fc": (np.full((4,), 0.5),),  # exactly on the grid -> SQNR capped at 99 dB
        "gap": (np.ones(3),),
    }
    report = quantization_report(fused, ["conv1", "gap", "fc"], FixedPoint(8, 4))
    assert set(report) == {"conv1", "fc"}
    assert report["conv1"]["sqnr_db"] < 20.0
    assert report["fc"]["sqnr_db"] == 99.0
    rows = {
        line.split()[0]: line
        for line in format_quantization_report(report, FixedPoint(8, 4)).splitlines()
        if line.startswith("  ")
    }
    assert "LOW SQNR" in rows["conv1"]
    assert "LOW SQNR" not in rows["fc"]


def test_quantize_fused_weights_uses_data_format_for_weights_and_acc_format_for_biases():
    precision = Precision(data=FixedPoint(8, 4), acc=FixedPoint(16, 8))
    fused = {"conv1": (np.array([0.123456]), np.array([0.123456]))}
    result = quantize_fused_weights(fused, precision)
    weight, bias = result["conv1"]
    assert weight[0] == np.float32(0.0625)  # floor(0.123456 * 16) / 16
    assert bias[0] == np.float32(0.12109375)  # floor(0.123456 * 256) / 256
    assert weight.dtype == bias.dtype == np.float32
    assert fused["conv1"][0][0] == 0.123456  # input is not modified
