import filecmp
from pathlib import Path

import pytest

from resnet8_hls import FixedPoint, PipelineConfig, QATConfig, run_pipeline

from .conftest import RandomImageDataset

FAST_QAT = QATConfig(epochs=1, batch_size=16, lr=0.05)


def _config(tmp_path, checkpoint, name="out", **overrides):
    options = {
        "checkpoint": checkpoint,
        "output_dir": tmp_path / name,
        "num_test_samples": 8,
        "qat": FAST_QAT,
    }
    options.update(overrides)
    return PipelineConfig(**options)


def _run(config):
    return run_pipeline(
        config,
        train_dataset=RandomImageDataset(size=32, seed=1),
        test_dataset=RandomImageDataset(size=16, seed=2),
    )


def _same_tree(a: Path, b: Path) -> bool:
    comparison = filecmp.dircmp(a, b)
    if comparison.left_only or comparison.right_only or comparison.diff_files:
        return False
    return all(_same_tree(a / sub, b / sub) for sub in comparison.subdirs)


def test_ptq_pipeline_produces_the_expected_layout(tmp_path, tiny_checkpoint):
    result = _run(_config(tmp_path, tiny_checkpoint, enable_qat=False))
    root = tmp_path / "out"
    assert result.output_dir == root
    assert sorted(p.name for p in root.iterdir()) == sorted(
        [f"pragma_{i}" for i in range(8)] + ["create_project.tcl"]
    )
    assert result.tcl_path == root / "create_project.tcl"
    assert (root / "pragma_0" / "tb_resnet8.cpp").is_file()
    assert (root / "pragma_0" / "testbench" / "test_samples.h").is_file()
    for i in range(1, 8):
        assert not (root / f"pragma_{i}" / "testbench").exists()
        assert (root / f"pragma_{i}" / "resnet8.cpp").is_file()
    assert result.best_val_accuracy is None
    assert 0.0 <= result.reference_accuracy <= 100.0
    assert len(result.pragma_configs) == 8
    assert "conv1" in result.quantization_report


def test_qat_pipeline_reports_validation_accuracy(tmp_path, tiny_checkpoint):
    result = _run(_config(tmp_path, tiny_checkpoint, enable_qat=True))
    assert result.best_val_accuracy is not None
    assert (tmp_path / "out" / "pragma_7" / "layer3_block0.cpp").is_file()


def test_qat_and_ptq_write_different_weights(tmp_path, tiny_checkpoint):
    _run(_config(tmp_path, tiny_checkpoint, "qat", enable_qat=True))
    _run(_config(tmp_path, tiny_checkpoint, "ptq", enable_qat=False))
    qat = (tmp_path / "qat" / "pragma_0" / "conv1.cpp").read_text()
    ptq = (tmp_path / "ptq" / "pragma_0" / "conv1.cpp").read_text()
    assert qat != ptq


@pytest.mark.parametrize("enable_qat", [False, True])
def test_runs_are_reproducible_and_do_not_share_state(tmp_path, tiny_checkpoint, enable_qat):
    _run(_config(tmp_path, tiny_checkpoint, "first", enable_qat=enable_qat, seed=3))
    _run(_config(tmp_path, tiny_checkpoint, "second", enable_qat=enable_qat, seed=3))
    assert _same_tree(tmp_path / "first", tmp_path / "second")


def test_custom_precision_reaches_the_generated_code_and_tcl(tmp_path, tiny_checkpoint):
    config = _config(
        tmp_path,
        tiny_checkpoint,
        enable_qat=False,
        acc_bits=FixedPoint(16, 8),
        data_bits=FixedPoint(12, 6),
        part="xc7z045ffg900-2",
        clock_ns=8,
    )
    _run(config)
    root = tmp_path / "out"
    assert "ap_fixed<12,6,AP_TRN,AP_SAT> data_t" in (root / "pragma_3" / "common.hpp").read_text()
    tcl = (root / "create_project.tcl").read_text()
    assert "# acc_t  : ap_fixed<16,8,AP_TRN,AP_SAT>" in tcl
    assert "set_part {xc7z045ffg900-2}" in tcl


def test_too_many_test_samples_is_rejected(tmp_path, tiny_checkpoint):
    with pytest.raises(ValueError, match="num_test_samples"):
        _run(_config(tmp_path, tiny_checkpoint, enable_qat=False, num_test_samples=999))


def test_missing_checkpoint_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        _run(_config(tmp_path, tmp_path / "missing.pth", enable_qat=False))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"acc_bits": FixedPoint(8, 9)}, "acc_bits"),
        ({"data_bits": FixedPoint(8, 0)}, "data_bits"),
        ({"num_test_samples": 0}, "num_test_samples"),
        ({"clock_ns": 0}, "clock_ns"),
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig(checkpoint=tmp_path / "x.pth", output_dir=tmp_path / "out", **overrides)


def test_data_bits_default_to_the_accumulator_format(tmp_path):
    config = PipelineConfig(
        checkpoint=tmp_path / "x.pth", output_dir=tmp_path, acc_bits=FixedPoint(10, 5)
    )
    assert config.precision.data == config.precision.acc == FixedPoint(10, 5)
    assert config.precision.data.cpp_type() == "ap_fixed<10,5,AP_TRN,AP_SAT>"
    assert config.precision.data.frac_bits == 5
