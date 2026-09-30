from pathlib import Path

import pytest

from resnet8_hls import cli
from resnet8_hls.config import FixedPoint


def _parse(*argv):
    return cli.config_from_args(cli.build_parser().parse_args(list(argv)))


def test_defaults_match_the_original_notebook():
    config = _parse("models/final_model_5pct_rate.pth")
    assert config.output_dir == Path("hls_output/final_model_5pct_rate")
    assert config.acc_bits == FixedPoint(8, 4)
    assert config.data_bits is None
    assert config.enable_qat is True
    assert (config.qat.epochs, config.qat.lr, config.qat.lr_min) == (10, 1e-3, 1e-5)
    assert (config.qat.batch_size, config.qat.weight_decay, config.qat.grad_clip) == (
        128,
        1e-4,
        1.0,
    )
    assert config.num_test_samples == 100
    assert config.part == "xc7z020clg400-2"
    assert config.clock_ns == 10
    assert config.weights_only is False
    assert config.download is True


def test_options_are_translated():
    config = _parse(
        "model.pth",
        "-o", "somewhere",
        "--no-qat",
        "--acc-bits", "16", "8",
        "--data-bits", "12", "6",
        "--epochs", "3",
        "--part", "xc7z045ffg900-2",
        "--clock-ns", "8.5",
        "--weights-only",
        "--no-download",
        "--seed", "5",
    )  # fmt: skip
    assert config.output_dir == Path("somewhere")
    assert config.enable_qat is False
    assert config.acc_bits == FixedPoint(16, 8)
    assert config.data_bits == FixedPoint(12, 6)
    assert config.qat.epochs == 3
    assert (config.part, config.clock_ns, config.seed) == ("xc7z045ffg900-2", 8.5, 5)
    assert config.weights_only is True
    assert config.download is False


def test_main_runs_the_pipeline_with_the_parsed_configuration(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "run_pipeline", lambda config: captured.setdefault("config", config))
    assert cli.main(["model.pth", "--no-qat", "-q"]) == 0
    assert captured["config"].enable_qat is False


def test_main_reports_a_missing_checkpoint(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main([str(tmp_path / "missing.pth"), "-q"]) == 1
    assert "Checkpoint not found" in capsys.readouterr().err


def test_main_reports_invalid_bit_widths(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["model.pth", "--acc-bits", "4", "9", "-q"]) == 1
    assert "acc_bits" in capsys.readouterr().err


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0
    assert "--no-qat" in capsys.readouterr().out
