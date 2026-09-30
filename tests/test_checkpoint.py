import numpy as np
import pytest
import torch

from resnet8_hls.checkpoint import (
    dequantize_state_dict,
    describe_state_dict,
    inject_identity_batchnorm,
    is_quantized_state_dict,
    load_state_dict_numpy,
    unpack_linear,
)


def test_plain_state_dict_is_loaded_as_numpy(tiny_model, tiny_checkpoint):
    loaded = load_state_dict_numpy(tiny_checkpoint)
    assert set(loaded) == set(tiny_model.state_dict())
    assert all(isinstance(v, np.ndarray) for v in loaded.values())
    np.testing.assert_array_equal(loaded["conv1.weight"], tiny_model.conv1.weight.detach().numpy())


def test_weights_only_loading_works_for_plain_state_dicts(tiny_checkpoint):
    assert "linear.weight" in load_state_dict_numpy(tiny_checkpoint, weights_only=True)


@pytest.mark.parametrize("key", ["state_dict", "model_state", "model_state_dict", "net", "params"])
def test_wrapped_state_dicts_are_found(tmp_path, tiny_model, key):
    path = tmp_path / "wrapped.pth"
    torch.save({key: tiny_model.state_dict(), "epoch": 3}, path)
    assert "conv1.weight" in load_state_dict_numpy(path)


def test_pickled_module_is_supported(tmp_path, tiny_model):
    path = tmp_path / "module.pth"
    torch.save(tiny_model, path)
    assert "conv1.weight" in load_state_dict_numpy(path, weights_only=False)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_state_dict_numpy(tmp_path / "missing.pth")


def test_dict_without_a_state_dict_raises_key_error(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"epoch": 1, "note": "no tensors here"}, path)
    with pytest.raises(KeyError, match="Cannot find state dict"):
        load_state_dict_numpy(path)


def test_identity_batchnorm_is_injected_for_every_conv():
    sd = {
        "conv1.weight": torch.randn(4, 3, 3, 3),
        "layer1.0.conv1.weight": torch.randn(4, 4, 3, 3),
        "layer1.0.conv2.weight": torch.randn(4, 4, 3, 3),
        "layer1.0.shortcut.0.weight": torch.randn(4, 4, 1, 1),
    }
    out = inject_identity_batchnorm(sd)
    for prefix in ("bn1", "layer1.0.bn1", "layer1.0.bn2", "layer1.0.shortcut.1"):
        assert torch.equal(out[f"{prefix}.weight"], torch.ones(4))
        assert torch.equal(out[f"{prefix}.bias"], torch.zeros(4))
        assert torch.equal(out[f"{prefix}.running_mean"], torch.zeros(4))
        assert torch.equal(out[f"{prefix}.running_var"], torch.ones(4))


def test_existing_batchnorm_parameters_are_not_overwritten(tiny_model):
    sd = dict(tiny_model.state_dict())
    out = inject_identity_batchnorm(sd)
    assert torch.equal(out["bn1.weight"], sd["bn1.weight"])


def test_quantized_tensors_are_detected_and_dequantized():
    quantized = torch.quantize_per_tensor(torch.tensor([0.5, -1.0]), 0.25, 0, torch.qint8)
    sd = {
        "conv1.weight": quantized,
        "conv1.scale": torch.tensor(0.25),
        "conv1.zero_point": torch.tensor(0),
        "x.activation_post_process.min_val": torch.tensor(0.0),
    }
    assert is_quantized_state_dict(sd)
    result = dequantize_state_dict(sd)
    assert set(result) == {"conv1.weight"}
    assert not result["conv1.weight"].is_quantized
    torch.testing.assert_close(result["conv1.weight"], torch.tensor([0.5, -1.0]))


def test_packed_linear_parameters_are_unpacked():
    weight = torch.quantize_per_tensor(torch.randn(3, 4), 0.1, 0, torch.qint8)
    bias = torch.zeros(3)
    sd = {"linear._packed_params._packed_params": (weight, bias), "other": torch.ones(1)}
    out = unpack_linear(sd)
    assert set(out) == {"linear.weight", "linear.bias", "other"}
    assert not out["linear.weight"].is_quantized


def test_describe_state_dict_hides_bookkeeping_entries():
    text = describe_state_dict(
        {"a.weight": np.zeros((2, 3)), "bn.num_batches_tracked": np.zeros(())}
    )
    assert "a.weight" in text
    assert "num_batches_tracked" not in text
