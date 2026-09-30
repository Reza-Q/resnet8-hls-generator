import numpy as np
import pytest
import torch

from resnet8_hls.config import FixedPoint, Precision, QATConfig
from resnet8_hls.data import make_loader
from resnet8_hls.model import hls_layer_names
from resnet8_hls.qat import (
    QATResNet8,
    evaluate_accuracy,
    export_qat_weights,
    ste_quantize,
    train_qat,
    uniform_bit_config,
)
from resnet8_hls.quantization import quantize_hls

from .conftest import RandomImageDataset

PRECISION = Precision(data=FixedPoint(8, 4), acc=FixedPoint(8, 4))


def test_ste_forward_matches_the_hls_quantizer():
    x = torch.randn(1000) * 3
    expected = torch.from_numpy(quantize_hls(x.numpy(), 8, 4)).float()
    torch.testing.assert_close(ste_quantize(x, 8, 4), expected)


def test_ste_backward_is_the_identity():
    x = torch.randn(50, requires_grad=True)
    ste_quantize(x, 8, 4).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones(50))


def _qat_model(model, precision=PRECISION):
    bit_cfg = uniform_bit_config(hls_layer_names(model.state_dict()), precision)
    return QATResNet8(model, bit_cfg, precision.data)


def test_wide_fixed_point_formats_reproduce_the_float_model(tiny_model):
    wide = Precision(data=FixedPoint(32, 16), acc=FixedPoint(32, 16))
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        difference = (_qat_model(tiny_model, wide)(x) - tiny_model(x)).abs().max()
    assert difference < 1e-2


def test_batch_norm_stays_in_inference_mode_during_training(tiny_model):
    model = _qat_model(tiny_model)
    model.train()
    assert model.training
    assert not model.base.bn1.training
    assert not model.base.layer2[0].shortcut[1].training


def test_qat_model_does_not_modify_the_base_model(tiny_model):
    before = tiny_model.linear.weight.detach().clone()
    model = _qat_model(tiny_model)
    model.base.linear.weight.data.add_(1.0)
    assert torch.equal(tiny_model.linear.weight, before)


def test_train_qat_updates_weights_and_reports_validation_accuracy(tiny_model):
    dataset = RandomImageDataset(size=32)
    config = QATConfig(epochs=2, lr=0.1, batch_size=16)
    model, best_val = train_qat(
        tiny_model,
        PRECISION,
        make_loader(dataset, 16, shuffle=True, drop_last=True),
        make_loader(dataset, 16, shuffle=False),
        config,
    )
    assert isinstance(model, QATResNet8)
    assert 0.0 <= best_val <= 100.0
    assert not torch.equal(model.base.linear.weight, tiny_model.linear.weight)
    assert 0.0 <= evaluate_accuracy(model, make_loader(dataset, 16, shuffle=False)) <= 100.0


def test_train_qat_rejects_an_empty_training_loader(tiny_model):
    dataset = RandomImageDataset(size=4)
    with pytest.raises(ValueError, match="no batches"):
        train_qat(
            tiny_model,
            PRECISION,
            make_loader(dataset, 16, shuffle=True, drop_last=True),
            make_loader(dataset, 16, shuffle=False),
            QATConfig(epochs=1),
        )


def test_exported_weights_lie_on_the_hls_grid(tiny_model):
    exported = export_qat_weights(_qat_model(tiny_model))
    for name in ("conv1.weight", "layer1.0.conv1.weight", "layer3.1.conv2.weight", "linear.weight"):
        assert exported[name].dtype == np.float32
        np.testing.assert_array_equal(exported[name] * 16, np.round(exported[name] * 16))
    # Batch-norm parameters are exported untouched.
    np.testing.assert_array_equal(exported["bn1.weight"], tiny_model.bn1.weight.detach().numpy())
