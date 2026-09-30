"""Quantization-aware training (QAT) with a straight-through estimator (STE).

The fake-quantization arithmetic is floor plus saturation, i.e. exactly the behaviour of
``ap_fixed<W,I,AP_TRN,AP_SAT>`` in the generated hardware.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .config import FixedPoint, Precision, QATConfig
from .model import STAGES, hls_layer_names
from .quantization import quantize_hls

logger = logging.getLogger(__name__)

BitWidthConfig = Mapping[str, Precision]
"""HLS layer name -> fixed-point formats used by that layer."""


def uniform_bit_config(layers: Iterable[str], precision: Precision) -> dict[str, Precision]:
    """Give every layer the same precision.

    The per-layer mapping exists so that mixed-precision experiments only need to change
    this one dictionary.
    """
    return dict.fromkeys(layers, precision)


class _STEFloorSat(torch.autograd.Function):
    """Floor + saturate in the forward pass, identity gradient in the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, total_bits: int, int_bits: int) -> torch.Tensor:  # type: ignore[override]
        scale = 2.0 ** (total_bits - int_bits)
        q_max = 2 ** (total_bits - 1) - 1
        q_min = -(2 ** (total_bits - 1))
        return x.mul(scale).floor_().clamp_(q_min, q_max).div_(scale)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return grad, None, None


def ste_quantize(x: torch.Tensor, total_bits: int, int_bits: int) -> torch.Tensor:
    """Fake-quantize ``x`` to ``ap_fixed<total_bits,int_bits,AP_TRN,AP_SAT>`` (STE gradient)."""
    return _STEFloorSat.apply(x, total_bits, int_bits)


class QATResNet8(nn.Module):
    """A ResNet with STE fake quantization of weights and activations.

    * Convolution and linear weights are quantized to the layer's ``data`` format.
    * Activations (network input and every ReLU output) are quantized to ``data_bits``.
    * Batch norm always runs in inference mode with frozen statistics, even while the
      model is in training mode. Biases are not fake-quantized; the generated C++ casts
      them to the accumulator type when the design is compiled.

    Args:
        base_model: Float model to fine-tune. It is deep-copied.
        bit_cfg: Per-layer formats, keyed by HLS layer name (see :func:`uniform_bit_config`).
        data_bits: Activation format.
    """

    def __init__(self, base_model: nn.Module, bit_cfg: BitWidthConfig, data_bits: FixedPoint):
        super().__init__()
        self.base = copy.deepcopy(base_model)
        self.bit_cfg = dict(bit_cfg)
        self.data_bits = data_bits
        self._freeze_batch_norm()

    def train(self, mode: bool = True) -> QATResNet8:
        """Switch the training mode while keeping every batch-norm layer in inference mode."""
        super().train(mode)
        self._freeze_batch_norm()
        return self

    def _freeze_batch_norm(self) -> None:
        for module in self.base.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def _quantize_weight(self, key: str, weight: torch.Tensor) -> torch.Tensor:
        total_bits, int_bits = self.bit_cfg[key].data
        return ste_quantize(weight, total_bits, int_bits)

    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        total_bits, int_bits = self.data_bits
        return ste_quantize(x, total_bits, int_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.base
        out = self._quantize_activation(x)
        out = F.conv2d(
            out, self._quantize_weight("conv1", net.conv1.weight), None, stride=1, padding=1
        )
        out = self._quantize_activation(F.relu(net.bn1(out)))
        for stage in STAGES:
            for block_index, block in enumerate(getattr(net, f"layer{stage}")):
                key = f"layer{stage}_block{block_index}"
                identity = out
                res = F.conv2d(
                    out,
                    self._quantize_weight(key, block.conv1.weight),
                    None,
                    stride=block.conv1.stride[0],
                    padding=1,
                )
                res = self._quantize_activation(F.relu(block.bn1(res)))
                res = F.conv2d(res, self._quantize_weight(key, block.conv2.weight), None, padding=1)
                res = block.bn2(res)
                if len(block.shortcut) > 0:
                    identity = F.conv2d(
                        out,
                        self._quantize_weight(key, block.shortcut[0].weight),
                        None,
                        stride=block.shortcut[0].stride[0],
                    )
                    identity = block.shortcut[1](identity)
                out = self._quantize_activation(F.relu(res + identity))
        out = F.avg_pool2d(out, out.size()[2])
        out = out.view(out.size(0), -1)
        return F.linear(out, self._quantize_weight("fc", net.linear.weight), net.linear.bias)


def export_qat_weights(qat_model: QATResNet8) -> dict[str, np.ndarray]:
    """Snap the trained weights to the HLS grid (floor + saturate) and return a state dict.

    The stem, the two convolutions of every residual block and the classifier are snapped.
    Projection-shortcut weights and all batch-norm parameters are exported unchanged (they
    are quantized by the C++ typedefs when the design is compiled).
    """
    new_sd = {k: v.detach().cpu().numpy() for k, v in qat_model.base.state_dict().items()}

    def snap(key: str, arr: np.ndarray) -> np.ndarray:
        total_bits, int_bits = qat_model.bit_cfg[key].data
        return quantize_hls(arr, total_bits, int_bits).astype(np.float32)

    new_sd["conv1.weight"] = snap("conv1", new_sd["conv1.weight"])
    for stage in STAGES:
        block = 0
        while f"layer{stage}.{block}.conv1.weight" in new_sd:
            key = f"layer{stage}_block{block}"
            for conv in ("conv1", "conv2"):
                name = f"layer{stage}.{block}.{conv}.weight"
                new_sd[name] = snap(key, new_sd[name])
            block += 1
    new_sd["linear.weight"] = snap("fc", new_sd["linear.weight"])
    return new_sd


def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """Return the top-1 accuracy of ``model`` on ``loader`` in percent."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            correct += model(images).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return correct / total * 100


def train_qat(
    base_model: nn.Module,
    precision: Precision,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: QATConfig,
    *,
    bit_cfg: BitWidthConfig | None = None,
) -> tuple[QATResNet8, float]:
    """Fine-tune ``base_model`` with fake quantization.

    Args:
        base_model: Trained float model.
        precision: Data / accumulator formats (``data`` is also used for activations).
        train_loader: Training data.
        test_loader: Data used for the per-epoch validation accuracy.
        config: Hyper-parameters (SGD with momentum, cosine learning-rate schedule and
            gradient-norm clipping).
        bit_cfg: Optional per-layer override; defaults to ``precision`` for every layer.

    Returns:
        The trained QAT model (its final weights, not the best epoch) and the best
        validation accuracy seen (percent).
    """
    if len(train_loader) == 0:
        raise ValueError("The training loader yields no batches (dataset smaller than one batch?)")
    if bit_cfg is None:
        bit_cfg = uniform_bit_config(hls_layer_names(base_model.state_dict()), precision)
    qat_model = QATResNet8(base_model, bit_cfg, precision.data)

    optimizer = torch.optim.SGD(
        qat_model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr_min
    )
    criterion = nn.CrossEntropyLoss()

    logger.info(
        "[QAT] epochs=%d lr=%s->%s batch=%d weight->ap_fixed<%d,%d> bias->ap_fixed<%d,%d>",
        config.epochs,
        config.lr,
        config.lr_min,
        config.batch_size,
        *precision.data,
        *precision.acc,
    )

    best_val = 0.0
    for epoch in range(config.epochs):
        qat_model.train()
        total_loss = 0.0
        correct = total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = qat_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += images.size(0)
        scheduler.step()
        val_acc = evaluate_accuracy(qat_model, test_loader)
        best_val = max(best_val, val_acc)
        logger.info(
            "  ep %2d/%d  loss=%.4f  train=%.2f%%  val=%.2f%%  lr=%.2e",
            epoch + 1,
            config.epochs,
            total_loss / total,
            correct / total * 100,
            val_acc,
            scheduler.get_last_lr()[0],
        )
    return qat_model, best_val
