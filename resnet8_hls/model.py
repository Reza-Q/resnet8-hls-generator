"""CIFAR-style ResNet, batch-norm fusion and network-shape helpers.

The expected parameter names follow the common CIFAR ResNet layout::

    conv1, bn1, layer{1,2,3}.{block}.{conv1,bn1,conv2,bn2,shortcut.{0,1}}, linear
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

BN_EPS = 1e-5
"""Epsilon used by ``nn.BatchNorm2d`` (must match the value used in training)."""

INPUT_SIZE = 32
"""Spatial size (height == width) of the network input."""

IMG_CHANNELS = 3
STAGES = (1, 2, 3)

FusedWeights = dict[str, tuple[np.ndarray, ...]]
"""HLS layer name -> parameter arrays.

* ``conv1``: ``(weight, bias)``
* ``layerS_blockB``: ``(w1, b1, w2, b2)`` plus ``(w_proj, b_proj)`` for projection shortcuts
* ``fc``: ``(weight, bias)`` (not batch-norm fused)
"""


# --------------------------------------------------------------------------- #
# PyTorch model
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    """Two 3x3 convolutions with batch norm and an identity or 1x1 projection shortcut."""

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """Three-stage CIFAR ResNet with a global average pool and a linear classifier."""

    def __init__(
        self,
        stem_channels: int,
        planes: Sequence[int],
        num_blocks: Sequence[int],
        num_classes: int,
    ) -> None:
        super().__init__()
        self.in_planes = stem_channels
        self.conv1 = nn.Conv2d(IMG_CHANNELS, stem_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_channels)
        self.layer1 = self._make_layer(planes[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(planes[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(planes[2], num_blocks[2], stride=2)
        self.linear = nn.Linear(planes[2], num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for block_stride in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[2])
        return self.linear(out.view(out.size(0), -1))


def count_blocks(sd: Mapping[str, Any], stage: int) -> int:
    """Number of residual blocks in ``layer{stage}``."""
    count = 0
    while f"layer{stage}.{count}.conv1.weight" in sd:
        count += 1
    return count


def build_resnet_from_state_dict(sd: Mapping[str, Any]) -> ResNet:
    """Create a randomly initialised :class:`ResNet` whose shape matches ``sd``.

    ``sd`` may hold tensors or numpy arrays; only shapes are inspected.
    """
    stem_channels = sd["conv1.weight"].shape[0]
    num_blocks = [count_blocks(sd, s) for s in STAGES]
    planes = [sd[f"layer{s}.0.conv2.weight"].shape[0] for s in STAGES]
    num_classes = sd["linear.weight"].shape[0]
    return ResNet(stem_channels, planes, num_blocks, num_classes)


def build_model_from_numpy(sd_np: Mapping[str, np.ndarray]) -> ResNet:
    """Build a model from ``sd_np``, load its weights and switch it to eval mode."""
    model = build_resnet_from_state_dict(sd_np)
    model.load_state_dict({k: torch.tensor(v) for k, v in sd_np.items()})
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Batch-norm fusion
# --------------------------------------------------------------------------- #
def fuse_bn(
    conv_w: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold an inference-mode batch norm into the preceding bias-free convolution.

    Returns:
        ``(fused_weight, fused_bias)`` as float32 arrays.
    """
    scale = gamma / np.sqrt(var + BN_EPS)
    fused_w = (conv_w * scale[:, None, None, None]).astype(np.float32)
    fused_b = (beta - mean * scale).astype(np.float32)
    return fused_w, fused_b


def fuse_bn_proj(
    conv_w: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`fuse_bn` for a 1x1 projection; the weight is returned as ``[out, in]``."""
    fused_w, fused_b = fuse_bn(conv_w, gamma, beta, mean, var)
    return fused_w[:, :, 0, 0], fused_b


def get_fused_weights(sd: Mapping[str, np.ndarray]) -> FusedWeights:
    """Return batch-norm-fused parameters for every HLS layer (see :data:`FusedWeights`)."""
    cache: FusedWeights = {}
    cache["conv1"] = fuse_bn(
        sd["conv1.weight"],
        sd["bn1.weight"],
        sd["bn1.bias"],
        sd["bn1.running_mean"],
        sd["bn1.running_var"],
    )
    for stage in STAGES:
        block = 0
        while f"layer{stage}.{block}.conv1.weight" in sd:
            prefix = f"layer{stage}.{block}"
            fw1, fb1 = fuse_bn(
                sd[f"{prefix}.conv1.weight"],
                sd[f"{prefix}.bn1.weight"],
                sd[f"{prefix}.bn1.bias"],
                sd[f"{prefix}.bn1.running_mean"],
                sd[f"{prefix}.bn1.running_var"],
            )
            fw2, fb2 = fuse_bn(
                sd[f"{prefix}.conv2.weight"],
                sd[f"{prefix}.bn2.weight"],
                sd[f"{prefix}.bn2.bias"],
                sd[f"{prefix}.bn2.running_mean"],
                sd[f"{prefix}.bn2.running_var"],
            )
            arrays = [fw1, fb1, fw2, fb2]
            if f"{prefix}.shortcut.0.weight" in sd:
                fwp, fbp = fuse_bn_proj(
                    sd[f"{prefix}.shortcut.0.weight"],
                    sd[f"{prefix}.shortcut.1.weight"],
                    sd[f"{prefix}.shortcut.1.bias"],
                    sd[f"{prefix}.shortcut.1.running_mean"],
                    sd[f"{prefix}.shortcut.1.running_var"],
                )
                arrays.extend([fwp, fbp])
            cache[f"layer{stage}_block{block}"] = tuple(arrays)
            block += 1
    cache["fc"] = (sd["linear.weight"], sd["linear.bias"])
    return cache


# --------------------------------------------------------------------------- #
# Network description
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BlockSpec:
    """Shape information of one residual block."""

    name: str
    """HLS function name, e.g. ``layer2_block0``."""
    prefix: str
    """State-dict prefix, e.g. ``layer2.0``."""
    c_in: int
    c_mid: int
    c_out: int
    h_in: int
    h_out: int
    stride: int


@dataclass(frozen=True)
class NetworkSpec:
    """Shape information needed by the design-space exploration and the code generator."""

    stem_channels: int
    final_channels: int
    final_spatial: int
    fc_in: int
    fc_out: int
    blocks: tuple[BlockSpec, ...]


def spatial_map(sd: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """Output feature-map size of the stem and of every residual block."""
    sizes = {"conv1": (INPUT_SIZE, INPUT_SIZE)}
    height = INPUT_SIZE
    for stage in STAGES:
        block = 0
        while f"layer{stage}.{block}.conv1.weight" in sd:
            if block == 0 and stage > 1:
                height //= 2
            sizes[f"layer{stage}.{block}"] = (height, height)
            block += 1
    return sizes


def hls_layer_names(sd: Mapping[str, Any]) -> list[str]:
    """HLS layer names in network order: stem, residual blocks, global pool, classifier."""
    names = ["conv1"]
    for stage in STAGES:
        block = 0
        while f"layer{stage}.{block}.conv1.weight" in sd:
            names.append(f"layer{stage}_block{block}")
            block += 1
    return [*names, "gap", "fc"]


def describe_network(sd: Mapping[str, Any]) -> NetworkSpec:
    """Collect all shape information from a state dict (tensors or numpy arrays)."""
    sizes = spatial_map(sd)
    blocks = []
    for stage in STAGES:
        block = 0
        while f"layer{stage}.{block}.conv1.weight" in sd:
            prefix = f"layer{stage}.{block}"
            stride = 2 if (block == 0 and stage > 1) else 1
            w1 = sd[f"{prefix}.conv1.weight"]
            h_out = sizes[prefix][0]
            blocks.append(
                BlockSpec(
                    name=f"layer{stage}_block{block}",
                    prefix=prefix,
                    c_in=int(w1.shape[1]),
                    c_mid=int(w1.shape[0]),
                    c_out=int(sd[f"{prefix}.conv2.weight"].shape[0]),
                    h_in=h_out * stride,
                    h_out=h_out,
                    stride=stride,
                )
            )
            block += 1
    return NetworkSpec(
        stem_channels=int(sd["conv1.weight"].shape[0]),
        final_channels=int(sd["layer3.0.conv2.weight"].shape[0]),
        final_spatial=sizes["layer3.0"][0],
        fc_in=int(sd["linear.weight"].shape[1]),
        fc_out=int(sd["linear.weight"].shape[0]),
        blocks=tuple(blocks),
    )
