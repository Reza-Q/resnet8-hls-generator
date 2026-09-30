"""Shared fixtures: a tiny random ResNet and a random CIFAR-like dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from resnet8_hls.model import ResNet


class RandomImageDataset(Dataset):
    """Deterministic random tensors shaped like normalized CIFAR-10 images."""

    def __init__(self, size: int = 64, num_classes: int = 10, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(size, 3, 32, 32, generator=generator)
        self.labels = torch.randint(0, num_classes, (size,), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def build_random_resnet(blocks=(1, 1, 1), stem=4, planes=(4, 8, 16), classes=10, seed=0) -> ResNet:
    """Create a small ResNet with non-trivial batch-norm parameters, in eval mode."""
    torch.manual_seed(seed)
    model = ResNet(stem, list(planes), list(blocks), classes)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data = torch.rand_like(module.weight) * 1.5 + 0.25
            module.bias.data = torch.randn_like(module.bias) * 0.3
            module.running_mean = torch.randn_like(module.running_mean) * 0.2
            module.running_var = torch.rand_like(module.running_var) * 1.5 + 0.5
    return model.eval()


@pytest.fixture
def tiny_model() -> ResNet:
    return build_random_resnet(blocks=(2, 1, 2))


@pytest.fixture
def tiny_state_dict(tiny_model) -> dict[str, np.ndarray]:
    return {k: v.detach().numpy() for k, v in tiny_model.state_dict().items()}


@pytest.fixture
def tiny_checkpoint(tmp_path, tiny_model):
    path = tmp_path / "tiny.pth"
    torch.save(tiny_model.state_dict(), path)
    return path


@pytest.fixture
def image_dataset() -> RandomImageDataset:
    return RandomImageDataset(size=64)
