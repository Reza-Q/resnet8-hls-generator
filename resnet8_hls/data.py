"""CIFAR-10 data loading helpers."""

from __future__ import annotations

from pathlib import Path

import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .model import INPUT_SIZE

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CROP_PADDING = 4
EVAL_BATCH_SIZE = 256
"""Batch size used when evaluating on the test set."""


def cifar10_transform(*, train: bool) -> transforms.Compose:
    """Standard CIFAR-10 preprocessing; the training variant adds crop and flip augmentation."""
    steps: list[object] = []
    if train:
        steps += [
            transforms.RandomCrop(INPUT_SIZE, padding=CROP_PADDING),
            transforms.RandomHorizontalFlip(),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(steps)


def load_cifar10(data_dir: str | Path, *, train: bool, download: bool = True) -> Dataset:
    """Load the CIFAR-10 training or test split (downloading it if allowed and missing).

    Raises:
        FileNotFoundError: If the dataset is missing and ``download`` is false.
    """
    try:
        return torchvision.datasets.CIFAR10(
            str(data_dir),
            train=train,
            download=download,
            transform=cifar10_transform(train=train),
        )
    except RuntimeError as exc:
        raise FileNotFoundError(
            f"CIFAR-10 was not found in '{data_dir}' or is corrupted ({exc}). "
            "Allow the download or place the dataset there."
        ) from exc


def make_loader(
    dataset: Dataset, batch_size: int, *, shuffle: bool, drop_last: bool = False
) -> DataLoader:
    """Create a single-process data loader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )
