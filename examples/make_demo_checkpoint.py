"""Create a randomly initialised ResNet8 checkpoint for trying out the generator.

The weights are random, so the resulting design is meaningless as a classifier, but it
exercises the complete flow (batch-norm fusion, quantization, code generation).

Usage::

    python examples/make_demo_checkpoint.py models/demo.pth
    resnet8-hls models/demo.pth --no-qat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from resnet8_hls.model import ResNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, nargs="?", default=Path("models/demo.pth"))
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = ResNet(stem_channels=16, planes=[16, 32, 64], num_blocks=[1, 1, 1], num_classes=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved a random ResNet8 checkpoint to {args.output}")


if __name__ == "__main__":
    main()
