# resnet8-hls-generator

[![CI](https://github.com/YOUR_USERNAME/resnet8-hls-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/resnet8-hls-generator/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Generate quantized **Vitis HLS C++** projects from a trained **PyTorch CIFAR-10 ResNet8**
checkpoint.

The tool folds batch normalization into the convolutions, quantizes the network to
`ap_fixed` fixed point (quantization-aware training or post-training quantization) and
writes **eight ready-to-synthesize HLS projects** that differ only in their pragma
settings (pipeline initiation interval, loop unrolling, array partitioning). Comparing
their synthesis reports shows the latency / resource trade-off on a small FPGA such as the
Zynq-7020. A C testbench with CIFAR-10 samples and one batch Tcl script are generated too.

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line reference](#command-line-reference)
- [Python API](#python-api)
- [Checkpoint format](#checkpoint-format)
- [Output layout](#output-layout)
- [Pragma configurations](#pragma-configurations)
- [Running Vitis HLS](#running-vitis-hls)
- [Project layout](#project-layout)
- [Development](#development)
- [Notes and limitations](#notes-and-limitations)
- [License](#license)

## How it works

1. **Load** the checkpoint and infer the network shape (channels, blocks per stage, number
   of classes) from the tensor shapes.
2. **Fuse** every batch-norm layer into the preceding convolution.
3. **Report** the per-layer quantization error (SQNR) of the fused parameters.
4. **Quantize** to `ap_fixed<W,I,AP_TRN,AP_SAT>`, either with quantization-aware training
   (QAT, straight-through estimator) or with post-training quantization (PTQ).
5. **Explore** eight pragma configurations (P0 to P7) and estimate their DSP / LUT /
   latency cost.
6. **Generate** one Vitis HLS source folder per configuration from Jinja2 templates, plus a
   C testbench for the first one.
7. **Write** `create_project.tcl`, which builds all eight projects in one batch run.

## Requirements

- Python 3.10 or newer
- PyTorch 2.0+ and torchvision 0.15+ (a CPU-only installation is enough)
- NumPy and Jinja2
- The CIFAR-10 dataset (about 170 MB). torchvision downloads it on first use; it is needed
  for QAT and for the testbench samples. QAT runs on the CPU, which is fine for a network
  this small.
- AMD/Xilinx **Vitis HLS**, only to synthesize the generated projects. It is not needed to
  run this tool and cannot be installed with `pip`.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/resnet8-hls-generator.git
cd resnet8-hls-generator

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .                   # add ".[dev]" for the test and lint tools
```

To use CPU-only PyTorch wheels (much smaller than the default CUDA builds), install them
first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Quick start

Place your trained checkpoint in `models/` and run:

```bash
resnet8-hls models/final_model_5pct_rate.pth
```

This runs 10 epochs of QAT and writes everything to `hls_output/final_model_5pct_rate/`.
`python -m resnet8_hls ...` works as well.

Useful variations:

```bash
# Post-training quantization instead of QAT (no training, no CIFAR-10 train split needed)
resnet8-hls models/final_model_5pct_rate.pth --no-qat

# 12-bit weights/activations (6 integer bits), 16-bit accumulators (8 integer bits)
resnet8-hls models/final_model_5pct_rate.pth --data-bits 12 6 --acc-bits 16 8

# A different device and clock, and a custom output folder
resnet8-hls models/final_model_5pct_rate.pth --part xc7z045ffg900-2 --clock-ns 8 -o build/z045
```

No checkpoint yet? Create a randomly initialised one to try the flow:

```bash
python examples/make_demo_checkpoint.py models/demo.pth
resnet8-hls models/demo.pth --no-qat
```

## Command-line reference

```
resnet8-hls [options] checkpoint
```

| Option | Default | Description |
| --- | --- | --- |
| `checkpoint` | (required) | Path to the trained `.pth` checkpoint. |
| `-o`, `--output-dir` | `hls_output/<checkpoint name>` | Output folder. |
| `--data-dir` | `data` | Where CIFAR-10 is stored / downloaded. |
| `--no-download` | off | Do not download CIFAR-10 if it is missing. |
| `--acc-bits TOTAL INT` | `8 4` | Accumulator / bias format `ap_fixed<TOTAL,INT>`. |
| `--data-bits TOTAL INT` | same as `--acc-bits` | Weight / activation format. |
| `--no-qat` | off | Use post-training quantization instead of QAT. |
| `--epochs` | `10` | QAT epochs. |
| `--lr`, `--lr-min` | `1e-3`, `1e-5` | Initial / final learning rate (cosine schedule). |
| `--batch-size` | `128` | QAT batch size. |
| `--weight-decay` | `1e-4` | SGD weight decay. |
| `--momentum` | `0.9` | SGD momentum. |
| `--grad-clip` | `1.0` | Gradient-norm clipping threshold. |
| `--seed` | `0` | Random seed for QAT (makes runs reproducible). |
| `--part` | `xc7z020clg400-2` | Xilinx part used in the Tcl script. |
| `--clock-ns` | `10` | Target clock period in nanoseconds. |
| `--num-test-samples` | `100` | CIFAR-10 test images exported to the testbench. |
| `--weights-only` | off | Load the checkpoint with `torch.load(weights_only=True)`. |
| `-q`, `--quiet` | off | Only print warnings and errors. |

## Python API

```python
import logging

from resnet8_hls import FixedPoint, PipelineConfig, QATConfig, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")  # progress output

config = PipelineConfig(
    checkpoint="models/final_model_5pct_rate.pth",
    output_dir="hls_output/final_model_5pct_rate",
    acc_bits=FixedPoint(8, 4),  # accumulator / bias format
    data_bits=None,  # None: same as acc_bits
    enable_qat=True,  # False: post-training quantization
    qat=QATConfig(epochs=10),
)
result = run_pipeline(config)

print(result.tcl_path)  # .../create_project.tcl
print(result.best_val_accuracy)  # best QAT validation accuracy in percent
print(result.reference_accuracy)  # float32 accuracy on the testbench samples
for cfg in result.pragma_configs:  # estimated DSP / LUT / latency per configuration
    print(cfg.label, cfg.est_dsp, cfg.est_lut_k, cfg.est_latency_k)
```

`run_pipeline` also accepts `train_dataset=` and `test_dataset=` (any dataset yielding
normalized `(image, label)` pairs) to use data other than the default CIFAR-10 splits.

## Checkpoint format

The network shape is inferred from the tensors, so any three-stage CIFAR ResNet with
3x3 kernels and 32x32x3 inputs works. Parameter names must follow the common layout:

```
conv1, bn1,
layer{1,2,3}.{block}.{conv1, bn1, conv2, bn2, shortcut.0, shortcut.1},
linear
```

Supported files:

- a plain `state_dict`,
- a dict that wraps the state dict under `model_state`, `state_dict`, `model_state_dict`,
  `net`, `model` or `params`,
- a fully pickled `nn.Module`,
- the state dict of a statically quantized model (quantized tensors and packed linear
  parameters are dequantized; identity batch-norm parameters are injected).

> **Security note:** by default checkpoints are loaded with `weights_only=False`, which
> can execute arbitrary code stored in the file. Only load checkpoints you trust, or pass
> `--weights-only` for plain state dicts.

## Output layout

```
hls_output/final_model_5pct_rate/
├── create_project.tcl          # Vitis HLS batch script for all eight projects
├── pragma_0/                   # one folder per pragma configuration (pragma_0 ... pragma_7)
│   ├── common.hpp              # data_t typedef
│   ├── conv1.hpp / conv1.cpp   # stem convolution + ReLU
│   ├── layer1_block0.hpp / .cpp    # residual blocks (one pair per block)
│   ├── layer2_block0.hpp / .cpp
│   ├── layer3_block0.hpp / .cpp
│   ├── gap_layer.hpp / .cpp    # global average pooling
│   ├── fc_layer.hpp / .cpp     # fully connected classifier
│   ├── resnet8.hpp / .cpp      # top-level function with AXI-stream input and output
│   ├── tb_resnet8.cpp          # C testbench (pragma_0 only)
│   └── testbench/
│       └── test_samples.h      # inputs, labels and float32 reference logits (pragma_0 only)
├── pragma_1/
│   └── ...
└── pragma_7/
```

## Pragma configurations

All configurations keep the design small so that every project fits a small device.
`II` is the initiation interval of the pipelined loops.

| # | Label | Description |
| --- | --- | --- |
| P0 | `P0_absolute_min_ii8` | II=8 in all layers, no unroll, no partition |
| P1 | `P1_lean_ii4` | II=4 in all layers, no unroll, no partition |
| P2 | `P2_lean_ii2` | II=2 in all layers, no unroll, no partition |
| P3 | `P3_stem_ii2_rest_ii4` | Stem II=2, residual blocks and FC II=4 |
| P4 | `P4_conv_ii4_fc_ii2` | Convolutions II=4, FC II=2 (FC-latency focus) |
| P5 | `P5_conv_seq_fc_x2` | Convolutions II=4, FC unroll factor 2 |
| P6 | `P6_load_proportional_ii` | II=8 / 4 / 2 for stem / residual blocks / FC |
| P7 | `P7_resblock_ic2_rest_seq` | Residual blocks unroll the input-channel loop by 2 with a complete weight partition; everything else II=4 |

The tool prints an estimate of DSP, LUT and latency for each configuration. The numbers
are only meant to compare configurations with each other; the Vitis HLS reports are
authoritative.

## Running Vitis HLS

```bash
cd hls_output/final_model_5pct_rate
vitis_hls -f create_project.tcl
```

The script creates the projects `resnet8_P0` to `resnet8_P7`. P0 runs C simulation, C
synthesis and co-simulation with the generated testbench; P1 to P7 run C synthesis only.
The testbench prints, for every sample, the label, the hardware prediction and the largest
difference to the float32 PyTorch logits, followed by a MATCH / MISS summary.

## Project layout

```
resnet8-hls-generator/
├── resnet8_hls/                # the Python package
│   ├── cli.py                  # command-line interface
│   ├── config.py               # FixedPoint, Precision, QATConfig, PipelineConfig
│   ├── checkpoint.py           # checkpoint loading and normalization
│   ├── model.py                # ResNet, batch-norm fusion, network description
│   ├── quantization.py         # fixed-point quantizers, SQNR report, PTQ
│   ├── qat.py                  # quantization-aware training
│   ├── data.py                 # CIFAR-10 loading
│   ├── pragma_dse.py           # the eight pragma configurations
│   ├── codegen.py              # renders the HLS C++ sources
│   ├── testbench.py            # testbench and reference data
│   ├── tcl_export.py           # create_project.tcl
│   ├── pipeline.py             # end-to-end orchestration
│   └── templates/              # Jinja2 templates for the C++ code (*.j2)
├── tests/                      # pytest suite
├── examples/                   # helper scripts
├── models/                     # put your checkpoint here (ignored by git)
├── .github/workflows/ci.yml    # lint and tests on every push / pull request
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Development

```bash
pip install -e ".[dev]"

pytest                     # tests use random data: no CIFAR-10 download needed
ruff check .               # lint
ruff format .              # format
```

## Notes and limitations

These points describe behaviour inherited from the original notebook, which this package
preserves on purpose.

- **QAT and batch norm.** During QAT the batch-norm layers stay un-fused and frozen, and
  the weights are folded into the convolutions only after training. The folded weights are
  written to the C++ sources as floating-point literals and are quantized by the `ap_fixed`
  types when the design is compiled, so the deployed weights are not exactly the ones that
  were simulated during training. The testbench reports the accuracy that really matters.
- **Overflow behaviour.** The stem and FC weight types (`conv1_w_t`, `fc_w_t`) are declared
  as `ap_fixed<W,I>` and wrap around on overflow, while the residual-block types saturate
  (`AP_SAT`). If a folded weight exceeds the range of the chosen format, check the SQNR table
  or use a wider format (`--data-bits`).
- **Projection shortcuts.** When QAT weights are exported, the projection-shortcut weights
  are not snapped to the fixed-point grid (the C++ types quantize them at compile time).
- **Seeding.** The original notebook did not seed the random number generators. This package
  seeds QAT with `--seed` (default 0) so that runs are reproducible.
- **Testbench.** It uses the first `--num-test-samples` images of the CIFAR-10 test split; it
  is a functional check, not a full accuracy evaluation.

## License

Released under the MIT License. See [LICENSE](LICENSE).
