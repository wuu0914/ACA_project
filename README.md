# Quantized Neural-Network Acceleration on RV64IMC - Midterm

This project evaluates software and ISA/RTL extensions for EfficientNet-Lite0
integer-only inference on bare-metal RV64IMC, using Spike for ISA-level analysis
and, finally, CV-Wally for hardware-level evaluation.

## What I have done

- Verified EfficientNet-Lite0 W8A8 workload & evaluated its PyTorch FP32, TFLite FP32, and TFLite INT8 models on ImageNet.
- Implemented every required quantized C kernel and a complete integer-only C
  inference model with byte-exact TFLite reference checks.
- Cross-compiled the full model as a RV64IMC bare-metal ELF and
  executed it on Spike without floating-point or operating-system support.
- Measured instruction mix, memory instructions, major-operator contribution,
  MACs per instruction, code size, and custom-instruction utilization.
- Implemented the B1/B2/B3 software optimizations and the Xqmac8/Xclip8 Spike
  instructions, **reducing full-model retired instructions by 81.74% from B0**.


## Environment Setup

### Supported host

Tested on Ubuntu 26.04.1 LTS (x86-64). Install the required host tools before
running `make setup`:

```sh
sudo apt update
sudo apt install -y build-essential git python3 curl ca-certificates xz-utils \
    device-tree-compiler libboost-regex-dev libboost-system-dev
```

Install [`uv`](https://docs.astral.sh/uv/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### Project Dependency

The pinned downloads are declared in `deps/manifests/*.toml` and
`models/manifest.toml`. `make setup` downloads or builds:

- RISC-V GCC 16.1.0 and Binutils 2.47
- Spike 1.1.1-dev at commit `02b1dc182164bb73b19b050676dd89f0834f8b2e`
- TensorFlow Lite reference source, gemmlowp, and Ruy
- EfficientNet-Lite0 INT8 and FP32 TFLite models
- EfficientNet-Lite0 PyTorch weights and the panda test image
- Locked LiteRT and PyTorch Python environments

Setup the whole project environment with:

```sh
make setup
```

Or prepare the environment one part at a time (Optional)

```sh
make fetch-toolchain
make fetch-spike
make build-spike
make fetch-model
make fetch-tflite-reference
make envs
```

### Download Dataset (Optional)

ImageNet is optional. Without it, the included panda image and generated kernel
inputs are sufficient to run:

```sh
make test-models
make test-all-kernels
make test-c-models
make spike-all
```

To use a local validation archive, keep `LOC_val_solution.csv` and
`LOC_synset_mapping.txt` beside it, then run:

```sh
make dataset IMAGENET_ARCHIVE=/path/to/ILSVRC2012_img_val.tar
```

Without a local archive, install and authenticate the Kaggle CLI, accept the
[competition rules](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/rules),
and download through Kaggle:

```sh
uv tool install kaggle
kaggle auth login
make dataset
```

The Kaggle fallback downloads its combined competition image archive; providing
the validation-only archive uses substantially less download and disk space.

The prepared 50,000-image validation set is stored under
`models/dataset/imagenet/`. Verify it without downloading anything with:

```sh
make check-dataset
```

## What to run

### PyTorch FP32 vs TFLite FP32 vs TFLite INT8 (Optional)

Run inferences of the python models.

```sh
make test-models
```

### C kernels (Optional)

Test the correctness of all C kernels.

```sh
make test-kernels
```

### TFLite INT8 C (Optional)

Build & Test the correctness of B0, B1, X1, B2, B3, and X3 versions.

```sh
make test-c-models
```

### Spike simulation

Run every version and write `build/spike_profile_summary.md`:

```sh
make spike-all
```

Or you can run one version seperately (Optional):

```sh
make spike-b0
make spike-b1
make spike-x1
make spike-b2
make spike-b3
make spike-x3
```

## Spike Profile Summary

| Version | Retired instructions | Reduction vs B0 | MACs / instruction | .text bytes | Xqmac8 | Xclip8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 4,449,648,842 | 0.00% | 0.086566 | 4,408 | 0 | 0 |
| B1 | 3,096,086,990 | 30.42% | 0.124411 | 5,012 | 0 | 0 |
| X1 | 1,647,729,506 | 62.97% | 0.233769 | 5,024 | 42,317,184 | 0 |
| B2 | 983,811,822 | 77.89% | 0.391526 | 5,860 | 42,317,184 | 0 |
| B3 | 851,551,446 | 80.86% | 0.452336 | 6,436 | 42,317,184 | 0 |
| X3 | 812,700,880 | 81.74% | 0.473960 | 6,150 | 42,317,184 | 6,737,696 |


## Submit Checklist
- Source code in C/C++ for the bare-metal workload: `src/apps/` and `src/kernels/`
- Quantized model in Python: will be download by `make setup`
- Spike/custom ISA implementation source codes: `src/platform/spike/`
- Verification tests for the added instructions: `tests/spike_sim/`
- Reproducible benchmark scripts: `scripts/` and `Makefile`

## AI Usage

- To write scripts but important tests (e.g. benchmark.py, cdata_generate.py, etc) are inspected by myself.
- To write small tests, such as test_b1_conv2d.py, test_kernels.py, etc.
- To accelerate coding. For example, tell him to write a pretty table output, write a new instruction as what I specified, or to generate a certain format of data output in cdata_generate.py.
