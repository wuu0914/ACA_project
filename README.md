# Quantized Neural-Network Acceleration on RV64IMC - Midterm

This project evaluates software and ISA/RTL extensions for EfficientNet-Lite0
integer-only inference on bare-metal RV64IMC, using Spike for ISA-level analysis
and, finally, CV-Wally for hardware-level evaluation.

## Environment Setup

Host needs `git`, `make`, `gcc`, `g++`, `python3`, `curl` or `wget`, and
[`uv`](https://docs.astral.sh/uv/). Toolchain, Spike, models, TFLite reference
sources, and Python packages are pinned by the manifests and lock files. **You can setup the whole environment by:**

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

### AI Usage

- To write scripts but important tests (e.g. benchmark.py, cdata_generate.py, etc) are inspected by myself.
- To write small tests, such as test_b1_conv2d.py, test_kernels.py, etc.
- To accelerate coding. For example, tell him to write a pretty table output, write a new instruction as what I specified, or to generate a certain format of data output in cdata_generate.py.
