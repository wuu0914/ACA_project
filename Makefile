ROOT := $(CURDIR)
RISCV := $(ROOT)/deps/tools/riscv
HOST_PREFIX ?= /usr
HOST_LIBDIR ?= $(HOST_PREFIX)/lib/x86_64-linux-gnu
IMAGENET_ROOT ?= $(ROOT)/models/dataset/imagenet
JOBS ?= 8

UV_LITERT := uv run --offline --project envs/litert
UV_PYTORCH := uv run --offline --project envs/pytorch

CROSS_CC := riscv64-unknown-elf-gcc
CROSS_CFLAGS := -std=c11 -march=rv64imc -mabi=lp64 -mcmodel=medany \
	-msmall-data-limit=0 -O2 -g -Wall -Wextra -Werror -ffreestanding \
	-fno-builtin -fno-stack-protector -ffunction-sections -fdata-sections
LINK_FLAGS := -march=rv64imc -mabi=lp64 -mcmodel=medany \
	-nostdlib -nostartfiles -static -Wl,-T,src/platform/spike/link.ld \
	-Wl,--gc-sections

export PATH := $(RISCV)/bin:$(HOST_PREFIX)/bin:$(PATH)
export LD_LIBRARY_PATH := $(HOST_LIBDIR):$(LD_LIBRARY_PATH)
export UV_CACHE_DIR := $(ROOT)/deps/.work/uv-cache
export PYTHONDONTWRITEBYTECODE := 1
export IMAGENET_ROOT IMAGENET_ARCHIVE IMAGENET_LABELS IMAGENET_SYNSETS

.PHONY: help setup check-host check-uv deps dataset check-dataset fetch-toolchain fetch-spike fetch-model \
	fetch-tflite-reference build-spike envs env-pytorch env-litert \
	check-litert check-pytorch check-spike \
	test-models test-kernels test-optimized-kernels test-all-kernels test-c-models \
	spike-all spike-b0 spike-b1 spike-x1 spike-b2 spike-b3 spike-x3 profile-report

help:
	@echo "Environment:"
	@echo "  make setup                  fetch and build the complete environment"
	@echo "  make check-host             check the Spike host dependencies"
	@echo "  make fetch-toolchain        install the RISC-V GNU toolchain"
	@echo "  make fetch-spike            fetch the pinned Spike source"
	@echo "  make build-spike            build and install Spike"
	@echo "  make fetch-model            download the three model files"
	@echo "  make fetch-tflite-reference fetch TFLite, gemmlowp, and Ruy"
	@echo "  make envs                   create both Python environments"
	@echo "  make dataset                prepare the ImageNet validation set"
	@echo "  make check-dataset          validate the prepared ImageNet data"
	@echo ""
	@echo "Validation:"
	@echo "  make test-models            compare PyTorch and both TFLite models"
	@echo "  make test-kernels           test the readable B0 C kernels"
	@echo "  make test-all-kernels       test B0, B1, B2, and B3 kernels"
	@echo "  make test-c-models          test all six complete C models"
	@echo ""
	@echo "Spike:"
	@echo "  make spike-all              run B0, B1, X1, B2, B3, and X3"
	@echo "  make spike-b0               run one configuration (also b1/x1/b2/b3/x3)"
	@echo "  make profile-report         collect the six existing profile files"

# Complete environment. The individual targets below can also be run separately.
setup: check-host check-uv deps envs build-spike

check-host:
	@command -v dtc >/dev/null || { echo "error: install device-tree-compiler"; exit 1; }
	@test -f "$(HOST_PREFIX)/include/boost/regex.hpp" || { echo "error: install libboost-regex-dev"; exit 1; }
	@test -e "$(HOST_LIBDIR)/libboost_system.so" || { echo "error: install libboost-system-dev"; exit 1; }

check-uv:
	@command -v uv >/dev/null || { echo "error: install uv and add it to PATH"; exit 1; }

deps: fetch-toolchain fetch-spike fetch-model fetch-tflite-reference

dataset:
	python3 scripts/prepare_imagenet.py

check-dataset:
	python3 scripts/prepare_imagenet.py --check

fetch-toolchain:
	scripts/fetch-deps.sh toolchain

fetch-spike:
	scripts/fetch-deps.sh spike

fetch-model:
	scripts/fetch-deps.sh model

fetch-tflite-reference:
	scripts/fetch-deps.sh tflite-reference

build-spike: check-host fetch-toolchain fetch-spike
	mkdir -p deps/.work/spike-build
	cd deps/.work/spike-build && \
		CFLAGS="-Wno-error" CXXFLAGS="-Wno-error" \
		../../sources/spike/configure \
		--prefix="$(RISCV)" \
		--with-boost="$(HOST_PREFIX)" \
		--with-boost-libdir="$(HOST_LIBDIR)"
	$(MAKE) -C deps/.work/spike-build -j$(JOBS)
	$(MAKE) -C deps/.work/spike-build install

envs: check-uv env-pytorch env-litert

env-pytorch:
	uv sync --project envs/pytorch

env-litert:
	uv sync --project envs/litert

check-litert:
	scripts/fetch-deps.sh --check model
	test -x envs/litert/.venv/bin/python

check-pytorch: check-litert
	test -x envs/pytorch/.venv/bin/python

check-spike: check-litert
	scripts/fetch-deps.sh --check toolchain
	scripts/fetch-deps.sh --check spike
	command -v riscv64-unknown-elf-gcc
	command -v spike

# Fixed-image PyTorch FP32 / TFLite FP32 / TFLite INT8 validation.
test-models: check-pytorch
	OMP_NUM_THREADS=1 $(UV_LITERT) scripts/benchmark.py tflite
	OMP_NUM_THREADS=1 $(UV_PYTORCH) scripts/benchmark.py pytorch
	$(UV_LITERT) scripts/benchmark.py compare

# B0 unit tests against the TFLite C++ reference kernels.
test-kernels: check-litert
	scripts/fetch-deps.sh --check tflite-reference
	mkdir -p build
	gcc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-Isrc/include \
		src/kernels/add.c \
		src/kernels/average_pool2d.c \
		src/kernels/conv2d.c \
		src/kernels/depthwise_conv2d.c \
		src/kernels/fully_connected.c \
		src/kernels/quantize.c \
		src/kernels/quantized_math.c \
		src/kernels/softmax.c \
		-o build/libca_kernels.so
	g++ -std=c++17 -O2 -Wall -Wextra -Werror \
		-Wno-attributes -Wno-unused-parameter -fPIC -shared \
		-Isrc/include \
		-Ideps/sources/tensorflow \
		-Ideps/sources/gemmlowp \
		-Ideps/sources/ruy \
		tests/kernels/tflite_reference.cc \
		deps/sources/tensorflow/tensorflow/lite/kernels/internal/common.cc \
		deps/sources/tensorflow/tensorflow/lite/kernels/internal/runtime_shape.cc \
		-o build/libtflite_reference.so
	$(UV_LITERT) tests/kernels/test_kernels.py

# Targeted tests for the three software optimizations.
test-optimized-kernels: test-kernels
	mkdir -p build/b1 build/b2 build/b3
	gcc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-Isrc/include src/kernels/b1/conv2d.c src/kernels/quantized_math.c \
		-o build/b1/libb1_conv2d.so
	$(UV_LITERT) tests/kernels/test_b1_conv2d.py
	gcc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-Isrc/include src/kernels/b2/depthwise_conv2d.c \
		src/kernels/quantized_math.c -o build/b2/libb2_depthwise_conv2d.so
	$(UV_LITERT) tests/kernels/test_b2_depthwise_conv2d.py
	gcc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-Isrc/include src/kernels/b3/conv2d.c \
		src/kernels/b3/depthwise_conv2d.c -o build/b3/libb3_kernels.so
	$(UV_LITERT) tests/kernels/test_b3_requantization.py

test-all-kernels: test-optimized-kernels

# Generate and trace all six complete host C models on the fixed panda image.
test-c-models: check-litert
	mkdir -p build build/b1 build/x1 build/b2 build/b3 build/x3
	$(UV_LITERT) scripts/cdata_generate.py \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--output-source "$(ROOT)/build/efficientnet_lite0_data.c" \
		--output-header "$(ROOT)/build/efficientnet_lite0_data.h"
	$(UV_LITERT) scripts/cdata_generate.py --b1 \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--output-source "$(ROOT)/build/b1/efficientnet_lite0_data.c" \
		--output-header "$(ROOT)/build/b1/efficientnet_lite0_data.h"
	$(UV_LITERT) scripts/cdata_generate.py --b1 --b3 \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--output-source "$(ROOT)/build/b3/efficientnet_lite0_data.c" \
		--output-header "$(ROOT)/build/b3/efficientnet_lite0_data.h"
	$(UV_LITERT) scripts/cdata_generate.py --b1 --b3 \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--output-source "$(ROOT)/build/x3/efficientnet_lite0_data.c" \
		--output-header "$(ROOT)/build/x3/efficientnet_lite0_data.h"
	gcc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-Isrc/include -Ibuild src/kernels/add.c src/kernels/average_pool2d.c \
		src/kernels/conv2d.c src/kernels/depthwise_conv2d.c \
		src/kernels/fully_connected.c src/kernels/quantize.c \
		src/kernels/quantized_math.c src/kernels/softmax.c \
		src/apps/efficientnet_lite0.c build/efficientnet_lite0_data.c \
		-o build/libefficientnet_lite0.so
	$(UV_LITERT) tests/c_model/test_c_model.py \
		--library build/libefficientnet_lite0.so --trace \
		--report build/b0_panda.json
	gcc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-Isrc/include -Ibuild/b1 src/kernels/add.c src/kernels/average_pool2d.c \
		src/kernels/b1/conv2d.c src/kernels/depthwise_conv2d.c \
		src/kernels/fully_connected.c src/kernels/quantize.c \
		src/kernels/quantized_math.c src/kernels/softmax.c \
		src/apps/efficientnet_lite0.c build/b1/efficientnet_lite0_data.c \
		-o build/b1/libefficientnet_lite0.so
	$(UV_LITERT) tests/c_model/test_c_model.py \
		--library build/b1/libefficientnet_lite0.so --trace \
		--report build/b1/b1_panda.json
	gcc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-Isrc/include -Ibuild/b1 src/kernels/add.c src/kernels/average_pool2d.c \
		src/kernels/x1/conv2d.c src/kernels/depthwise_conv2d.c \
		src/kernels/fully_connected.c src/kernels/quantize.c \
		src/kernels/quantized_math.c src/kernels/softmax.c \
		src/apps/efficientnet_lite0.c build/b1/efficientnet_lite0_data.c \
		-o build/x1/libefficientnet_lite0.so
	$(UV_LITERT) tests/c_model/test_c_model.py \
		--library build/x1/libefficientnet_lite0.so --trace \
		--report build/x1/x1_panda.json
	gcc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-Isrc/include -Ibuild/b1 src/kernels/add.c src/kernels/average_pool2d.c \
		src/kernels/x1/conv2d.c src/kernels/b2/depthwise_conv2d.c \
		src/kernels/fully_connected.c src/kernels/quantize.c \
		src/kernels/quantized_math.c src/kernels/softmax.c \
		src/apps/efficientnet_lite0.c build/b1/efficientnet_lite0_data.c \
		-o build/b2/libefficientnet_lite0.so
	$(UV_LITERT) tests/c_model/test_c_model.py \
		--library build/b2/libefficientnet_lite0.so --trace \
		--report build/b2/b2_panda.json
	gcc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-Isrc/include -Ibuild/b3 src/kernels/add.c src/kernels/average_pool2d.c \
		src/kernels/b3/conv2d.c src/kernels/b3/depthwise_conv2d.c \
		src/kernels/fully_connected.c src/kernels/quantize.c \
		src/kernels/quantized_math.c src/kernels/softmax.c \
		src/apps/efficientnet_lite0.c build/b3/efficientnet_lite0_data.c \
		-o build/b3/libefficientnet_lite0.so
	$(UV_LITERT) tests/c_model/test_c_model.py \
		--library build/b3/libefficientnet_lite0.so --trace \
		--report build/b3/b3_panda.json
	gcc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-Isrc/include -Ibuild/x3 src/kernels/add.c src/kernels/average_pool2d.c \
		src/kernels/x3/conv2d.c src/kernels/x3/depthwise_conv2d.c \
		src/kernels/fully_connected.c src/kernels/quantize.c \
		src/kernels/quantized_math.c src/kernels/softmax.c \
		src/apps/efficientnet_lite0.c build/x3/efficientnet_lite0_data.c \
		-o build/x3/libefficientnet_lite0.so
	$(UV_LITERT) tests/c_model/test_c_model.py \
		--library build/x3/libefficientnet_lite0.so --trace \
		--report build/x3/x3_panda.json

# Run all configurations in order, then collect their profile summaries.
spike-all:
	$(MAKE) spike-b0
	$(MAKE) spike-b1
	$(MAKE) spike-x1
	$(MAKE) spike-b2
	$(MAKE) spike-b3
	$(MAKE) spike-x3
	$(MAKE) profile-report

spike-b0: check-spike
	mkdir -p build/phase2
	$(UV_LITERT) scripts/cdata_generate.py \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--output-source "$(ROOT)/build/efficientnet_lite0_data.c" \
		--output-header "$(ROOT)/build/efficientnet_lite0_data.h"
	$(UV_LITERT) scripts/cIO_prepare.py \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--image models/testdata/panda.jpg \
		--output-data build/phase2/phase2_test_data.c \
		--output-macs build/phase2/phase2_estimated_macs.txt
	for source in src/kernels/*.c; do \
		name=$$(basename "$$source" .c); \
		$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild \
			-c "$$source" -o "build/phase2/$$name.o"; \
	done
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild \
		-c src/apps/efficientnet_lite0.c -o build/phase2/efficientnet_lite0.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild \
		-c build/efficientnet_lite0_data.c -o build/phase2/efficientnet_lite0_data.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild \
		-c src/apps/efficientnet_spike.c -o build/phase2/efficientnet_spike.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild \
		-c build/phase2/phase2_test_data.c -o build/phase2/phase2_test_data.o
	$(CROSS_CC) -march=rv64imc -mabi=lp64 -mcmodel=medany \
		-c src/platform/spike/start.S -o build/phase2/start.o
	$(CROSS_CC) $(LINK_FLAGS) -Wl,-Map,build/phase2/efficientnet_spike.map \
		build/phase2/start.o build/phase2/efficientnet_spike.o \
		build/phase2/phase2_test_data.o build/phase2/efficientnet_lite0.o \
		build/phase2/efficientnet_lite0_data.o build/phase2/add.o \
		build/phase2/average_pool2d.o build/phase2/conv2d.o \
		build/phase2/depthwise_conv2d.o build/phase2/fully_connected.o \
		build/phase2/quantize.o build/phase2/quantized_math.o \
		build/phase2/softmax.o -o build/phase2/efficientnet_spike.elf
	riscv64-unknown-elf-objdump -d -S build/phase2/efficientnet_spike.elf \
		> build/phase2/instruction_list.txt
	spike --isa=rv64imc -m16 build/phase2/efficientnet_spike.elf
	spike -g --isa=rv64imc -m16 build/phase2/efficientnet_spike.elf \
		2> build/phase2/spike.histogram
	python3 tests/spike_sim/profile_spike.py \
		--elf build/phase2/efficientnet_spike.elf \
		--histogram build/phase2/spike.histogram \
		--estimated-macs build/phase2/phase2_estimated_macs.txt \
		--output build/phase2/phase2_spike_profile.txt

spike-b1: check-spike
	mkdir -p build/b1
	$(UV_LITERT) scripts/cdata_generate.py --b1 \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--output-source "$(ROOT)/build/b1/efficientnet_lite0_data.c" \
		--output-header "$(ROOT)/build/b1/efficientnet_lite0_data.h"
	$(UV_LITERT) scripts/cIO_prepare.py \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--image models/testdata/panda.jpg \
		--output-data build/b1/phase3_2_test_data.c \
		--output-macs build/b1/phase3_2_estimated_macs.txt
	for source in \
		src/kernels/add.c src/kernels/average_pool2d.c \
		src/kernels/b1/conv2d.c src/kernels/depthwise_conv2d.c \
		src/kernels/fully_connected.c src/kernels/quantize.c \
		src/kernels/quantized_math.c src/kernels/softmax.c; do \
		name=$$(basename "$$source" .c); \
		$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
			-c "$$source" -o "build/b1/$$name.o"; \
	done
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
		-c src/apps/efficientnet_lite0.c -o build/b1/efficientnet_lite0.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
		-c build/b1/efficientnet_lite0_data.c -o build/b1/efficientnet_lite0_data.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
		-c src/apps/efficientnet_spike.c -o build/b1/efficientnet_spike.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
		-c build/b1/phase3_2_test_data.c -o build/b1/phase3_2_test_data.o
	$(CROSS_CC) -march=rv64imc -mabi=lp64 -mcmodel=medany \
		-c src/platform/spike/start.S -o build/b1/start.o
	$(CROSS_CC) $(LINK_FLAGS) -Wl,-Map,build/b1/efficientnet_spike.map \
		build/b1/start.o build/b1/efficientnet_spike.o \
		build/b1/phase3_2_test_data.o build/b1/efficientnet_lite0.o \
		build/b1/efficientnet_lite0_data.o build/b1/add.o \
		build/b1/average_pool2d.o build/b1/conv2d.o \
		build/b1/depthwise_conv2d.o build/b1/fully_connected.o \
		build/b1/quantize.o build/b1/quantized_math.o build/b1/softmax.o \
		-o build/b1/efficientnet_spike.elf
	riscv64-unknown-elf-objdump -d -S build/b1/efficientnet_spike.elf \
		> build/b1/instruction_list.txt
	spike --isa=rv64imc -m16 build/b1/efficientnet_spike.elf
	spike -g --isa=rv64imc -m16 build/b1/efficientnet_spike.elf \
		2> build/b1/spike.histogram
	python3 tests/spike_sim/profile_spike.py \
		--elf build/b1/efficientnet_spike.elf \
		--histogram build/b1/spike.histogram \
		--estimated-macs build/b1/phase3_2_estimated_macs.txt \
		--output build/b1/phase3_2_spike_profile.txt

spike-x1: check-spike
	mkdir -p build/b1 build/x1
	$(UV_LITERT) scripts/cdata_generate.py --b1 \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--output-source "$(ROOT)/build/b1/efficientnet_lite0_data.c" \
		--output-header "$(ROOT)/build/b1/efficientnet_lite0_data.h"
	$(UV_LITERT) scripts/cIO_prepare.py \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--image models/testdata/panda.jpg \
		--output-data build/x1/phase3_3_test_data.c \
		--output-macs build/x1/phase3_3_estimated_macs.txt
	g++ -std=c++20 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-I"$(RISCV)/include" src/platform/spike/xqmac8.cc \
		-o build/x1/libxqmac8.so
	for source in \
		src/kernels/add.c src/kernels/average_pool2d.c \
		src/kernels/x1/conv2d.c src/kernels/depthwise_conv2d.c \
		src/kernels/fully_connected.c src/kernels/quantize.c \
		src/kernels/quantized_math.c src/kernels/softmax.c; do \
		name=$$(basename "$$source" .c); \
		$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
			-c "$$source" -o "build/x1/$$name.o"; \
	done
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
		-c src/apps/efficientnet_lite0.c -o build/x1/efficientnet_lite0.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
		-c build/b1/efficientnet_lite0_data.c -o build/x1/efficientnet_lite0_data.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
		-c src/apps/efficientnet_spike.c -o build/x1/efficientnet_spike.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
		-c build/x1/phase3_3_test_data.c -o build/x1/phase3_3_test_data.o
	$(CROSS_CC) -march=rv64imc -mabi=lp64 -mcmodel=medany \
		-c src/platform/spike/start.S -o build/x1/start.o
	$(CROSS_CC) $(LINK_FLAGS) -Wl,-Map,build/x1/efficientnet_spike.map \
		build/x1/start.o build/x1/efficientnet_spike.o \
		build/x1/phase3_3_test_data.o build/x1/efficientnet_lite0.o \
		build/x1/efficientnet_lite0_data.o build/x1/add.o \
		build/x1/average_pool2d.o build/x1/conv2d.o \
		build/x1/depthwise_conv2d.o build/x1/fully_connected.o \
		build/x1/quantize.o build/x1/quantized_math.o build/x1/softmax.o \
		-o build/x1/efficientnet_spike.elf
	riscv64-unknown-elf-objdump -d -S build/x1/efficientnet_spike.elf \
		> build/x1/instruction_list.txt
	spike --extlib="$(ROOT)/build/x1/libxqmac8.so" --extension=xqmac8 \
		--isa=rv64imc -m16 build/x1/efficientnet_spike.elf
	spike -g --extlib="$(ROOT)/build/x1/libxqmac8.so" --extension=xqmac8 \
		--isa=rv64imc -m16 build/x1/efficientnet_spike.elf \
		2> build/x1/spike.histogram
	python3 tests/spike_sim/profile_spike.py \
		--elf build/x1/efficientnet_spike.elf \
		--histogram build/x1/spike.histogram \
		--estimated-macs build/x1/phase3_3_estimated_macs.txt \
		--output build/x1/phase3_3_spike_profile.txt

spike-b2: check-spike
	mkdir -p build/b1 build/b2
	$(UV_LITERT) scripts/cdata_generate.py --b1 \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--output-source "$(ROOT)/build/b1/efficientnet_lite0_data.c" \
		--output-header "$(ROOT)/build/b1/efficientnet_lite0_data.h"
	$(UV_LITERT) scripts/cIO_prepare.py \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--image models/testdata/panda.jpg \
		--output-data build/b2/phase4_2_test_data.c \
		--output-macs build/b2/phase4_2_estimated_macs.txt
	g++ -std=c++20 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-I"$(RISCV)/include" src/platform/spike/xqmac8.cc \
		-o build/b2/libxqmac8.so
	for source in \
		src/kernels/add.c src/kernels/average_pool2d.c \
		src/kernels/x1/conv2d.c src/kernels/b2/depthwise_conv2d.c \
		src/kernels/fully_connected.c src/kernels/quantize.c \
		src/kernels/quantized_math.c src/kernels/softmax.c; do \
		name=$$(basename "$$source" .c); \
		$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
			-c "$$source" -o "build/b2/$$name.o"; \
	done
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
		-c src/apps/efficientnet_lite0.c -o build/b2/efficientnet_lite0.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
		-c build/b1/efficientnet_lite0_data.c -o build/b2/efficientnet_lite0_data.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
		-c src/apps/efficientnet_spike.c -o build/b2/efficientnet_spike.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b1 \
		-c build/b2/phase4_2_test_data.c -o build/b2/phase4_2_test_data.o
	$(CROSS_CC) -march=rv64imc -mabi=lp64 -mcmodel=medany \
		-c src/platform/spike/start.S -o build/b2/start.o
	$(CROSS_CC) $(LINK_FLAGS) -Wl,-Map,build/b2/efficientnet_spike.map \
		build/b2/start.o build/b2/efficientnet_spike.o \
		build/b2/phase4_2_test_data.o build/b2/efficientnet_lite0.o \
		build/b2/efficientnet_lite0_data.o build/b2/add.o \
		build/b2/average_pool2d.o build/b2/conv2d.o \
		build/b2/depthwise_conv2d.o build/b2/fully_connected.o \
		build/b2/quantize.o build/b2/quantized_math.o build/b2/softmax.o \
		-o build/b2/efficientnet_spike.elf
	riscv64-unknown-elf-objdump -d -S build/b2/efficientnet_spike.elf \
		> build/b2/instruction_list.txt
	spike --extlib="$(ROOT)/build/b2/libxqmac8.so" --extension=xqmac8 \
		--isa=rv64imc -m16 build/b2/efficientnet_spike.elf
	spike -g --extlib="$(ROOT)/build/b2/libxqmac8.so" --extension=xqmac8 \
		--isa=rv64imc -m16 build/b2/efficientnet_spike.elf \
		2> build/b2/spike.histogram
	python3 tests/spike_sim/profile_spike.py \
		--elf build/b2/efficientnet_spike.elf \
		--histogram build/b2/spike.histogram \
		--estimated-macs build/b2/phase4_2_estimated_macs.txt \
		--output build/b2/phase4_2_spike_profile.txt

spike-b3: check-spike
	mkdir -p build/b3
	$(UV_LITERT) scripts/cdata_generate.py --b1 --b3 \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--output-source "$(ROOT)/build/b3/efficientnet_lite0_data.c" \
		--output-header "$(ROOT)/build/b3/efficientnet_lite0_data.h"
	$(UV_LITERT) scripts/cIO_prepare.py \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--image models/testdata/panda.jpg \
		--output-data build/b3/phase5_1_test_data.c \
		--output-macs build/b3/phase5_1_estimated_macs.txt
	g++ -std=c++20 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-I"$(RISCV)/include" src/platform/spike/xqmac8.cc \
		-o build/b3/libxqmac8.so
	for source in \
		src/kernels/add.c src/kernels/average_pool2d.c \
		src/kernels/b3/conv2d.c src/kernels/b3/depthwise_conv2d.c \
		src/kernels/fully_connected.c src/kernels/quantize.c \
		src/kernels/quantized_math.c src/kernels/softmax.c; do \
		name=$$(basename "$$source" .c); \
		$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b3 \
			-c "$$source" -o "build/b3/$$name.o"; \
	done
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b3 \
		-c src/apps/efficientnet_lite0.c -o build/b3/efficientnet_lite0.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b3 \
		-c build/b3/efficientnet_lite0_data.c -o build/b3/efficientnet_lite0_data.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b3 \
		-c src/apps/efficientnet_spike.c -o build/b3/efficientnet_spike.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/b3 \
		-c build/b3/phase5_1_test_data.c -o build/b3/phase5_1_test_data.o
	$(CROSS_CC) -march=rv64imc -mabi=lp64 -mcmodel=medany \
		-c src/platform/spike/start.S -o build/b3/start.o
	$(CROSS_CC) $(LINK_FLAGS) -Wl,-Map,build/b3/efficientnet_spike.map \
		build/b3/start.o build/b3/efficientnet_spike.o \
		build/b3/phase5_1_test_data.o build/b3/efficientnet_lite0.o \
		build/b3/efficientnet_lite0_data.o build/b3/add.o \
		build/b3/average_pool2d.o build/b3/conv2d.o \
		build/b3/depthwise_conv2d.o build/b3/fully_connected.o \
		build/b3/quantize.o build/b3/quantized_math.o build/b3/softmax.o \
		-o build/b3/efficientnet_spike.elf
	riscv64-unknown-elf-objdump -d -S build/b3/efficientnet_spike.elf \
		> build/b3/instruction_list.txt
	spike --extlib="$(ROOT)/build/b3/libxqmac8.so" --extension=xqmac8 \
		--isa=rv64imc -m16 build/b3/efficientnet_spike.elf
	spike -g --extlib="$(ROOT)/build/b3/libxqmac8.so" --extension=xqmac8 \
		--isa=rv64imc -m16 build/b3/efficientnet_spike.elf \
		2> build/b3/spike.histogram
	python3 tests/spike_sim/profile_spike.py \
		--elf build/b3/efficientnet_spike.elf \
		--histogram build/b3/spike.histogram \
		--estimated-macs build/b3/phase5_1_estimated_macs.txt \
		--output build/b3/phase5_1_spike_profile.txt

spike-x3: check-spike
	mkdir -p build/x3
	$(UV_LITERT) scripts/cdata_generate.py --b1 --b3 \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--output-source "$(ROOT)/build/x3/efficientnet_lite0_data.c" \
		--output-header "$(ROOT)/build/x3/efficientnet_lite0_data.h"
	$(UV_LITERT) scripts/cIO_prepare.py \
		--model models/upstream/efficientnet-lite0-int8.tflite \
		--image models/testdata/panda.jpg \
		--output-data build/x3/phase5_2_test_data.c \
		--output-macs build/x3/phase5_2_estimated_macs.txt
	g++ -std=c++20 -O2 -Wall -Wextra -Werror -fPIC -shared \
		-I"$(RISCV)/include" src/platform/spike/xqmac8.cc \
		src/platform/spike/xclip8.cc -o build/x3/libx3.so
	for source in \
		src/kernels/add.c src/kernels/average_pool2d.c \
		src/kernels/x3/conv2d.c src/kernels/x3/depthwise_conv2d.c \
		src/kernels/fully_connected.c src/kernels/quantize.c \
		src/kernels/quantized_math.c src/kernels/softmax.c; do \
		name=$$(basename "$$source" .c); \
		$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/x3 \
			-c "$$source" -o "build/x3/$$name.o"; \
	done
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/x3 \
		-c src/apps/efficientnet_lite0.c -o build/x3/efficientnet_lite0.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/x3 \
		-c build/x3/efficientnet_lite0_data.c -o build/x3/efficientnet_lite0_data.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/x3 \
		-c src/apps/efficientnet_spike.c -o build/x3/efficientnet_spike.o
	$(CROSS_CC) $(CROSS_CFLAGS) -Isrc/include -Ibuild/x3 \
		-c build/x3/phase5_2_test_data.c -o build/x3/phase5_2_test_data.o
	$(CROSS_CC) -march=rv64imc -mabi=lp64 -mcmodel=medany \
		-c src/platform/spike/start.S -o build/x3/start.o
	$(CROSS_CC) $(LINK_FLAGS) -Wl,-Map,build/x3/efficientnet_spike.map \
		build/x3/start.o build/x3/efficientnet_spike.o \
		build/x3/phase5_2_test_data.o build/x3/efficientnet_lite0.o \
		build/x3/efficientnet_lite0_data.o build/x3/add.o \
		build/x3/average_pool2d.o build/x3/conv2d.o \
		build/x3/depthwise_conv2d.o build/x3/fully_connected.o \
		build/x3/quantize.o build/x3/quantized_math.o build/x3/softmax.o \
		-o build/x3/efficientnet_spike.elf
	riscv64-unknown-elf-objdump -d -S build/x3/efficientnet_spike.elf \
		> build/x3/instruction_list.txt
	spike --extlib="$(ROOT)/build/x3/libx3.so" \
		--extension=xqmac8 --extension=xclip8 \
		--isa=rv64imc -m16 build/x3/efficientnet_spike.elf
	spike -g --extlib="$(ROOT)/build/x3/libx3.so" \
		--extension=xqmac8 --extension=xclip8 \
		--isa=rv64imc -m16 build/x3/efficientnet_spike.elf \
		2> build/x3/spike.histogram
	python3 tests/spike_sim/profile_spike.py \
		--elf build/x3/efficientnet_spike.elf \
		--histogram build/x3/spike.histogram \
		--estimated-macs build/x3/phase5_2_estimated_macs.txt \
		--output build/x3/phase5_2_spike_profile.txt

profile-report:
	python3 scripts/collect_spike_profiles.py \
		--output build/spike_profile_summary.md
