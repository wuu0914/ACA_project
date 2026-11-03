#!/usr/bin/env python3
"""
    Generate static C data from the EfficientNet-Lite0 TFLite model.
    Flow:
        Read tensors and operators from .tflite
        Prepare kernel parameters and tensor arena offsets
        Generate efficientnet_lite0_data.h/.c
"""

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
from ai_edge_litert import schema_py_generated as fb


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models/upstream/efficientnet-lite0-int8.tflite"
DEFAULT_HEADER = ROOT / "build/efficientnet_lite0_data.h"
DEFAULT_SOURCE = ROOT / "build/efficientnet_lite0_data.c"

TYPE_INFO = {
    # TFLite type: (generated enum, C type, bytes per element, NumPy dtype)
    fb.TensorType.INT8: ("MODEL_TENSOR_INT8", "int8_t", 1, np.dtype("i1")),
    fb.TensorType.UINT8: ("MODEL_TENSOR_UINT8", "uint8_t", 1, np.dtype("u1")),
    fb.TensorType.INT32: ("MODEL_TENSOR_INT32", "int32_t", 4, np.dtype("<i4")),
}

# TFLite builtin operator -> generated C operator type
# Every operator listed here is implemented by a Phase 1-3 kernel.
OP_INFO = {
    fb.BuiltinOperator.QUANTIZE: "MODEL_OP_QUANTIZE",
    fb.BuiltinOperator.CONV_2D: "MODEL_OP_CONV_2D",
    fb.BuiltinOperator.DEPTHWISE_CONV_2D: "MODEL_OP_DEPTHWISE_CONV_2D",
    fb.BuiltinOperator.ADD: "MODEL_OP_ADD",
    fb.BuiltinOperator.AVERAGE_POOL_2D: "MODEL_OP_AVERAGE_POOL_2D",
    fb.BuiltinOperator.RESHAPE: "MODEL_OP_RESHAPE",
    fb.BuiltinOperator.FULLY_CONNECTED: "MODEL_OP_FULLY_CONNECTED",
    fb.BuiltinOperator.SOFTMAX: "MODEL_OP_SOFTMAX",
}


def fail(message):
    raise ValueError(message)


def product(values):
    """Return the number of elements in a tensor shape."""
    result = 1
    for value in values:
        result *= int(value)
    return result


def align_up(value, alignment=16):
    """Align every activation address to a 16-byte boundary."""
    return (value + alignment - 1) // alignment * alignment


def tflite_round(value):
    """TFLite round-to-nearest with halfway cases away from zero."""
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def quantize_multiplier(real_multiplier):
    """
    Convert a real multiplier to the integer form used by TFLite kernels.

    Math:
        real_multiplier ~= multiplier / 2^31 * 2^shift
        significand is in [0.5, 1.0), so multiplier is a Q0.31 value.

    Notes:
        If rounding produces 2^31, divide it by two and increase shift.
        A shift below -31 is too small for TFLite and becomes zero.
    """
    if real_multiplier == 0.0:
        return 0, 0
    significand, shift = math.frexp(real_multiplier)
    multiplier = tflite_round(significand * (1 << 31))
    if multiplier == 1 << 31:
        multiplier //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    if not 0 <= multiplier <= (1 << 31) - 1:
        fail(f"invalid quantized multiplier for {real_multiplier}")
    return multiplier, shift


def tensor_quantization(tensor, tensor_index):
    """Read per-tensor or per-channel scale and zero point arrays."""
    quantization = tensor.quantization
    if quantization is None or quantization.scale is None or quantization.zeroPoint is None:
        fail(f"tensor {tensor_index} has no affine quantization")
    scales = [float(value) for value in quantization.scale]
    zero_points = [int(value) for value in quantization.zeroPoint]
    if not scales or not zero_points:
        fail(f"tensor {tensor_index} has empty quantization")
    return scales, zero_points


def scalar_quantization(tensor, tensor_index):
    """Read one scale and zero point from a per-tensor quantized tensor."""
    scales, zero_points = tensor_quantization(tensor, tensor_index)
    if len(scales) != 1 or len(zero_points) != 1:
        fail(f"tensor {tensor_index} is not per-tensor quantized")
    return scales[0], zero_points[0]


def activation_range(activation, output, output_index):
    """
    Convert a fused activation's real limits to output quantized limits.

    Math:
        q = output_zero_point + round(real_value / output_scale)
    """
    scale, zero_point = scalar_quantization(output, output_index)
    if output.type == fb.TensorType.INT8:
        qmin, qmax = -128, 127
    elif output.type == fb.TensorType.UINT8:
        qmin, qmax = 0, 255
    else:
        fail(f"unsupported activation output type for tensor {output_index}")

    def quantize(real_value):
        # TFLite CalculateActivationRangeQuantized performs this division in float.
        scaled = float(np.float32(real_value) / np.float32(scale))
        return zero_point + tflite_round(scaled)

    if activation == fb.ActivationFunctionType.NONE:
        return qmin, qmax
    if activation == fb.ActivationFunctionType.RELU:
        return max(qmin, quantize(0.0)), qmax
    if activation == fb.ActivationFunctionType.RELU6:
        return max(qmin, quantize(0.0)), min(qmax, quantize(6.0))
    if activation == fb.ActivationFunctionType.RELU_N1_TO_1:
        return max(qmin, quantize(-1.0)), min(qmax, quantize(1.0))
    fail(f"unsupported fused activation {activation}")


def padding_before(input_size, filter_size, output_size,
                   stride, dilation, padding):
    """
    Calculate the top or left padding used by TFLite SAME padding.

    Math:
        effective_filter = (filter_size - 1) * dilation + 1
        total_padding = (output_size - 1) * stride
                        + effective_filter - input_size
        padding_before = max(0, total_padding) // 2
    """
    if padding == fb.Padding.VALID:
        return 0
    if padding != fb.Padding.SAME:
        fail(f"unsupported padding enum {padding}")
    effective_filter = (filter_size - 1) * dilation + 1
    total = max(0, (output_size - 1) * stride + effective_filter - input_size)
    return total // 2


def format_array(values, c_type, name, alignment=None):
    """Format a NumPy array as a static C array, 16 elements per line."""
    flat = values.reshape(-1)
    attribute = (f" __attribute__((aligned({alignment})))"
                 if alignment is not None else "")
    lines = [f"static const {c_type} {name}[]{attribute} = {{"]
    for start in range(0, flat.size, 16):
        text = ", ".join(str(int(value)) for value in flat[start:start + 16])
        lines.append(f"  {text},")
    lines.append("};")
    lines.append("")
    return lines


def load_model(path):
    """Read and validate the EfficientNet-Lite0 TFLite FlatBuffer."""
    raw = path.read_bytes()
    if not fb.Model.ModelBufferHasIdentifier(raw, 0):
        fail(f"{path} is not a TFLite FlatBuffer")
    model = fb.ModelT.InitFromObj(fb.Model.GetRootAs(raw, 0))
    if model.version != 3 or len(model.subgraphs) != 1:
        fail("expected one TFLite schema-v3 subgraph")
    return model, raw


def make_tensor_records(model, graph):
    """
    Prepare one record for every tensor in the TFLite graph.

    Notes:
        A tensor with FlatBuffer data is a constant weight or bias.
        A tensor without data is an activation stored in the tensor arena.
    """
    records = []
    for index, tensor in enumerate(graph.tensors):
        if tensor.type not in TYPE_INFO:
            fail(f"tensor {index} has unsupported type {tensor.type}")
        shape = [int(value) for value in tensor.shape]
        if any(value <= 0 for value in shape):
            fail(f"tensor {index} has dynamic/invalid shape {shape}")
        _, _, item_size, _ = TYPE_INFO[tensor.type]
        size = product(shape) * item_size
        buffer_data = model.buffers[int(tensor.buffer)].data
        data = bytes(buffer_data) if buffer_data is not None else b""
        if data and len(data) != size:
            fail(f"tensor {index} buffer has {len(data)} bytes, expected {size}")
        records.append({
            "index": index,
            "tensor": tensor,
            "shape": shape,
            "bytes": size,
            "constant": bool(data),
            "data": data,
            "arena_offset": 0,
        })
    return records


def plan_arena(graph, tensors, op_codes):
    """
    Reuse memory between activation tensors that are not alive together.

    Flow:
        Make RESHAPE input/output tensors share one buffer
        Find every tensor's producer and last consumer
        Merge the lifetime of tensors that share a buffer
        Assign the first available 16-byte-aligned arena address

    Tensor lifetime:
        start = producer operator
        end   = last consumer operator

    Notes:
        The model input starts before operator 0, so its producer is -1.
        The model output stays alive through operator_count.
        Tensors with non-overlapping lifetimes can share an arena address.
    """
    parent = list(range(len(tensors)))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    # RESHAPE only changes the tensor view, so input and output share a buffer.
    for op, code in zip(graph.operators, op_codes):
        if code == fb.BuiltinOperator.RESHAPE:
            input_index = int(op.inputs[0])
            output_index = int(op.outputs[0])
            if tensors[input_index]["constant"] or tensors[output_index]["constant"]:
                fail("RESHAPE activation unexpectedly uses a constant tensor")
            if tensors[input_index]["bytes"] != tensors[output_index]["bytes"]:
                fail("RESHAPE changes element count")
            union(input_index, output_index)

    # Model input exists before the first operator.
    producer = {int(index): -1 for index in graph.inputs}
    consumers = {i: [] for i in range(len(tensors))}

    # Find each tensor's producer and consumers.
    for op_index, op in enumerate(graph.operators):
        for tensor_index in op.inputs:
            if int(tensor_index) >= 0:
                consumers[int(tensor_index)].append(op_index)
        for tensor_index in op.outputs:
            producer[int(tensor_index)] = op_index

    # Keep model output alive after the final operator returns.
    for output_index in graph.outputs:
        consumers[int(output_index)].append(len(graph.operators))

    # Merge tensors that must share one buffer.
    groups = {}
    for tensor in tensors:
        if tensor["constant"]:
            continue
        index = tensor["index"]
        root = find(index)
        start = producer.get(index)
        if start is None:
            fail(f"activation tensor {index} has no producer")
        end = max(consumers[index], default=start)
        group = groups.setdefault(root, {
            "root": root,
            "members": [],
            "start": start,
            "end": end,
            "bytes": tensor["bytes"],
        })
        group["members"].append(index)
        group["start"] = min(group["start"], start)
        group["end"] = max(group["end"], end)
        group["bytes"] = max(group["bytes"], tensor["bytes"])

    # First-fit allocation: memory from an expired lifetime is reusable.
    allocated = []
    arena_size = 0
    for group in sorted(groups.values(), key=lambda item: (item["start"], item["root"])):
        occupied = sorted(
            (other["offset"], other["bytes"])
            for other in allocated
            # end == start still overlaps because the operator reads its input
            # before writing its output.
            if other["end"] >= group["start"]
        )
        candidate = 0
        for offset, size in occupied:
            candidate = align_up(candidate)
            if candidate + group["bytes"] <= offset:
                break
            candidate = max(candidate, offset + size)
        group["offset"] = align_up(candidate)
        allocated.append(group)
        arena_size = max(arena_size, group["offset"] + group["bytes"])
        for index in group["members"]:
            tensors[index]["arena_offset"] = group["offset"]

    return align_up(arena_size)


def build_operator_records(model, graph, tensors):
    """
    Convert TFLite operator options into Phase 1-3 kernel parameters.

    Flow:
        Decode the TFLite builtin operator
        Read its input/output shapes and quantization
        Calculate padding, activation range, multiplier and shift
        Store tensor indices and prepared parameters in one operator record

    Notes:
        This is the offline version of TFLite Prepare().
        Records contain tensor indices; the C executor resolves their pointers.
    """
    # Older FlatBuffers store operators <= 127 in deprecatedBuiltinCode and
    # leave builtinCode at zero. Newer operators use the wider builtinCode.
    op_codes = [
        max(int(model.operatorCodes[int(op.opcodeIndex)].deprecatedBuiltinCode),
            int(model.operatorCodes[int(op.opcodeIndex)].builtinCode))
        for op in graph.operators
    ]
    if any(code not in OP_INFO for code in op_codes):
        unsupported = sorted(set(op_codes) - set(OP_INFO))
        fail(f"unsupported operators: {unsupported}")

    records = []
    for op_index, (op, code) in enumerate(zip(graph.operators, op_codes)):
        inputs = [int(value) for value in op.inputs]
        outputs = [int(value) for value in op.outputs]
        if len(outputs) != 1:
            fail(f"operator {op_index} has {len(outputs)} outputs")
        padded_inputs = (inputs + [-1, -1, -1])[:3]
        record = {
            "index": op_index,
            "code": code,
            "type": OP_INFO[code],
            "inputs": padded_inputs,
            "output": outputs[0],
            "params_member": "quantize",
            "params": [],
            "multipliers": None,
            "shifts": None,
            "multiplier": 0,
            "shift": 0,
        }

        input0 = graph.tensors[inputs[0]]
        output = graph.tensors[outputs[0]]

        if code == fb.BuiltinOperator.QUANTIZE:
            # Math:
            #   real_multiplier = input_scale / output_scale
            input_scale, input_zero = scalar_quantization(input0, inputs[0])
            output_scale, output_zero = scalar_quantization(output, outputs[0])
            multiplier, shift = quantize_multiplier(input_scale / output_scale)
            record["params_member"] = "quantize"
            record["params"] = [
                ("size", product(tensors[outputs[0]]["shape"])),
                ("input_zero_point", input_zero),
                ("output_zero_point", output_zero),
                ("multiplier", multiplier),
                ("shift", shift),
                ("input_type", TYPE_INFO[input0.type][0]),
                ("output_type", TYPE_INFO[output.type][0]),
            ]

        elif code == fb.BuiltinOperator.CONV_2D:
            # Math:
            #   real_multiplier[c] = input_scale * filter_scale[c]
            #                          / output_scale
            options = op.builtinOptions
            filter_tensor = graph.tensors[inputs[1]]
            input_shape = tensors[inputs[0]]["shape"]
            filter_shape = tensors[inputs[1]]["shape"]
            output_shape = tensors[outputs[0]]["shape"]
            input_scale, input_zero = scalar_quantization(input0, inputs[0])
            output_scale, output_zero = scalar_quantization(output, outputs[0])
            filter_scales, filter_zeros = tensor_quantization(filter_tensor, inputs[1])
            if any(value != 0 for value in filter_zeros):
                fail(f"CONV_2D {op_index} filter is not symmetric")
            if len(filter_scales) == 1:
                filter_scales *= output_shape[3]
            if len(filter_scales) != output_shape[3]:
                fail(f"CONV_2D {op_index} has invalid per-channel scales")
            quantized = [quantize_multiplier(input_scale * scale / output_scale)
                         for scale in filter_scales]
            record["multipliers"] = [value[0] for value in quantized]
            record["shifts"] = [value[1] for value in quantized]
            act_min, act_max = activation_range(
                options.fusedActivationFunction, output, outputs[0])
            pad_h = padding_before(input_shape[1], filter_shape[1], output_shape[1],
                                   options.strideH, options.dilationHFactor,
                                   options.padding)
            pad_w = padding_before(input_shape[2], filter_shape[2], output_shape[2],
                                   options.strideW, options.dilationWFactor,
                                   options.padding)
            record["params_member"] = "conv2d"
            record["params"] = [
                ("batches", input_shape[0]), ("input_H", input_shape[1]),
                ("input_W", input_shape[2]), ("input_C", input_shape[3]),
                ("filter_H", filter_shape[1]), ("filter_W", filter_shape[2]),
                ("filter_input_C", filter_shape[3]),
                ("output_H", output_shape[1]), ("output_W", output_shape[2]),
                ("output_C", output_shape[3]), ("stride_H", options.strideH),
                ("stride_W", options.strideW),
                ("dilation_H", options.dilationHFactor),
                ("dilation_W", options.dilationWFactor),
                ("pad_H", pad_h), ("pad_W", pad_w),
                ("input_offset", -input_zero), ("output_offset", output_zero),
                ("activation_min", act_min), ("activation_max", act_max),
            ]

        elif code == fb.BuiltinOperator.DEPTHWISE_CONV_2D:
            # Math:
            #   real_multiplier[c] = input_scale * filter_scale[c]
            #                          / output_scale
            options = op.builtinOptions
            filter_tensor = graph.tensors[inputs[1]]
            input_shape = tensors[inputs[0]]["shape"]
            filter_shape = tensors[inputs[1]]["shape"]
            output_shape = tensors[outputs[0]]["shape"]
            input_scale, input_zero = scalar_quantization(input0, inputs[0])
            output_scale, output_zero = scalar_quantization(output, outputs[0])
            filter_scales, filter_zeros = tensor_quantization(filter_tensor, inputs[1])
            if any(value != 0 for value in filter_zeros):
                fail(f"DEPTHWISE_CONV_2D {op_index} filter is not symmetric")
            if len(filter_scales) == 1:
                filter_scales *= output_shape[3]
            quantized = [quantize_multiplier(input_scale * scale / output_scale)
                         for scale in filter_scales]
            record["multipliers"] = [value[0] for value in quantized]
            record["shifts"] = [value[1] for value in quantized]
            act_min, act_max = activation_range(
                options.fusedActivationFunction, output, outputs[0])
            pad_h = padding_before(input_shape[1], filter_shape[1], output_shape[1],
                                   options.strideH, options.dilationHFactor,
                                   options.padding)
            pad_w = padding_before(input_shape[2], filter_shape[2], output_shape[2],
                                   options.strideW, options.dilationWFactor,
                                   options.padding)
            record["params_member"] = "depthwise_conv2d"
            record["params"] = [
                ("batches", input_shape[0]), ("input_H", input_shape[1]),
                ("input_W", input_shape[2]), ("input_C", input_shape[3]),
                ("filter_H", filter_shape[1]), ("filter_W", filter_shape[2]),
                ("output_H", output_shape[1]), ("output_W", output_shape[2]),
                ("output_C", output_shape[3]),
                ("depth_multiplier", options.depthMultiplier),
                ("stride_H", options.strideH), ("stride_W", options.strideW),
                ("dilation_H", options.dilationHFactor),
                ("dilation_W", options.dilationWFactor),
                ("pad_H", pad_h), ("pad_W", pad_w),
                ("input_offset", -input_zero), ("output_offset", output_zero),
                ("activation_min", act_min), ("activation_max", act_max),
            ]

        elif code == fb.BuiltinOperator.ADD:
            # Math:
            #   common_scale = 2 * max(input1_scale, input2_scale)
            #   Each input is mapped to common_scale before addition.
            #   The sum is then mapped from common_scale to output_scale.
            #
            # Notes:
            #   TFLite reserves 20 left-shift bits for addition precision.
            if tensors[inputs[0]]["shape"] != tensors[inputs[1]]["shape"]:
                fail(f"ADD {op_index} requires broadcasting")
            scale1, zero1 = scalar_quantization(input0, inputs[0])
            scale2, zero2 = scalar_quantization(graph.tensors[inputs[1]], inputs[1])
            output_scale, output_zero = scalar_quantization(output, outputs[0])
            common_scale = 2.0 * max(scale1, scale2)
            m1, s1 = quantize_multiplier(scale1 / common_scale)
            m2, s2 = quantize_multiplier(scale2 / common_scale)
            mo, so = quantize_multiplier(common_scale / ((1 << 20) * output_scale))
            act_min, act_max = activation_range(
                op.builtinOptions.fusedActivationFunction, output, outputs[0])
            record["params_member"] = "add"
            record["params"] = [
                ("size", product(tensors[outputs[0]]["shape"])),
                ("input1_offset", -zero1), ("input2_offset", -zero2),
                ("output_offset", output_zero), ("left_shift", 20),
                ("input1_multiplier", m1), ("input1_shift", s1),
                ("input2_multiplier", m2), ("input2_shift", s2),
                ("output_multiplier", mo), ("output_shift", so),
                ("activation_min", act_min), ("activation_max", act_max),
            ]

        elif code == fb.BuiltinOperator.AVERAGE_POOL_2D:
            # Notes:
            #   The Phase 1-3 kernel directly averages quantized values, so
            #   input and output must use the same scale and zero point.
            options = op.builtinOptions
            input_shape = tensors[inputs[0]]["shape"]
            output_shape = tensors[outputs[0]]["shape"]
            input_q = scalar_quantization(input0, inputs[0])
            output_q = scalar_quantization(output, outputs[0])
            if input_q != output_q:
                fail("AVERAGE_POOL_2D input/output quantization differs")
            act_min, act_max = activation_range(
                options.fusedActivationFunction, output, outputs[0])
            record["params_member"] = "average_pool2d"
            record["params"] = [
                ("batches", input_shape[0]), ("input_H", input_shape[1]),
                ("input_W", input_shape[2]), ("input_C", input_shape[3]),
                ("output_H", output_shape[1]), ("output_W", output_shape[2]),
                ("filter_H", options.filterHeight),
                ("filter_W", options.filterWidth),
                ("stride_H", options.strideH), ("stride_W", options.strideW),
                ("pad_H", padding_before(input_shape[1], options.filterHeight,
                                          output_shape[1], options.strideH, 1,
                                          options.padding)),
                ("pad_W", padding_before(input_shape[2], options.filterWidth,
                                          output_shape[2], options.strideW, 1,
                                          options.padding)),
                ("activation_min", act_min), ("activation_max", act_max),
            ]

        elif code == fb.BuiltinOperator.RESHAPE:
            # Notes:
            #   RESHAPE changes only the tensor view. plan_arena() assigns its
            #   input and output the same address, so no kernel is required.
            if tensors[inputs[0]]["bytes"] != tensors[outputs[0]]["bytes"]:
                fail("RESHAPE changes byte count")
            record["params_member"] = "reshape"
            record["params"] = [("unused", 0)]

        elif code == fb.BuiltinOperator.FULLY_CONNECTED:
            # Math:
            #   real_multiplier = input_scale * weight_scale / output_scale
            #
            # Notes:
            #   The Phase 1-3 kernel uses one weight scale and one multiplier.
            options = op.builtinOptions
            input_shape = tensors[inputs[0]]["shape"]
            filter_shape = tensors[inputs[1]]["shape"]
            output_shape = tensors[outputs[0]]["shape"]
            input_scale, input_zero = scalar_quantization(input0, inputs[0])
            output_scale, output_zero = scalar_quantization(output, outputs[0])
            filter_scales, filter_zeros = tensor_quantization(
                graph.tensors[inputs[1]], inputs[1])
            if any(value != 0 for value in filter_zeros):
                fail("FULLY_CONNECTED weights are not symmetric")
            if any(scale != filter_scales[0] for scale in filter_scales):
                fail("current FULLY_CONNECTED kernel requires one weight scale")
            multiplier, shift = quantize_multiplier(
                input_scale * filter_scales[0] / output_scale)
            act_min, act_max = activation_range(
                options.fusedActivationFunction, output, outputs[0])
            record["params_member"] = "fully_connected"
            record["params"] = [
                ("batches", product(input_shape[:-1])),
                ("input_C", filter_shape[-1]), ("output_C", filter_shape[-2]),
                ("input_offset", -input_zero), ("output_offset", output_zero),
                ("activation_min", act_min), ("activation_max", act_max),
            ]
            record["multiplier"] = multiplier
            record["shift"] = shift
            if product(output_shape) != product(input_shape[:-1]) * filter_shape[-2]:
                fail("FULLY_CONNECTED output shape is inconsistent")

        elif code == fb.BuiltinOperator.SOFTMAX:
            # Math:
            #   input_scale * 2^26 = multiplier / 2^31 * 2^left_shift
            #   diff_min = -floor((2^5 - 1) * 2^26 / 2^left_shift)
            #
            # Notes:
            #   Q5.26 leaves five integer bits for the scaled input difference.
            options = op.builtinOptions
            if options.beta != 1.0 or input0.type != fb.TensorType.INT8:
                fail("only beta=1 INT8 SOFTMAX is supported")
            input_scale, _ = scalar_quantization(input0, inputs[0])
            integer_bits = 5
            real_multiplier = min(
                float(options.beta) * input_scale * (1 << (31 - integer_bits)),
                float((1 << 31) - 1),
            )
            multiplier, left_shift = quantize_multiplier(real_multiplier)
            if left_shift < 0:
                fail("SOFTMAX expected a greater-than-one multiplier")
            radius = math.floor(
                ((1 << integer_bits) - 1)
                * (1 << (31 - integer_bits))
                / (1 << left_shift)
            )
            shape = tensors[inputs[0]]["shape"]
            record["params_member"] = "softmax"
            record["params"] = [
                ("outer_size", product(shape[:-1])), ("depth", shape[-1]),
                ("input_multiplier", multiplier),
                ("input_left_shift", left_shift), ("diff_min", -radius),
            ]

        records.append(record)

    return records, op_codes


# B1 implementation ---------------------------------------------------------
def prepare_b1_conv2d(tensors, operators):
    """
    Fold the input offset into each eligible 1x1 Conv2D bias.

    Math:
        sum(filter * (input + input_offset)) + bias
      = sum(filter * input) + input_offset * sum(filter) + bias

    The B1 kernel therefore receives:
        adjusted_bias = bias + input_offset * sum(filter)
        input_offset = 0

    INT64 is used while generating the data.  The two bounds below guarantee
    that neither the original centered-input sum nor the reordered B1 sum can
    overflow an INT32 accumulator for any INT8 input.
    """
    int32_max = np.iinfo(np.int32).max
    adjusted_tensors = set()
    fast_path_count = 0

    for operator in operators:
        if operator["code"] != fb.BuiltinOperator.CONV_2D:
            continue

        params = dict(operator["params"])
        eligible = (
            params["filter_H"] == 1
            and params["filter_W"] == 1
            and params["input_C"] == params["filter_input_C"]
            and params["pad_H"] == 0
            and params["pad_W"] == 0
            and (params["output_C"] >> 2) << 2 == params["output_C"]
        )
        if not eligible:
            continue

        filter_index = operator["inputs"][1]
        bias_index = operator["inputs"][2]
        if bias_index < 0:
            fail(f"B1 CONV_2D {operator['index']} requires a bias tensor")
        if bias_index in adjusted_tensors:
            fail(f"B1 bias tensor {bias_index} is shared by multiple operators")

        weights = np.frombuffer(
            tensors[filter_index]["data"], dtype=np.int8
        ).astype(np.int64).reshape(params["output_C"], -1)
        bias = np.frombuffer(
            tensors[bias_index]["data"], dtype="<i4"
        ).astype(np.int64)
        if bias.size != params["output_C"]:
            fail(f"B1 CONV_2D {operator['index']} has an invalid bias shape")

        input_offset = params["input_offset"]
        adjusted_bias = bias + input_offset * weights.sum(axis=1)

        # Original B0 uses filter * (input + input_offset).
        centered_input_max = max(abs(-128 + input_offset),
                                 abs(127 + input_offset))
        original_bound = (
            np.abs(bias) + np.abs(weights).sum(axis=1) * centered_input_max
        )

        # B1 starts at adjusted_bias and accumulates filter * raw INT8 input.
        reordered_bound = np.abs(adjusted_bias) + np.abs(weights).sum(axis=1) * 128
        if (
            np.any(np.abs(adjusted_bias) > int32_max)
            or np.any(original_bound > int32_max)
            or np.any(reordered_bound > int32_max)
        ):
            fail(f"B1 CONV_2D {operator['index']} may overflow INT32")

        tensors[bias_index]["data"] = adjusted_bias.astype("<i4").tobytes()
        operator["params"] = [
            (name, 0 if name == "input_offset" else value)
            for name, value in operator["params"]
        ]
        adjusted_tensors.add(bias_index)
        fast_path_count += 1

    return fast_path_count


# B3 implementation ---------------------------------------------------------
def validate_b3_requantization(operators):
    """Validate the model-wide contract used by the B3 inline fast path."""
    fast_path_count = 0

    for operator in operators:
        if operator["code"] not in (
            fb.BuiltinOperator.CONV_2D,
            fb.BuiltinOperator.DEPTHWISE_CONV_2D,
        ):
            continue

        params = dict(operator["params"])
        multipliers = operator["multipliers"]
        shifts = operator["shifts"]
        output_c = params["output_C"]

        if params["activation_min"] != -128 or params["activation_max"] != 127:
            fail(f"B3 operator {operator['index']} does not use full INT8 clamp")
        if (multipliers is None or shifts is None
                or len(multipliers) != output_c or len(shifts) != output_c):
            fail(f"B3 operator {operator['index']} has invalid per-channel data")
        if any(multiplier <= 0 for multiplier in multipliers):
            fail(f"B3 operator {operator['index']} has a non-positive multiplier")
        if any(shift < -31 or shift >= 0 for shift in shifts):
            fail(f"B3 operator {operator['index']} requires a negative shift")

        fast_path_count += 1

    return fast_path_count


def generate_header(path, tensors, operators, graph, arena_size):
    """
    Generate C enums, record structures, model sizes and table declarations.

    Notes:
        This file describes the data contract used by the C executor.
    """
    input_index = int(graph.inputs[0])
    output_index = int(graph.outputs[0])
    text = f"""#ifndef EFFICIENTNET_LITE0_DATA_H_
#define EFFICIENTNET_LITE0_DATA_H_

#include <stdint.h>

#include "ca_kernels.h"

/*
 * Quite similar to TFLite graph: 
 *   - ModelOperator order & ModelTensor index order is the same as TFLite graph order.
 *   - TFLite tensor graph (ModelTensor here) stores the address of constant buffers.
 *   - TFLite tensor graph (ModelTensor here) only stores the information of activation, whose data will be allocated during runtime.
 *   - I do what runtime dos by adding coordinate arena address to ModelTensor[index] of activation.
 *
 * ModelOperator
 * ├─ type
 * ├─ input0 ─┐
 * ├─ input1 ─┼──> ModelTensor[index] (Tensor address map)
 * ├─ input2 ─┤         │
 * ├─ output ─┘         ├─> Read-only memory: weight / bias / constant
 * │                    └─> R/W tensor arena: activation
 * └─ params (ex: Conv)
 *      ├─ batches
 *      ├─ input_H
 *      ├─ input_W
 *      ├─ input_C
 *      └─ ...
 */

/* Tensor types */
enum {{
  MODEL_TENSOR_INT8 = 0,
  MODEL_TENSOR_UINT8 = 1,
  MODEL_TENSOR_INT32 = 2,
}};

/* Operator types */
enum {{
  MODEL_OP_QUANTIZE = 0,
  MODEL_OP_CONV_2D = 1,
  MODEL_OP_DEPTHWISE_CONV_2D = 2,
  MODEL_OP_ADD = 3,
  MODEL_OP_AVERAGE_POOL_2D = 4,
  MODEL_OP_RESHAPE = 5,
  MODEL_OP_FULLY_CONNECTED = 6,
  MODEL_OP_SOFTMAX = 7,
}};

/* Tensor address map */
typedef struct {{
  const void *constant_data;  // weight, bias, or other constant data
  uint32_t arena_offset;      // activation
  uint32_t bytes;
  int32_t type;
}} ModelTensor;

typedef struct {{
  int size;
  int input_zero_point;
  int output_zero_point;
  int multiplier;
  int shift;
  int input_type;
  int output_type;
}} ModelQuantizeParams;

typedef struct {{
  int unused;
}} ModelReshapeParams;

typedef struct {{
  int outer_size;
  int depth;
  int input_multiplier;
  int input_left_shift;
  int diff_min;
}} ModelSoftmaxParams;

typedef union {{
  ModelQuantizeParams quantize;
  Conv2DParams conv2d;
  DepthwiseConv2DParams depthwise_conv2d;
  AddParams add;
  AveragePool2DParams average_pool2d;
  ModelReshapeParams reshape;
  FullyConnectedParams fully_connected;
  ModelSoftmaxParams softmax;
}} ModelOperatorParams;

/* One entry for each TFLite operator. */
typedef struct {{
  int32_t type;
  int32_t input0;  // ex: activation in Conv2D
  int32_t input1;  // ex: weight in Conv2D
  int32_t input2;  // ex: bias in Conv2D
  int32_t output;
  const int32_t *multipliers; // per output channel for Conv2D and DepthwiseConv2D
  const int32_t *shifts;
  int32_t multiplier;         // single multiplier for FullyConnected
  int32_t shift;
  ModelOperatorParams params;
}} ModelOperator;

#define EFFICIENTNET_LITE0_TENSOR_COUNT {len(tensors)}    // Total number of tensors in the model
#define EFFICIENTNET_LITE0_OPERATOR_COUNT {len(operators)}
#define EFFICIENTNET_LITE0_INPUT_TENSOR {input_index}    // Input tensor index to the tensor address map
#define EFFICIENTNET_LITE0_OUTPUT_TENSOR {output_index}   // Output tensor index to the tensor address map
#define EFFICIENTNET_LITE0_INPUT_SIZE {tensors[input_index]['bytes']}
#define EFFICIENTNET_LITE0_OUTPUT_SIZE {tensors[output_index]['bytes']}
#define EFFICIENTNET_LITE0_ARENA_SIZE {arena_size}

extern const ModelTensor efficientnet_lite0_tensors[EFFICIENTNET_LITE0_TENSOR_COUNT];
extern const ModelOperator efficientnet_lite0_operators[EFFICIENTNET_LITE0_OPERATOR_COUNT];

#endif
"""
    path.write_text(text, encoding="utf-8")


def generate_source(path, model_path, model_sha256, tensors, operators,
                    variant="B0", align_constants=False):
    """
    Generate all static data needed by the C model.

    Flow:
        Write constant weights and biases
        Write per-channel multiplier and shift arrays
        Write the tensor table with constant addresses or arena offsets
        Write the operator table in TFLite execution order
    """
    lines = [
        '#include "efficientnet_lite0_data.h"',
        "",
        "/* Generated by scripts/cdata_generate.py; do not edit.",
        f" * source: {model_path.as_posix()}",
        f" * sha256: {model_sha256}",
        f" * variant: {variant}",
        " */",
        "",
        "/* Constant tensors copied from the TFLite FlatBuffer. */",
        "",
    ]

    # Weights, bias and other constant tensors.
    constant_names = {}
    for tensor in tensors:
        if not tensor["constant"]:
            continue
        _, c_type, _, dtype = TYPE_INFO[tensor["tensor"].type]
        name = f"tensor_{tensor['index']}_data"
        values = np.frombuffer(tensor["data"], dtype=dtype)
        # X1 implementation: B1 and X1 share the same adjusted-bias data, but
        # X1 must not depend on the linker accidentally aligning its weights.
        alignment = 8 if align_constants else None
        lines.extend(format_array(values, c_type, name, alignment))
        constant_names[tensor["index"]] = name

    lines.extend([
        "/* Per-output-channel requantization parameters prepared offline. */",
        "",
    ])

    # Per-output-channel multiplier/shift arrays.
    for operator in operators:
        if operator["multipliers"] is not None:
            values = np.asarray(operator["multipliers"], dtype=np.int32)
            lines.extend(format_array(values, "int32_t",
                                      f"op_{operator['index']}_multipliers"))
        if operator["shifts"] is not None:
            values = np.asarray(operator["shifts"], dtype=np.int32)
            lines.extend(format_array(values, "int32_t",
                                      f"op_{operator['index']}_shifts"))

    # Tensor table: each entry selects constant data or one arena address.
    lines.extend([
        "/* Constant addresses and planned activation-arena offsets. */",
        "const ModelTensor efficientnet_lite0_tensors[] = {",
    ])
    for tensor in tensors:
        data = constant_names.get(tensor["index"], "0")
        tensor_type = TYPE_INFO[tensor["tensor"].type][0]
        lines.extend([
            f"  [{tensor['index']}] = {{",
            f"    .constant_data = {data},",
            f"    .arena_offset = {tensor['arena_offset']},",
            f"    .bytes = {tensor['bytes']},",
            f"    .type = {tensor_type},",
            "  },",
        ])
    lines.extend(["};", ""])

    # Operator table: the C executor walks this array from first to last.
    lines.extend([
        "/* Operators remain in the execution order stored by TFLite. */",
        "const ModelOperator efficientnet_lite0_operators[] = {",
    ])
    for operator in operators:
        multiplier_ptr = (f"op_{operator['index']}_multipliers"
                          if operator["multipliers"] is not None else "0")
        shift_ptr = (f"op_{operator['index']}_shifts"
                     if operator["shifts"] is not None else "0")
        lines.extend([
            f"  [{operator['index']}] = {{",
            f"    .type = {operator['type']},",
            f"    .input0 = {operator['inputs'][0]},",
            f"    .input1 = {operator['inputs'][1]},",
            f"    .input2 = {operator['inputs'][2]},",
            f"    .output = {operator['output']},",
            f"    .multipliers = {multiplier_ptr},",
            f"    .shifts = {shift_ptr},",
            f"    .multiplier = {operator['multiplier']},",
            f"    .shift = {operator['shift']},",
            f"    .params.{operator['params_member']} = {{",
        ])
        for name, value in operator["params"]:
            lines.append(f"      .{name} = {value},")
        lines.extend(["    },", "  },"])
    lines.extend(["};", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--output-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--b1", action="store_true",
        help="generate adjusted biases and zero input offsets for B1 1x1 Conv2D",
    )
    parser.add_argument(
        "--b3", action="store_true",
        help="validate Conv2D/DepthwiseConv2D for B3 inline requantization",
    )
    args = parser.parse_args()

    if args.b3 and not args.b1:
        fail("--b3 requires --b1 because B3 keeps the X1 Conv2D dataflow")

    # Load .tflite
    model, raw = load_model(args.model)
    graph = model.subgraphs[0]

    # Prepare everything normally calculated by TFLite at runtime.
    tensors = make_tensor_records(model, graph)
    operators, op_codes = build_operator_records(model, graph, tensors)
    arena_size = plan_arena(graph, tensors, op_codes)

    # B1 implementation: B0 remains the default generated-data format.
    b1_fast_paths = prepare_b1_conv2d(tensors, operators) if args.b1 else 0

    # B3 implementation: validate once here, not inside every output element.
    b3_fast_paths = validate_b3_requantization(operators) if args.b3 else 0

    # Check: .tflite I/O 
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        fail("expected exactly one model input and output")
    input_tensor = tensors[int(graph.inputs[0])]["tensor"]
    output_tensor = tensors[int(graph.outputs[0])]["tensor"]
    if input_tensor.type != fb.TensorType.UINT8 or output_tensor.type != fb.TensorType.UINT8:
        fail("expected uint8 model input and output")

    # Output C files
    # Inference only needs these generated files; it does not parse TFLite.
    args.output_header.parent.mkdir(parents=True, exist_ok=True)
    args.output_source.parent.mkdir(parents=True, exist_ok=True)
    generate_header(args.output_header, tensors, operators, graph, arena_size)
    variant = "B1 adjusted-bias"
    if args.b3:
        variant += " + B3 validated requantization"
    elif not args.b1:
        variant = "B0"
    generate_source(args.output_source, args.model,
                    hashlib.sha256(raw).hexdigest(), tensors, operators,
                    variant,
                    align_constants=args.b1)

    constant_bytes = sum(tensor["bytes"] for tensor in tensors
                         if tensor["constant"])
    print(f"generated {args.output_header.relative_to(ROOT)}")
    print(f"generated {args.output_source.relative_to(ROOT)}")
    print(f"model: {len(tensors)} tensors, {len(operators)} operators")
    print(f"constant data: {constant_bytes} bytes")
    print(f"tensor arena: {arena_size} bytes")
    if args.b1:
        print(f"B1 1x1 Conv2D fast paths: {b1_fast_paths}")
    if args.b3:
        print(f"B3 requantization fast paths: {b3_fast_paths}")


if __name__ == "__main__":
    main()
