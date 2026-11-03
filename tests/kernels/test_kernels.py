#!/usr/bin/env python3
"""
    A testbench for our C kernels against TFLite reference kernels.
    Folw:
        Prepare a corner testcase
        Send the same tensors to our kernels and TFLite
        Compare bytes
"""

import ctypes as C
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OURS = C.CDLL(str(ROOT / "build/libca_kernels.so"))
TFLITE = C.CDLL(str(ROOT / "build/libtflite_reference.so"))


# These structures must have the same field order as src/include/ca_kernels.h.
def c_int_fields(names):
    return [(name, C.c_int) for name in names.split()]


class Conv2DParams(C.Structure):
    _fields_ = c_int_fields(
        "batches input_H input_W input_C filter_H filter_W filter_input_C "
        "output_H output_W output_C stride_H stride_W dilation_H dilation_W "
        "pad_H pad_W input_offset output_offset activation_min activation_max"
    )


class DepthwiseConv2DParams(C.Structure):
    _fields_ = c_int_fields(
        "batches input_H input_W input_C filter_H filter_W output_H output_W "
        "output_C depth_multiplier stride_H stride_W dilation_H dilation_W "
        "pad_H pad_W input_offset output_offset activation_min activation_max"
    )


class AddParams(C.Structure):
    _fields_ = c_int_fields(
        "size input1_offset input2_offset output_offset left_shift "
        "input1_multiplier input1_shift input2_multiplier input2_shift "
        "output_multiplier output_shift activation_min activation_max"
    )


class AveragePool2DParams(C.Structure):
    _fields_ = c_int_fields(
        "batches input_H input_W input_C output_H output_W filter_H filter_W "
        "stride_H stride_W pad_H pad_W activation_min activation_max"
    )


class FullyConnectedParams(C.Structure):
    _fields_ = c_int_fields(
        "batches input_C output_C input_offset output_offset "
        "activation_min activation_max"
    )


def pointer(array, c_type):
    return array.ctypes.data_as(C.POINTER(c_type))


def quantize_multiplier(real_multiplier):
    """real_multiplier ~= multiplier / 2^31 * 2^shift."""
    significand, shift = math.frexp(real_multiplier)
    multiplier = math.floor(significand * (1 << 31) + 0.5)
    if multiplier == 1 << 31:
        multiplier //= 2
        shift += 1
    return multiplier, shift


def corner(size, dtype=np.int8, zero_point=0, reverse=False):
    """Repeat min, max, zero-point, and their neighboring values for testcases generation."""
    limits = np.iinfo(dtype)
    pattern = np.array([
        limits.min, limits.min + 1, zero_point - 1, zero_point,
        zero_point + 1, limits.max - 1, limits.max,
    ])
    pattern = np.clip(pattern, limits.min, limits.max).astype(dtype)
    if reverse:
        pattern = pattern[::-1]
    return np.ascontiguousarray(np.resize(pattern, size))


def byte_exact(name, ours, tflite):
    """Raise AssertionError if the two arrays differ, otherwise PASS."""
    if np.array_equal(ours, tflite):
        print(f"PASS  {name}")
        return

    different = np.flatnonzero(ours != tflite)
    first = int(different[0])
    raise AssertionError(
        f"{name}: {different.size} byte(s) differ; first index {first}: "
        f"C={ours[first]}, TFLite={tflite[first]}"
    )


def test_quantize():
    # uint8 -> int8
    # Input preparation
    source = corner(37, np.uint8, zero_point=131)
    multiplier, shift = quantize_multiplier(0.75)
    arguments = (37, 131, 3, multiplier, shift)

    # Output storage
    ours = np.empty(37, np.int8)
    tflite = np.empty(37, np.int8)

    # Run 
    OURS.quantize_u8_to_s8(pointer(source, C.c_uint8), pointer(ours, C.c_int8), *arguments)
    TFLITE.tflite_ref_quantize_u8_to_s8(pointer(source, C.c_uint8), pointer(tflite, C.c_int8), *arguments)
    byte_exact("QUANTIZE uint8 -> int8", ours, tflite)

    # int8 -> uint8
    # Input preparation
    source = corner(37, np.int8, zero_point=-128)
    multiplier, shift = quantize_multiplier(1.0)
    arguments = (37, -128, 0, multiplier, shift)

    # Output storage
    ours = np.empty(37, np.uint8)
    tflite = np.empty(37, np.uint8)

    # Run
    OURS.quantize_s8_to_u8(pointer(source, C.c_int8), pointer(ours, C.c_uint8), *arguments)
    TFLITE.tflite_ref_quantize_s8_to_u8(pointer(source, C.c_int8), pointer(tflite, C.c_uint8), *arguments)
    byte_exact("QUANTIZE int8 -> uint8", ours, tflite)


def test_conv2d():
    # Input Preparation
    # N/H/W/C, filter H/W/input C, output H/W/C, stride H/W, dilation H/W,
    # pad H/W, offsets, activation clamp.
    p = Conv2DParams(1, 4, 5, 2, 3, 3, 2, 2, 3, 3,
                     2, 2, 1, 1, 0, 1, 11, -7, -100, 100)
    source = corner(p.batches * p.input_H * p.input_W * p.input_C,
                    zero_point=-p.input_offset)
    weights = corner(p.output_C * p.filter_H * p.filter_W * p.filter_input_C,
                     reverse=True)
    bias = np.arange(p.output_C, dtype=np.int32) * 97 - 101
    scales = [quantize_multiplier(scale)
              for scale in np.linspace(0.017, 0.41, p.output_C)]
    multipliers = np.array([value[0] for value in scales], np.int32)
    shifts = np.array([value[1] for value in scales], np.int32)
    inputs = (pointer(source, C.c_int8), pointer(weights, C.c_int8),
              pointer(bias, C.c_int32), pointer(multipliers, C.c_int32),
              pointer(shifts, C.c_int32))

    # Output storage
    ours = np.empty(p.batches * p.output_H * p.output_W * p.output_C, np.int8)
    tflite = np.empty_like(ours)

    # Run
    OURS.conv2d_s8(*inputs, pointer(ours, C.c_int8), C.byref(p))
    TFLITE.tflite_ref_conv2d_s8(*inputs, pointer(tflite, C.c_int8), C.byref(p))
    byte_exact("CONV_2D 3x3 stride-2 SAME", ours, tflite)


def test_depthwise_conv2d():
    # Input Preparation
    # N/H/W/C, filter H/W, output H/W/C, depth multiplier,
    # stride H/W, dilation H/W, pad H/W, offsets, activation clamp.
    p = DepthwiseConv2DParams(1, 5, 6, 2, 5, 5, 3, 3, 2, 1,
                              2, 2, 1, 1, 2, 2, -7, 5, -128, 127)
    source = corner(p.batches * p.input_H * p.input_W * p.input_C,
                    zero_point=-p.input_offset)
    weights = corner(p.filter_H * p.filter_W * p.output_C, reverse=True)
    bias = np.arange(p.output_C, dtype=np.int32) * 53 - 71
    scales = [quantize_multiplier(scale) for scale in np.linspace(0.013, 0.29, p.output_C)]
    multipliers = np.array([value[0] for value in scales], np.int32)
    shifts = np.array([value[1] for value in scales], np.int32)
    inputs = (pointer(source, C.c_int8), pointer(weights, C.c_int8),
              pointer(bias, C.c_int32), pointer(multipliers, C.c_int32),
              pointer(shifts, C.c_int32))
    
    # Output storage
    ours = np.empty(p.batches * p.output_H * p.output_W * p.output_C, np.int8)
    tflite = np.empty_like(ours)

    # Run
    OURS.depthwise_conv2d_s8(*inputs, pointer(ours, C.c_int8), C.byref(p))
    TFLITE.tflite_ref_depthwise_conv2d_s8(*inputs, pointer(tflite, C.c_int8), C.byref(p))
    byte_exact("DEPTHWISE_CONV_2D 5x5 stride-2 SAME", ours, tflite)


def test_add():
    # Input Preparation
    size = 41
    scale1, scale2, output_scale = 0.031, 0.047, 0.052
    common_scale = 2.0 * max(scale1, scale2)
    m1, s1 = quantize_multiplier(scale1 / common_scale)
    m2, s2 = quantize_multiplier(scale2 / common_scale)
    mo, so = quantize_multiplier(common_scale / ((1 << 20) * output_scale))
    # size, three offsets, left shift, three multiplier/shift pairs, clamp.
    p = AddParams(size, 12, -7, -3, 20, m1, s1, m2, s2, mo, so, -120, 115)
    input1 = corner(size, zero_point=-p.input1_offset)
    input2 = corner(size, zero_point=-p.input2_offset, reverse=True)
    inputs = (pointer(input1, C.c_int8), pointer(input2, C.c_int8), C.byref(p))

    # Output storage
    ours = np.empty(size, np.int8)
    tflite = np.empty_like(ours)

    # Run
    OURS.add_s8(*inputs, pointer(ours, C.c_int8))
    TFLITE.tflite_ref_add_s8(*inputs, pointer(tflite, C.c_int8))
    byte_exact("ADD mixed scales", ours, tflite)


def test_average_pool2d():
    # Input Preparation
    # Input N/H/W/C, output H/W, filter H/W, stride H/W, pad H/W, clamp.
    p = AveragePool2DParams(1, 4, 5, 3, 2, 3, 3, 3, 2, 2, 1, 1, -100, 100)
    source = corner(p.batches * p.input_H * p.input_W * p.input_C)

    # Output storage
    ours = np.empty(p.batches * p.output_H * p.output_W * p.input_C, np.int8)
    tflite = np.empty_like(ours)

    OURS.average_pool2d_s8(pointer(source, C.c_int8), pointer(ours, C.c_int8), C.byref(p))
    TFLITE.tflite_ref_average_pool2d_s8(pointer(source, C.c_int8), pointer(tflite, C.c_int8), C.byref(p))
    byte_exact("AVERAGE_POOL_2D padded windows", ours, tflite)


def test_fully_connected():
    # Input Preparation
    # Batches, input/output channels, input/output offsets, clamp.
    p = FullyConnectedParams(2, 7, 5, 17, -11, -128, 127)
    source = corner(p.batches * p.input_C, zero_point=-p.input_offset)
    weights = corner(p.output_C * p.input_C, reverse=True)
    bias = np.arange(p.output_C, dtype=np.int32) * 113 - 207
    input_scale = np.float32(0.023529412)
    output_scale = np.float32(0.09174982)
    weight_scales = np.full(p.output_C, np.float32(0.00381672))
    multiplier, shift = quantize_multiplier(float(input_scale * weight_scales[0] / output_scale))
    common = (pointer(source, C.c_int8), pointer(weights, C.c_int8), pointer(bias, C.c_int32))
    
    # Output storage
    ours = np.empty(p.batches * p.output_C, np.int8)
    tflite = np.empty_like(ours)

    # Run
    OURS.fully_connected_s8(
        *common, multiplier, shift, pointer(ours, C.c_int8), C.byref(p)
    )
    TFLITE.tflite_ref_fully_connected_s8(
        *common, C.c_float(input_scale), C.c_float(output_scale),
        pointer(weight_scales, C.c_float), pointer(tflite, C.c_int8), C.byref(p)
    )
    byte_exact("FULLY_CONNECTED", ours, tflite)


def test_softmax():
    # Input Preparation
    depth = 17
    source = np.full((3, depth), -128, np.int8)
    source[0] = corner(depth)
    source[1, depth // 2] = 127        # one dominant class
    source[2, depth // 3] = 127        # two equally dominant classes
    source[2, 2 * depth // 3] = 127
    source = source.reshape(-1)

    input_scale = 0.09174982
    multiplier, left_shift = quantize_multiplier(input_scale * (1 << 26))
    diff_min = -math.floor(31.0 * (1 << 26) / (1 << left_shift))
    parameters = (3, depth, multiplier, left_shift, diff_min)

    # Output storage
    ours = np.empty_like(source)
    tflite = np.empty_like(source)

    # Run
    OURS.softmax_s8(pointer(source, C.c_int8), pointer(ours, C.c_int8), *parameters)
    TFLITE.tflite_ref_softmax_s8(pointer(source, C.c_int8), pointer(tflite, C.c_int8), *parameters)
    byte_exact("SOFTMAX corners + one peak + two peaks", ours, tflite)


def main():
    test_quantize()
    test_conv2d()
    test_depthwise_conv2d()
    test_add()
    test_average_pool2d()
    test_fully_connected()
    test_softmax()
    print("All 7 kernel types are byte-exact with TFLite C++ reference kernels.")


if __name__ == "__main__":
    main()
