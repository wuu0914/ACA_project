# B3 / X3 Requantization design considerations


- B2：Phase 4 已量測的 RV64IMC + Xqmac8 + Depthwise fast path baseline。
- B3：只使用既有 RV64IMC 指令，針對目前模型的 Conv2D / DepthwiseConv2D requantization 建立專用 fast path。
- X3：延續 B3，只加入硬體簡單的 Xclip8 處理 signed INT8 clamp。



## 0. B2 requantization profile

B2 的實測結果為：

```text
B2 全模型 instructions：                    983,811,822
output_scaling_round2 instructions：         263,126,439
output_scaling_round2 calls：                  7,519,560
平均 function body：                              34.99 instructions / call
```

動態 calls 來源為：

| Operator | Calls | 比例 |
| --- | ---: | ---: |
| Conv2D | 4,431,168 | 58.93% |
| DepthwiseConv2D | 2,306,528 | 30.67% |
| ADD | 630,336 | 8.38% |
| Quantize | 151,528 | 2.02% |

B3 先只處理 Conv2D 與 DepthwiseConv2D，共：

```text
4,431,168 + 2,306,528 = 6,737,696 calls
```

盤點所有 49 組 per-channel parameters 後，全部符合：

```text
multiplier > 0
shift = -29 .. -2
activation_min = -128
activation_max = 127
```

因此這兩個 operators 不需要在每個 output 重複處理：

- positive shift / left shift；
- `INT32_MIN * INT32_MIN` saturation；
- runtime activation range。

ADD 有 `shift == 0`，Quantize 也有 positive shift，因此它們仍走原本的通用
`output_scaling_round2()`。FullyConnected 使用 `output_scaling_round1()`，不屬於 B3。




## 1. B3 優化前：B2 完整 output fast path

目前 Conv2D 與 DepthwiseConv2D 在 accumulator 完成後，執行以下 C path。這裡列出
完整 helpers，因為 GCC 目前不會跨 translation unit inline
`output_scaling_round2()`。

```c
static int32_t ab_fxp_mul_round(int32_t a, int32_t b) {
  if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;

  const int64_t product = (int64_t)a * (int64_t)b;
  const int64_t nudge = product >= 0 ? (INT64_C(1) << 30) : 1 - (INT64_C(1) << 30);
  return (int32_t)((product + nudge) / (INT64_C(1) << 31));
}

static int32_t right_shift_round(int32_t value, int exponent) {
  if (exponent == 0) return value;

  const uint32_t mask = (UINT32_C(1) << exponent) - 1;
  const int32_t remainder = (int32_t)((uint32_t)value & mask);
  const int32_t threshold = (int32_t)(mask >> 1) + (value < 0);
  return (value >> exponent) + (remainder > threshold);
}

int32_t output_scaling_round2(int32_t value,
                              int32_t multiplier,
                              int shift) {
  const int left_shift = shift > 0 ? shift : 0;
  const int right_shift = shift > 0 ? 0 : -shift;
  const int32_t shifted = value * (1 << left_shift);
  return right_shift_round(ab_fxp_mul_round(shifted, multiplier), right_shift);
}

static int32_t clamp(int32_t value, int32_t low, int32_t high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

static int8_t requantize(int32_t acc,
                         int output_c,
                         const int32_t *output_multipliers,
                         const int32_t *output_shifts,
                         const Conv2DParams *p) {
  acc = output_scaling_round2(acc, output_multipliers[output_c], output_shifts[output_c]);
  acc = clamp(acc + p->output_offset, p->activation_min, p->activation_max);
  return (int8_t)acc;
}

output[output_index + output_c] = requantize(acc, output_c, output_multipliers, output_shifts, p);
```

DepthwiseConv2D 使用相同欄位與計算順序，只是 parameter type 是
`DepthwiseConv2DParams`。

### RV64IMC instructions

以下是目前 `rv64imc -O2` 的實際結構。
```asm
# Caller: load per-channel parameters and call the shared function
lw      a2, 0(shift_pointer)       # a2 = this channel's shift
lw      a1, 0(multiplier_pointer)  # a1 = this channel's Q0.31 multiplier
jal     output_scaling_round2      # a0 = scaled result after returning

# output_scaling_round2: general shift selection
mv      a5, a2                     # Start by assuming left_shift = shift
bgez    a2, positive_shift         # shift >= 0: keep that left shift
li      a5, 0                      # shift < 0: left_shift = 0
positive_shift:
sllw    a5, a0, a5                # shifted = value << left_shift

# General INT32_MIN * INT32_MIN saturation check
lui     a0, 0x80000                # a0 = INT32_MIN
sub     a3, a5, a0                # a3 = shifted - INT32_MIN
sub     a4, a1, a0                # a4 = multiplier - INT32_MIN
seqz    a3, a3                    # a3 = (shifted == INT32_MIN)
seqz    a4, a4                    # a4 = (multiplier == INT32_MIN)
and     a4, a4, a3                # a4 = both operands are INT32_MIN
not     a0, a0                    # a0 = INT32_MAX, the saturation result
bnez    a4, multiply_done          # Skip multiply when the special case occurs

# Round((value * multiplier) / 2^31)
mul     a5, a5, a1                # a5 = 64-bit product
lui     a4, 0xc0000                # a4 = -2^30
addi    a4, a4, 1                 # Negative-product nudge = 1 - 2^30
bltz    a5, negative_product       # product < 0: keep the negative nudge
lui     a4, 0x40000                # Positive-product nudge = 2^30
negative_product:
add     a5, a5, a4                # a5 = product + nudge
srai    a0, a5, 63                # All ones when the nudged product is negative
srli    a0, a0, 33                # Division correction = 2^31 - 1 or 0
add     a0, a0, a5                # Correct signed division toward zero
srai    a0, a0, 31                # Divide by 2^31
sext.w  a0, a0                    # q31 = signed INT32 result

# Round(q31 / 2^right_shift)
multiply_done:
bgez    a2, scaling_done           # shift >= 0 means right_shift = 0
negw    a2, a2                     # a2 = right_shift = -shift
li      a5, 1                      # Base value used to construct the mask
sllw    a5, a5, a2                # a5 = 2^right_shift
addiw   a5, a5, -1                # mask = 2^right_shift - 1
srliw   a3, a5, 1                 # a3 = mask / 2
srliw   a4, a0, 31                # a4 = (q31 < 0), either 0 or 1
and     a5, a5, a0                # a5 = discarded bits, the remainder
addw    a4, a4, a3                # threshold = mask / 2 + (q31 < 0)
sraw    a0, a0, a2                # Base result = q31 >> right_shift
slt     a4, a4, a5                # Round bit = (remainder > threshold)
addw    a0, a0, a4                # Add the round bit to the base result
scaling_done:
ret                                # Return the scaled INT32 value in a0

# Caller: output offset, clamp and store
lw      a3, output_offset(p)       # Load the operator-wide output zero point
lw      a4, activation_min(p)      # a4 defaults to the lower clamp bound
addw    a3, a3, a0                # Add output_offset to the scaled value
blt     a3, a4, store              # Below minimum: store activation_min
lw      a2, activation_max(p)      # Load the upper clamp bound
sext.w  a4, a2                     # a4 defaults to activation_max
bge     a3, a2, store              # At/above maximum: store activation_max
sext.w  a4, a3                     # In range: store the unclamped value
store:
sb      a4, 0(output_pointer)      # Store the final signed INT8 output
```

常見未飽和路徑約為：

```text
2 parameter loads
1 function call
35～36 output_scaling_round2 body instructions
9 output-offset / clamp / store instructions
------------------------------------------------
47～48 instructions / output
```

實際數量會隨 product 正負與 clamp branch 改變；B2 Spike profile 量到
`output_scaling_round2()` function body 的全模型平均是 34.99 instructions/call。




## 2. B3 優化後：negative-shift INT8 fast path

B3 把目前模型不會使用的通用分支移除，並改寫成適合 RV64 算術右移的等價形式。


完整 C fast path 為：

```c
static inline int8_t b3_requantize_s8(
    int32_t acc,
    int32_t multiplier,
    int shift,
    int32_t output_offset) {
  const int right_shift = -shift;

  /* 1. TFLite SaturatingRoundingDoublingHighMul without its unreachable INT32_MIN * INT32_MIN case.  
   * 
   * 2. "Round to nearest, ties toward +∞" can be simplified.
   *     Original: 
   *       nudge = product >= 0 ? (1LL << 30) : (1 - (1LL << 30));
   *       q31 = (product + nudge) / (1LL << 31);
   *     Optimized:
   *       q31 = (int32_t)((product + (INT64_C(1) << 30)) >> 31);
   */
  const int64_t product = (int64_t)acc * multiplier;
  const int32_t q31 = (int32_t)((product + (INT64_C(1) << 30)) >> 31);

  /* 1. Equivalent to TFLite RoundingDivideByPOT.  Subtracting one for a negative
   *
   * 2. "Round to nearest, ties away from zero" can be simplified.
   *     Original:
   *       const uint32_t mask = (UINT32_C(1) << right_shift) - 1;
   *       const int32_t remainder = (int32_t)((uint32_t)value & mask);
   *       const int32_t threshold = (int32_t)(mask >> 1) + (value < 0);
   *       return (value >> right_shift) + (remainder > threshold);
   *     Optimized:
   *       const int64_t half = INT64_C(1) << (right_shift - 1);
   *       const int32_t scaled = (int32_t)(((int64_t)q31 + half - (q31 < 0)) >> right_shift);
   */
  const int64_t half = INT64_C(1) << (right_shift - 1);
  const int32_t scaled = (int32_t)(((int64_t)q31 + half - (q31 < 0)) >> right_shift);

  int32_t output_value = scaled + output_offset;
  if (output_value < -128) output_value = -128;
  if (output_value > 127) output_value = 127;
  return (int8_t)output_value;
}

output[output_index + output_c] = b3_requantize_s8(acc0, output_multipliers[output_c], output_shifts[output_c], output_offset);
```

### RV64IMC instructions

用專案相同的 GCC 16.1.0、`-march=rv64imc -mabi=lp64 -O2` 獨立編譯四個
inline outputs，GCC 沒有產生 stack spill。以下是一個未飽和 output 的完整路徑：

```asm
lw      t3, 0(multiplier_pointer)  # t3 = multiplier
li      t6, 1073741824             # 2^30
li      t1, 1                      # 1
mul     t3, t3, acc                # multiplier * acc
lw      a0, 0(shift_pointer)       # a0 = shift
li      t5, 127                    # Signed INT8 upper bound
not     t4, a0                     # t4 = -shift - 1
sll     t1, t1, t4                 # half = 1 << (right_shift - 1)
negw    t4, a0                     # t4 = right_shift = -shift
add     t3, t3, t6                 # product + 2^30
srai    t3, t3, 31                 # Arithmetic divide by 2^31
sext.w  t3, t3                     # t3 = q31 as a signed INT32
add     a0, t1, t3                 # a0 = q31 + half
srli    t3, t3, 63                 # t3 = (q31 < 0)
sub     a0, a0, t3                 # q31 + half - (q31 < 0)
sra     a0, a0, t4                 # a0 = scaled 
addw    a0, a0, output_offset      # scaled + offset
mv      t1, a0                     
ble     a0, t5, max_done           # value <= 127
mv      t1, t5                     # Only executed when output > 127
max_done:
sext.w  a0, t1                     # Sign-extend the upper-clamped candidate
li      t3, -128                   # Signed INT8 lower bound
bge     a0, t3, min_done           # value >= -128: lower clamp is unnecessary
mv      t1, t3                     # Only executed when output < -128
min_done:
sb      t1, 0(output_pointer)      # Store the final signed INT8 output
```

未飽和路徑為：

```text
2 parameter loads
13 multiply + two-rounding instructions
7 output-offset + clamp instructions
1 output store
------------------------------------------
23 instructions / output
```

若 output 超過其中一側的 INT8 範圍，會多執行一個 `mv`，因此是 24 instructions。
四個 outputs 完整 inline 後，常見動態路徑是 `23 instructions`；獨立
probe function 另有一個最後的 `ret`。




## 3. 為什麼 B3 不 pack multiplier 與 shift

目前每個 channel 使用：

```text
int32 multiplier + int32 shift = 8 bytes
```

改成一個 `uint64_t` 仍是 8 bytes，不節省常數空間。它也不會直接省下一條 RV64
instruction，因為 `ld` 之後必須再拆出兩個 signed INT32 values：

```asm
ld      parameter, 0(pointer)
sext.w  multiplier, parameter
srai    shift, parameter, 32
```

實際四-output probe 結果為：

| Parameter layout | 常見 instructions / 4 outputs，包含 probe 的 `ret` |
| --- | ---: |
| 兩個 `int32_t` arrays | 93 |
| packed `uint64_t` array | 97 |

packed 版本由 8 次 loads 降到 4 次 loads，但增加 8 次 extraction instructions，總
retired instructions 反而多 4。因此 B3 保留現有兩個 arrays。




## 4. B3 instruction-count 試算

只看 scaling，B2 常見的：

```text
jal + 35-instruction function body
```

會變成 13 條 inline arithmetic instructions，保守省下：

```text
1 + 35 - 13 = 23 instructions / output
```

先不把 constant clamp 可能額外省下的 instruction 算進去：

```text
B3 影響的 Conv2D + Depthwise outputs
    = 6,737,696

保守節省
    = 6,737,696 * 23
    = 154,967,008 instructions

預估 B3 全模型
    = 983,811,822 - 154,967,008
    = 828,844,814 instructions

相對 B2
    = -15.75%

預估 MACs / retired instruction
    = 385,187,552 / 828,844,814
    = 0.464728
```

若 integrated assembly 也維持 probe 的完整常見路徑 `47 -> 23`，則預估為：

```text
983,811,822 - 6,737,696 * 24
= 822,107,118 instructions
= -16.44% relative to B2
```

因此 B3 合理的實作前預測為：

| 版本 | 預估全模型 instructions | 相對 B2 | MACs / instruction |
| --- | ---: | ---: | ---: |
| B2 實測 | 983,811,822 | - | 0.391526 |
| B3 保守預測 | 828,844,814 | -15.75% | 0.464728 |
| B3 常見路徑預測 | 822,107,118 | -16.44% | 0.468537 |

這是 retired-instruction 預測，不是 cycle 預測。inline 後的 register pressure、branch
結果與未來處理器的 multiplier latency，都必須在實作後另外量測。




## 5. X3：以 Xclip8 處理 signed INT8 clamp

### Midterm 前的設計取捨

量化相關程式中，`output_scaling_round2()` 在 B2 佔全模型 26.75% 相對較少，且增加指令後雖然
retired instructions 可能下降更多，但我擔心增加太多 area 與影響 timing，需要在實際設計 RTL 時有通盤的考量。
因此 midterm 前的目標是：

- 希望不增加太多複雜硬體的情況下做優化。
- 只實作硬體簡單、容易驗證且 timing 風險較低的 Xclip8；
- 實際設計 RTL 時仍要一起檢查 decode、area、critical path 與整體效能，不能只看
  retired-instruction reduction。


### Xclip8 instruction contract

```text
xclip8 rd, rs1

value = INT32(rs1)

if value > 127:
    result = 127
else if value < -128:
    result = -128
else:
    result = value

rd = sign_extend_64(INT8(result))
```

Xclip8 只負責 signed INT8 clamp，不包含 output offset 的加法。B3 原本最後一段：

```asm
li      t5, 127
addw    a0, a0, output_offset
mv      t1, a0
ble     a0, t5, max_done
mv      t1, t5
max_done:
sext.w  a0, t1
li      t3, -128
bge     a0, t3, min_done
mv      t1, t3
min_done:
sb      t1, 0(output_pointer)
```

在 X3 變成：

```asm
addw    a0, a0, output_offset  # scaled + output_offset
xclip8  a0, a0                 # clamp(a0, -128, 127)
sb      a0, 0(output_pointer)   # Store signed INT8 output
```


### 最小硬體需求

對 `value[31:0]` 而言：

```text
value[31:7] 全為 0  -> value 是   0 .. 127
value[31:7] 全為 1  -> value 是 -128 ..  -1
其他且 sign = 0     -> 輸出 127
其他且 sign = 1     -> 輸出 -128
```

因此 Xclip8 只需要：

- upper bits 的 all-zero / all-one detection；
- 在原值、`127`、`-128` 之間選擇的小型 mux；
- custom instruction decode 與 writeback control。

它不需要新增 multiplier、adder 或 variable shifter。


### Instruction-count 試算

B3 實作後量到 851,551,446 instructions。這比原本的 common-path prediction 多
29,444,328 instructions，主要是 inline 後增加的 register pressure 產生 loop-state stack traffic。因此 X3 改用 B3 實測結果當 baseline。

B3 常見路徑中，output offset 與 clamp 共 7 instructions；X3 使用 `addw + xclip8`
後只剩 2 instructions，因此先只計算每個 output 能直接省下的 5 instructions：

```text
節省
    = 6,737,696 * 5
    = 33,688,480 instructions

預估 X3 全模型
    = 851,551,446 - 33,688,480
    = 817,862,966 instructions
```

| 版本 | 全模型 instructions | 相對 B2 | 相對 B3 實測 |
| --- | ---: | ---: | ---: |
| B2 實測 | 983,811,822 | - | - |
| B3 實測 | 851,551,446 | -13.44% | - |
| X3 預測 | 817,862,966 | -16.87% | -3.96% |


## 6. B3 / X3 correctness and measurement gates

1. 通用 `output_scaling_round2()` 保留給 ADD、Quantize 與任何不符合 B3 contract 的
   operator。
2. Generator 必須為每個 Conv2D / DepthwiseConv2D operator 驗證 multiplier、shift
   與 activation range；不符合 contract 時使用原本 path，不能在每個 output 內重複
   執行這些檢查。
3. Host test 必須確認所有 intermediate outputs 與 TFLite byte-exact。
4. Spike final output 的 1000 bytes 必須 byte-exact。
5. objdump 必須確認四個 outputs 已 inline、沒有 `jal output_scaling_round2`，也沒有
   accumulator 或 requantization temporary stack spills。
6. Xclip8 必須針對 `INT32_MIN`、`-129`、`-128`、`-1`、`0`、`127`、`128`、
   `INT32_MAX` 驗證 instruction contract。
7. X3 必須維持 Host intermediate outputs 與 Spike final output byte-exact。
8. RTL 完成後必須比較加入 Xclip8 前後的 area 與 timing report，再判斷它是否符合
   midterm 的低硬體複雜度目標。
