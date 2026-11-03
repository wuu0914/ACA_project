# B1 / X1 Conv2D design considerations


- B0：Phase 2 已量測的 RV64IMC baseline。
- B1：根據 Efficient lite0 Conv2d 特性，預先算 `sum(filter) * offset + bias` 以及使用 input stationary 優化 Conv2D。
- X1：延續 B1 ，再把八組 INT8 MAC 換成 Xqmac8。

## 0. 觀察

Phase 2 的 B0 RV64IMC baseline 實測結果為：

```text
B0 全模型 instructions：         4,449,648,842
conv2d_s8 instructions：         3,242,185,190  (72.86%)
Conv2D estimated MACs：            349,375,488  (90.70%)
```

Conv2D 同時占最多 MACs 與最多 retired instructions，因此是第一個必須優化的
operator。更重要的是，模型的 33 個 Conv2D 中只有第一層是 3x3；其餘 32 個都是
1x1 Conv，共占 `338,537,472` MACs。B0 卻仍在每個 output 重複執行泛用 Conv 的
filter loops、padding checks、完整 index 計算與 input offset 加法。

盤點這 32 個 1x1 Conv 後，全部符合：

```text
filter_H == 1
filter_W == 1
input_C == filter_input_C
pad_H == 0
pad_W == 0
output_C % 4 == 0
input_C % 8 == 0
filter zero point == 0
```

因此可以分兩層優化：

- B1 在 generate-time 將 `input_offset * sum(filter)` 折進 bias，runtime 只計算
  raw INT8 dot product；再以 input stationary dataflow 讓一個 input 同時更新四個
  output-channel accumulators，提高 registers utility。
- X1 保留相同 dataflow，但把連續八個 scalar INT8 MAC 合併成一個 Xqmac8。

第一層 3x3 Conv 不符合上述 contract，繼續走 B0 scalar fallback。如此只替最常見
且收益最大的 1x1 shape 建立 fast path。

## 1. B1 優化：量化數學轉換

B0 對一個 output element 計算：

```text
acc = sum(filter[i] * (input[i] + input_offset)) + bias[output_c]
```

展開後：

```text
acc = sum(filter[i] * input[i]) + input_offset * sum(filter[i]) + bias[output_c]
```

對 1x1 Conv 的每個 output channel，先算一次：

```text
adjusted_bias[output_c] = bias[output_c] + input_offset * sum(filter[output_c][input_c])
```

之後每個 output pixel 只需要：

```text
acc = adjusted_bias[output_c] + sum(filter[input_c] * input[input_c])
```

`adjusted_bias` 先在 generate-time 算好存下，避免 runtime 還要多花 instruction 計算。

只接受：

```text
filter_H == 1
filter_W == 1
input_C == filter_input_C      
pad_H == 0
pad_W == 0
```

不符合時直接執行從 B0 保留的 scalar fallback。盤點 Efficient Lite0：
- 第一層 3x3 Conv 走 B0 的方法
- 其餘都是 1x1 Conv，所以能全部走 B1 的 fast path


### Byte-exact 注意事項

數學重排把 `input_offset * filter` 與 bias 移到 raw dot product 前面。只要 INT32 accumulator 沒有 overflow，加法順序改變不會改變結果。

```text
abs(bias)
+ filter_input_C * max(abs(filter)) * max(abs(input + input_offset))
```

- Phase 3-2 實作前需要確認本模型各 1x1 Conv 的上述 bound 不超過 INT32
- Phase 3-3 實作前應確定模型的資料不會 overflow。因為 Xqmac8 contract 明確規定 INT32 overflow 使用 modulo-2^32 wraparound。


## 2. B1 優化：Input Stationary

B1 的 1x1 conv 一次計算四個 output channels，但每一次 MAC 仍然是一般
RV64IMC scalar instruction：

```text
output_c + 0 -> accumulator 0
output_c + 1 -> accumulator 1
output_c + 2 -> accumulator 2
output_c + 3 -> accumulator 3
```

對同一個 input channel：

```text
讀取一次 input
讀取四個 output channels 的 weights
更新四個互相獨立的 accumulators
```

> 目前 EfficientNet-Lite0 的 32 個 1x1 Conv，其 output channels 都能被 4 整除，所以完整模型不會遇到 tail channel 問題。

## 3. B1 instruction-count 試算

目前模型的實際資料為：

```text
1x1 Conv operators：       32
1x1 Conv MACs：             338,537,472
1x1 Conv output elements：   4,029,760
B0 全模型 instructions： 4,449,648,842
```


```text
B0：
    1 lb input
    1 lb filter
    1 addi filter pointer
    1 addw input, input_offset
    1 mulw
    1 addi input pointer
    1 addw accumulator
    1 branch
    ------------------------
    8 instructions / MAC

B1 adjust bias：
    1 lb input
    1 lb filter
    1 addi filter pointer
    1 mulw
    1 addi input pointer
    1 addw accumulator
    1 branch
    ------------------------
    7 instructions / MAC

B1 adjust bias + tile 4：
    1 input load
    4 weight loads
    4 multiplies
    4 accumulator adds
    1 input-pointer update
    4 weight-pointer updates
    1 loop branch
    ------------------------
    19 instructions / 4 MACs
    = 4.75 instructions/MAC
```

| 方案 | 1x1 inner-loop cost | 預估全模型 instructions | 相對 B0 |
| --- | ---: | ---: | ---: |
| B0 | 8/MAC | 4.450B | - |
| B1 adjust bias | 7/MAC | 4.111B | 約 -7.6% |
| B1 adjust bias + tile 4 | 4.75～5/MAC | 3.349～3.434B | 約 -22.8～-24.7% |

> 預期透過專用 1×1 path 還能額外移除的通用 Conv2D overhead。
> 例如額外移除 filter loops、額外的 loop branches、padding boundary check 、group 計算等，還有移除 dilation 計算改成 walking pointer。



## 4. X1：以 Xqmac8 取代八次 scalar MAC

Xqmac8 contract：

```text
xqmac8 rd, rs1, rs2

rd = INT32(rd)
     + INT8(rs1[ 7: 0]) * INT8(rs2[ 7: 0])
     + ...
     + INT8(rs1[63:56]) * INT8(rs2[63:56])
```

- `rs1` 包含八個 raw INT8 inputs。
- `rs2` 包含八個 INT8 weights。
- 舊 `rd` 是 accumulator，新 `rd` 是結果。
- lane 0 位於最低 8 bits。
- overflow 使用 modulo-2^32，最後 sign-extend 到 RV64。
- encoding 使用 `CUSTOM_0 = 0x0b`、`funct3 = 0`、`funct7 = 0`。


### Input-channel group 與 tail

X1 將 input-channel loop 拆成：

```text
完整的 8-channel groups  -> Xqmac8
最後剩下的 0～7 channels -> scalar MAC
```

例如 `filter_input_C = 11`：

```text
channels 0..7  -> 一次 Xqmac8
channels 8..10 -> 三次 scalar MAC
```

> 不過所有 1×1 Conv 的 input channel 都可以被 8 整除，不會遇到不足 8 個 channel 的尾端。因此在 X1 **不實作處理 tail**

### Packed load 與 alignment


Phase 3-3 仍應明確保證 constant weight arrays 至少 8-byte aligned，而不是依賴
linker 剛好給出的位址。若這項保證需要修改共用 generator，新增區塊要清楚標記為
X1 alignment requirement，而且不得改變 B0/B1 的 tensor bytes。

X1 load helper 分成兩種情況：

```text
8-byte aligned pointer   -> 一次 64-bit load
unaligned test pointer   -> 安全的 byte packing，或改走 scalar
```

實際 EfficientNet fast path 必須從 objdump 確認產生 `ld`。不能只因為 Spike 可以
執行 misaligned load，就假設未來 CV-Wally 也能接受。


## 5. X1 instruction-count 試算
### B1 實作前的預測

X1 保留 B1 的四個 output accumulators。

若不做 output tile，每個 8-channel group 約需要：

```text
1 packed input load
1 packed filter load
1 Xqmac8
2 pointer updates
1 loop branch
--------------------
6 instructions / 8 MACs
= 0.75 instruction/MAC
```

使用 tile 4 後，每 32 個 MACs 約為：

```text
1 packed input load
4 packed filter loads
4 Xqmac8
5 pointer updates
1 loop branch
--------------------
15 instructions / 32 MACs
= 0.47 instruction/MAC
```

| 方案 | 1x1 inner-loop cost | 預估全模型 instructions | 相對 B0 |
| --- | ---: | ---: | ---: |
| X1、沒有 output tile | 約 0.75/MAC | 約 1.995B | 約 -55.2% |
| X1、output tile 4 | 約 0.47～0.50/MAC | 約 1.900～1.911B | 約 -57.1～-57.3% |

### B1 已實作後的修正預測

B1 實作前的預測是從 B0 全模型扣除簡化後的 inner-loop cost，沒有完整計入
專用 1x1 path 一併移除的通用 Conv2D overhead。
B1 已經實作後，可以直接使用 B1 當基準。

Phase 3-2 實測 B1 全模型為：

```text
B1 全模型 instructions： 3,096,086,990
1x1 Conv MACs：             338,537,472
B1 tile 4 inner-loop cost：       4.75 instructions/MAC
```


若 X1 保留 output tile 4：

```text
3,096,086,990 - (4.75 - 0.50) * 338,537,472
= 1,657,302,734 instructions

3,096,086,990 - (4.75 - 15 / 32) * 338,537,472
= 1,646,723,438 instructions
```

| 方案 | 1x1 inner-loop cost | 修正後預估 instructions | 相對 B0 | 相對 B1 |
| --- | ---: | ---: | ---: | ---: |
| X1、沒有 output tile | 0.75/MAC | 約 1.742B | 約 -60.9% | 約 -43.7% |
| X1、output tile 4 | 0.47～0.50/MAC | 約 1.647～1.657B | 約 -62.8～-63.0% | 約 -46.5～-46.8% |

> 結果高度符合預測。


## 6. Correctness gates

### Phase 3-2：B1

1. B0 原始檔不變。所有同檔案內的 B1 更新需要明確標示。
2. Generator 以 INT64 計算 adjusted bias，超出 INT32 時停止。
3. 1x1 targeted test 從非零原始 input offset 產生 adjusted bias，驗證 output tile 4
   與 output-channel tail。
4. 3x3/padding testcase 證明原始 bias/input offset 的 scalar fallback 仍 byte-exact。
5. Host 完整 C model 與 TFLite fixed panda intermediate outputs byte-exact。
6. B1 Spike ELF 的固定 panda uint8[1000] output byte-exact。
7. objdump 確認 hot loop 沒有 accumulator stack spill。
8. 通過 correctness 後才量測 B1 retired instructions。

### Phase 3-3：X1

1. Tutorial 與正式 Xqmac8 implementation 相互獨立。
2. Targeted test 使用由非零原始 input offset 產生的 adjusted bias。
3. 主要 aligned path 的 objdump 確認使用 64-bit load。
4. objdump 確認四個 accumulators 沒有 stack spill。
5. 載入正式 Spike extension 後，固定 panda uint8[1000] output byte-exact。
6. 通過 correctness 後才量測 Xqmac8 count 與完整 instruction profile。


## 7. 目前不做的項目

- 不優化 DepthwiseConv2D。
- 不優化 FullyConnected。
- 不修改 quantization rounding。
- 不修改 TFLite model tensor layout。
- 不為 B1/X1 複製 executor 或 testbench。
- B1/X1 只維護一份 generate-time adjusted model data。
- 不用 Phase 3-1 的估計取代 Phase 3-2、3-3 的實際 Spike 量測。
