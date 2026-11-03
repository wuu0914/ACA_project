# B2 / X2 DepthwiseConv2D design considerations


- X1：Phase 3 已量測的 RV64IMC + Xqmac8 baseline。
- B2-1：使用 walking pointer，並在 channel loop 外決定有效 filter 範圍。
- B2-2：延續 B2-1，一次計算四個 output channels。
- B2-3：延續 B2-2，討論 adjusted bias 與實體 padding；目前只做設計評估。
- X2：延續 B2-2 的有效-tap fast path，再加入兩個獨立 lane 的 Xdwmac2。


## 0. 觀察

Phase 2 的 B0 實測結果為：

```text
B0 全模型 instructions：             4,449,648,842
conv2d_s8 instructions：              3,242,185,190  (72.86%)
depthwise_conv2d_s8 instructions：      921,897,095  (20.72%)
```

Phase 3 以 B1/X1 優化 Conv2D，沒有修改 DepthwiseConv2D。完成 X1 後：

```text
X1 全模型 instructions：             1,647,729,506
conv2d_s8 instructions：                440,265,854  (26.72%)
depthwise_conv2d_s8 instructions：      921,897,095  (55.95%)
Depthwise estimated MACs：               34,532,064  (8.96%)
有效 taps MACs：                         30,648,416
```

X1 將 Conv2D 從 `3,242,185,190` 降至 `440,265,854` instructions，DepthwiseConv2D 從 B0 的第二大成本變成 X1 全模型最大的瓶頸。 Depthwise 的 estimated MACs 為 `34,532,064`；扣除 SAME padding 外不會執行的 taps 後，實際只有 `30,648,416` 次有效 MAC。但在 B0 卻占用 20.72% 的 retired instructions。

因此 B2 應優先消除重複的控制與位址工作，而不是先新增更多乘法硬體：padding
有效範圍只和 output position 有關，可以移到 channel loop 外判斷；可以用 walking pointers；相鄰 channels 可以共用同位置的 index，能用多個 accumulators 分攤控制成本。

目前 16 個 DepthwiseConv2D 都符合：

```text
depth_multiplier == 1
dilation_H == 1
dilation_W == 1
output_C == input_C
input_C % 8 == 0
input_offset == 128
```


### 試算方式

目前 `depthwise_conv2d_s8` 的 Spike histogram 與 objdump 可分成：

```text
保留的 output、requantize 呼叫準備與外層工作：    94,705,543
B2 filter-range、row 與 tile setup 預留：        10,000,000
```

`output_scaling_round2` 的 function body 在 profile 中是另一個 function，不包含在
上述 `94,705,543`。三個 B2 版本都不修改 requantization，因此它會原封不動留在
全模型 instruction count 中。

後續預測使用：

```text
預測 B2 Depthwise body
    = 94,705,543
    + 10,000,000
    + 新 inner-loop instructions
    + 額外資料準備 instructions

預測全模型
    = 1,647,729,506
    - 921,897,095
    + 預測 B2 Depthwise body
```

`10,000,000` 是尚未實作前保守保留的 fast-path setup，不是 Spike 實測值。
最終仍要以 objdump 與 Spike profile 修正預測。




## 1. B2-1：移動判定位置與 pointer walking - 穩賺不賠

目前 kernel 在每一個 channel、每一個 filter tap 內計算 input index，並判斷該
位置是否落在圖片內：

```text
output pixel
  channel
    filter_y
      計算 input_y
      判斷 y boundary
      filter_x
        計算 input_x
        判斷 x boundary
        計算 input/filter index
        MAC
```

B2-1 可在 channel loop 外算：

```text
filter_y_start = max(0, -input_y_origin)
filter_y_end   = min(filter_H, input_H - input_y_origin)

filter_x_start = max(0, -input_x_origin)
filter_x_end   = min(filter_W, input_W - input_x_origin)
```

這組公式以目前 `dilation == 1` 為前提。之後所有 channels 共用相同範圍：

```text
output pixel
  決定有效 filter_y/filter_x 範圍
  channel
    只走有效 filter taps
      MAC
```

因此 inner loop 不再執行 padding boundary check。


### Pointer walking

每一列先算 input 與 filter 的起點，之後不再為每個 tap 重算完整 index：

```c
input_pointer  += input_C;
filter_pointer += output_C;
```

NHWC input 與 `1HWO` filter 中，相鄰 spatial taps 的同一個 channel 相隔一整組
channels，因此兩個 pointer 都依 channel stride 前進。

B2-1 每個有效 MAC 的 inner-loop 預測為：

```text
1 input load
1 filter load
1 input-offset add
1 multiply
1 accumulator add
2 pointer updates
1 loop branch
-----------------
8 instructions / MAC
```

Row 起點與有效範圍準備已保守放入前面的 `10,000,000` setup budget。


### B2-1 instruction-count 試算

```text
inner loop
    = 30,648,416 * 8
    = 245,187,328

Depthwise body
    = 94,705,543 + 10,000,000 + 245,187,328
    = 349,892,871

全模型
    = 1,647,729,506 - 921,897,095 + 349,892,871
    = 1,075,725,282
```

| 比較 | 預估改善 |
| --- | ---: |
| B2-1 Depthwise body 相對目前 X1 Depthwise | 約 -62.05% |
| B2-1 全模型相對目前 X1 | 約 -34.71% |



## 3. B2-2：四個 output channels - 提高 Register Utilization

DepthwiseConv2D 的相鄰 channels 對應四個不同 outputs：

```text
acc0 += input[channel + 0] * filter[channel + 0]
acc1 += input[channel + 1] * filter[channel + 1]
acc2 += input[channel + 2] * filter[channel + 2]
acc3 += input[channel + 3] * filter[channel + 3]
```

四個 channels 不能共用同一個 input value，Weight 跟 Input 需要 load 同一個地址 4 次，但能共用：

```text
有效 filter 範圍
filter_y/filter_x loops
input/filter pointer updates
inner-loop branch
```

所以 B2-2 保留四條獨立 accumulator，一次處理四個相鄰 channels。

每個有效 spatial tap、四個 channels 的預測為：

```text
4 input loads
4 filter loads
4 input-offset adds
4 multiplies
4 accumulator adds
2 pointer updates
1 loop branch
--------------------
23 instructions / 4 MACs
= 5.75 instructions / MAC
```

相較 B2-1 的四個 MACs：

```text
B2-1：4 * 8 = 32 instructions
B2-2：        23 instructions
```

省下的是三組 input/filter pointer updates 與三次 inner-loop branches；四個
channels 所需的 input、filter、multiply 與 accumulator update 仍然存在。


### B2-2 instruction-count 試算

```text
inner loop
    = 30,648,416 * 5.75
    = 176,228,392

Depthwise body
    = 94,705,543 + 10,000,000 + 176,228,392
    = 280,933,935

全模型
    = 1,647,729,506 - 921,897,095 + 280,933,935
    = 1,006,766,346
```

| 比較 | 預估改善 |
| --- | ---: |
| B2-2 全模型相對 B2-1 | 約 -6.41% |
| B2-2 全模型相對目前 X1 | 約 -38.90% |

B2-2 相較 B2-1 再省：

```text
1,075,725,282 - 1,006,766,346
= 68,958,936 instructions
```

四個 accumulators 必須從 objdump 確認都留在 registers。若 GCC 發生 stack spill，
這份試算便不成立。




## 4. B2-3：adjusted bias 與實體 padding - 最好解法中，省下的指令數要打折且需要額外記憶體空間 (未做)

原始 DepthwiseConv2D 對有效 taps 計算：

```text
acc[c]
  = bias[c]
  + sum_valid(filter[tap][c] * (input[tap][c] + input_offset))
```

展開後：

```text
acc[c]
  = bias[c]
  + sum_valid(filter[tap][c] * input[tap][c])
  + input_offset * sum_valid(filter[tap][c])
```

與 1x1 Conv 不同，Depthwise SAME padding 的有效 filter taps 會隨 output position
改變。若直接使用完整 filter sum：

```text
adjusted_bias[c]
  = bias[c] + input_offset * sum_all(filter[tap][c])
```

卻仍略過 padding taps，邊界 output 就會多出：

```text
input_offset * sum_padding(filter[tap][c])
```

因此這個作法不會 byte-exact。


### 可行方法比較

第一種方法是為每個 output position 與 channel 準備不同 adjusted bias：

```text
adjusted_bias[y][x][c]
  = bias[c] + input_offset * sum_valid(y, x, filter[tap][c])
```

全模型共有 `2,306,528` 個 Depthwise output elements。直接保存 INT32 correction
需要：

```text
2,306,528 * 4 = 9,226,112 bytes
```

雖然可以再依 interior、edge、corner 分類壓縮，但 runtime 必須選擇 border class，
又會增加 index 與 branch。因此這不是目前最簡單的方法。

最可行的方法是建立 padded input buffer，padding byte 填 input zero point：

```text
raw padding = -input_offset = -128
```

再固定執行完整 3x3 或 5x5 filter。此時 padding tap 的貢獻為：

```text
filter * (-input_offset) + input_offset * filter = 0
```

所以 generate-time 可以安全計算：

```text
adjusted_bias[c]
  = bias[c] + input_offset * sum_all(filter[tap][c])

acc[c]
  = adjusted_bias[c] + sum_all(filter[tap][c] * raw_input[tap][c])
```

Generator 必須使用 INT64 計算 adjusted bias，並確認所有可能 accumulator 不會
超出 INT32；否則改變加法順序可能改變 TFLite 結果。


### B2-3 inner-loop 試算

移除四次 input-offset add 後：

```text
4 input loads
4 filter loads
4 multiplies
4 accumulator adds
2 pointer updates
1 loop branch
--------------------
19 instructions / 4 MACs
= 4.75 instructions / MAC
```

但是 padded input 會讓 kernel 執行全部名義 MACs：

```text
B2-2：30,648,416 有效 MACs
B2-3：34,532,064 完整 MACs
增加： 3,883,648 MACs
```

因此 inner loop 只從：

```text
B2-2：30,648,416 * 5.75 = 176,228,392
B2-3：34,532,064 * 4.75 = 164,027,304

差異：12,201,088 instructions
```


### Padded buffer 成本

16 層 DepthwiseConv2D 一次 inference 的資料量為：

```text
原始 input interiors：3,788,288 bytes
padded buffers：       4,694,016 bytes
可重複使用的最大 scratch buffer：約 1.25 MiB
```

若使用對齊的 64-bit loop，先估：

```text
填 padded buffer：
    4,694,016 / 8 * 3
    = 1,760,256 instructions

複製 input interior：
    3,788,288 / 8 * 5
    = 2,367,680 instructions

合計：4,127,936 instructions
```

填入 loop 的三條指令是 `sd + pointer update + branch`；複製 loop 的五條指令是
`ld + sd + 2 pointer updates + branch`。這仍是對 alignment 與 GCC codegen 有利的
估計，不是實測值。


### B2-3 instruction-count 試算

```text
Depthwise body
    = 94,705,543
    + 10,000,000
    + 164,027,304
    + 4,127,936
    = 272,860,783

全模型
    = 1,647,729,506 - 921,897,095 + 272,860,783
    = 998,693,194
```

B2-3 相較 B2-2 只省：

```text
1,006,766,346 - 998,693,194
= 8,073,152 instructions
```

| 比較 | 預估改善 |
| --- | ---: |
| B2-3 全模型相對 B2-2 | 約 -0.80% |
| B2-3 全模型相對目前 X1 | 約 -39.39% |


## 5. 三個 B2 版本比較

| 版本 | Depthwise inner-loop | 預估 Depthwise body | 預估全模型 | 相對 X1 |
| --- | ---: | ---: | ---: | ---: |
| X1 現況 | 約 30.08 instructions/有效 MAC | 921,897,095 | 1,647,729,506 | - |
| B2-1 range + pointer walking | 8/有效 MAC | 349,892,871 | 1,075,725,282 | -34.71% |
| B2-2 + channel tile 4 | 5.75/有效 MAC | 280,933,935 | 1,006,766,346 | -38.90% |
| B2-3 + adjusted bias/padding | 4.75/名義 MAC | 272,860,783 | 998,693,194 | -39.39% |
| X2 = B2-2 + Xdwmac2 | 2.25/有效 MAC | 173,664,479 | 899,496,890 | -45.41% |

模型的 estimated MACs 維持 `385,187,552`，因此預估：

| 版本 | Estimated MACs / retired instruction |
| --- | ---: |
| X1 實測 | 0.233769 |
| B2-1 預測 | 0.358072 |
| B2-2 預測 | 0.382599 |
| B2-3 預測 | 0.385692 |
| X2 預測 | 0.428226 |

B2-1 與 B2-2 的改善值得實作；B2-3 的軟體單獨收益不足以合理化新增的資料準備
與記憶體成本。




## 6. X2：Xdwmac2 instruction contract (未做)

Xqmac8 將八個 lane products 水平加總到一個 accumulator，不能直接用於
DepthwiseConv2D。四個相鄰 Depthwise channels 必須保留四個獨立 accumulators。

RV64 register 可以裝兩個 INT32 accumulators，因此 X2 定義：

```text
xdwmac2 rd, rs1, rs2

old_acc0 = UINT32(rd[31:0])
old_acc1 = UINT32(rd[63:32])

input0 = INT32(INT8(rs1[7:0]))  + 128
input1 = INT32(INT8(rs1[15:8])) + 128

product0 = input0 * INT8(rs2[7:0])
product1 = input1 * INT8(rs2[15:8])

rd[31:0]  = old_acc0 + UINT32(product0)
rd[63:32] = old_acc1 + UINT32(product1)
```

- 低 32 bits 與高 32 bits 是兩個互相獨立的 accumulators。
- 每個 lane 各自使用 modulo-2^32 overflow，不允許 carry 跨越兩個 lanes。
- lane 0 位於最低 byte。
- 與目前 Xqmac8 一樣，舊 `rd` 是 implicit accumulator source，新 `rd` 是結果。
- 兩個 input lanes 都在 instruction 內加上固定 `input_offset = 128`。
- 暫定使用 `CUSTOM_0 = 0x0b`、`funct3 = 1`、`funct7 = 0`，與 Xqmac8 的
  `funct3 = 0` 分開。

此 contract 刻意針對目前 EfficientNet-Lite0 的 16 個 DepthwiseConv2D；它們的
`input_offset` 全部都是 128。X2 因此可以直接沿用 B2-2，只走有效 taps，padding
仍然略過，也不需要建立 padded buffer。

如果其他模型的 `input_offset != 128`，必須走 scalar fallback。要把 offset 做成
任意值，需要額外 operand、設定 CSR，或改用 B2-3 adjusted bias；這些都不是目前
X2 的最小設計。


### 為什麼一次只做兩個 channels？

兩個 INT32 accumulators 正好填滿一個 RV64 register：

```text
64 bits = 2 * 32-bit accumulator
```

若一條指令同時更新四個 channels，需要同時寫入兩個 destination registers，或
新增隱藏 accumulator state。這會增加 register dependency、precise exception 與
compiler register allocation 的複雜度。Xdwmac2 保留單一 `rd`，較適合實作與驗證。




## 7. X2 instruction-count 試算

B2-2 已經保留四個 channel tile、在 channel loop 外決定有效範圍，並略過所有
padding taps。X2 每次先用兩次 32-bit load 取得四個相鄰 inputs 與 filters，再以
兩次 Xdwmac2 完成 `filter * (input + 128)`，更新 `acc01` 與 `acc23`：

```text
1 lw packed input
1 lw packed filter
1 Xdwmac2 for channel 0/1
1 shift packed input by 16
1 shift packed filter by 16
1 Xdwmac2 for channel 2/3
2 pointer updates
1 loop branch
----------------------------
9 instructions / 4 MACs
= 2.25 instructions / MAC
```

這裡假設：

- channel tile 起點與 tensor/filter base 至少 4-byte aligned；
- input_C 與 output_C 都能被 4 整除；
- objdump 確認 packed access 各自是單一 `lw`；
- `acc01` 與 `acc23` 留在 registers，沒有 stack spill。

因為 X2 和 B2-2 一樣只走有效 taps，Xdwmac2 動態執行次數為：

```text
30,648,416 MACs / 2 MACs per Xdwmac2
= 15,324,208 Xdwmac2 instructions
```

Inner loop：

```text
B2-2 scalar：30,648,416 * 5.75 = 176,228,392
X2 Xdwmac2： 30,648,416 * 2.25 =  68,958,936

額外節省：107,269,456 instructions
```

保守保留 B2-2 的其他成本；X2 沒有 padded-buffer preparation：

```text
X2 Depthwise body
    = 94,705,543
    + 10,000,000
    + 68,958,936
    = 173,664,479

X2 全模型
    = 1,647,729,506 - 921,897,095 + 173,664,479
    = 899,496,890
```

| 比較 | 預估改善 |
| --- | ---: |
| X2 Depthwise body 相對目前 X1 Depthwise | 約 -81.16% |
| X2 全模型相對 B2-2 | 約 -10.65% |
| X2 全模型相對目前 X1 | 約 -45.41% |

X2 預估的 estimated MACs / retired instruction 為：

```text
385,187,552 / 899,496,890 = 0.428226
```

Xdwmac2 對 Depthwise kernel 本身的改善很大，但全模型還保留 Conv2D、
requantization 與其他 operators，因此新增 instruction 相對軟體 B2 的 end-to-end
收益約為 10.65%，不是另一個 Xqmac8 等級的改善。




## 8. X2 需要增加的硬體

### 與 Xqmac8 datapath 共用

X1 的 Xqmac8 已經需要：

```text
8 個 signed INT8 * INT8 multipliers
product sign extension
horizontal adder tree
32-bit accumulator add
CUSTOM_0 decode
rd implicit read 與 integer register writeback
```

Xdwmac2 可以重用 Xqmac8 的前兩個 signed `INT8 * INT8` multipliers，但 B2-2 沒有
adjusted bias，所以硬體還要實現：

```text
weight * (input + 128)
    = weight * input + (weight << 7)
```

這樣不必新增 9-bit activation multiplier，但必須為每個 lane 加上 offset
correction。X2 相對 X1 需要新增或修改：

1. 一組 `funct3 = 1` decoder/control。
2. 將 product 0 與 product 1 繞過 horizontal reduction tree 的 bypass routing。
3. 兩組 weight sign-extension 與 `weight << 7` wiring。
4. 兩個 32-bit correction adders，分別計算
   `weight * input + (weight << 7)`。
5. 將舊 `rd` 拆成兩個 UINT32 accumulators。
6. 第二個獨立的 32-bit accumulator adder；第一個可以重用 Xqmac8 最後的
   accumulator adder。
7. 防止 low-lane carry 進入 high lane 的兩組 modulo-2^32 carry handling。
8. 將兩個 32-bit 結果重新 pack 成一個 64-bit writeback value。
9. Xqmac8/Xdwmac2/result datapath 的 writeback mux。

實際 RTL 也可以把 correction 與 accumulator add 合成三輸入 adder/compressor，
但不能在 instruction-count 試算中假設它沒有面積或 timing 成本。

若不共用 Xqmac8 datapath，另一種作法是新增兩個 unsigned-9-bit activation 乘
signed-8-bit weight 的 multipliers，再加兩個 32-bit accumulator adders；這不是
建議方案。


### Register file 與 critical path

Xdwmac2 與目前 Xqmac8 一樣，instruction encoding 只明寫 `rs1`、`rs2` 與 `rd`，
但執行時也要讀舊 `rd`。因此真正的 CV-Wally implementation 必須處理第三個
source dependency 或改用明確帶 accumulator source 的 encoding。

這不是 X2 新增加的問題；X1 的 Xqmac8 已經有同樣需求。若 X1 已提供 implicit
`rd` read，X2 不需要額外 register-file read/write port，也不增加 architectural
state。

使用 correction-add 方法時，Xdwmac2 的主要 combinational path 為：

```text
INT8 multiply -> offset-correction add -> accumulator add -> result mux
```

它可能仍比 Xqmac8 的八 lane horizontal reduction tree 短，但不能在 RTL 前直接
保證不影響 critical path。Area、Fmax 與 cycle latency 必須由 CV-Wally synthesis
決定，不能由 Spike instruction count 代替。這也是 X2 堅持基於 B2-2、不要做
adjusted bias 時所付出的硬體成本。




## 9. Correctness gates

### B2-1 / B2-2

1. 3x3、5x5、stride 1、stride 2 的 interior、edge、corner targeted tests 都與
   TFLite byte-exact。
2. Fast path 只接受 `depth_multiplier == 1`、`dilation == 1` 與相容 channel count；
   其餘形狀安全走原始 scalar fallback。
3. B2-2 targeted test 同時驗證四個獨立 accumulators，不把 channels 水平加總。
4. 固定 panda 的 64 個 intermediate outputs 與 TFLite byte-exact。
5. Spike 固定 panda 的最終 1000-byte output byte-exact。
6. objdump 確認 inner loop 無 padding branch、使用 walking pointers、四個
   accumulators 無 stack spill。
7. 通過 correctness 後才量測 retired instructions，修正本文件預測。


### X2

1. Fast path 只接受 `input_offset == 128`；其他 input zero point 安全走 scalar
   fallback。
2. Xdwmac2 directed tests 覆蓋 raw input `-128`、`127`，驗證 instruction 內得到
   centered input `0`、`255`。
3. Directed tests 覆蓋正負 weights、兩 lanes 不同符號、low-lane carry 與每 lane
   modulo-2^32 overflow。
4. Randomized test 將 Xdwmac2 與兩組
   `weight * (input + 128) + accumulator` 軟體 reference byte-compare。
5. X2 沿用 B2-2 的有效-tap範圍；edge/corner 不得執行 padding taps。
6. Input channel tile 與 packed loads 的 4-byte alignment 明確保證。
7. objdump 確認每四個 channels 使用兩個 `lw`、兩個 Xdwmac2，且 accumulator
   registers 無 spill。
8. Spike profile 應量到約 `15,324,208` 個 Xdwmac2；若不同，先解釋有效 taps、
   fallback 或 compiler codegen，再引用 instruction-count 結果。
9. 固定 panda 的 intermediate 與 final outputs 都與 TFLite byte-exact。
10. ISA byte-exact 後，再用 CV-Wally 評估 cycles、CPI、area、Fmax 與 critical
    path。
