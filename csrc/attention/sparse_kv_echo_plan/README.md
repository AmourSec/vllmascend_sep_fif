# ECHO 稀疏 KV 缓存算子：echo_perf 分支中文代码说明

本文介绍 `echo_perf` 分支怎样在 Ascend A5/950 上管理稀疏 KV 驻留缓存，
包括调用链、数据结构、Plan 的执行步骤、MTP 处理，以及这一版性能优化的依据。
代码基准为 `6530f3e9389b016444bc9a8619e293056f9ac84b`。

当前完成了本地算法对照与内存边界测试，尚未完成这一版的 CANN 编译、A5 实机性能测试、
图模式 replay 和 PD/MTP 端到端验证。文中的操作次数都是源码分析或 CPU 模型结果，不是 NPU 耗时。

## 1. 先理解这个分支要解决什么

模型的历史 KV 可以很长，NPU 上的稀疏 attention 驻留区只保留其中一部分。
每次 Indexer 给出 TopK 后，需要回答三个问题：

1. 哪些 token 的 KV 已经在当前 NPU 驻留区，可以直接使用？
2. 哪些 token 的 KV 不在，需要从 Host 池搬进来？
3. 新 KV 应该放在哪个槽位；空间不够时，哪些旧 KV 可以被替换？

本分支用 NPU 上的 ECHO 风格 Plan 处理这些问题，再由 Transfer 执行搬运。
CPU 不需要逐 token 生成本轮的缓存淘汰结果和搬运清单。

几个版本的关系如下：

| 版本 | Plan 的主要实现 | 本文中的作用 |
| --- | --- | --- |
| `clean_simt`，`cd84b83c5` | 哈希表、前缀和及驻留状态管理 | 原 SIMT 性能基线 |
| `echo`，`0603ced0c` | 直接索引表、优先级、二分寻找淘汰阈值 | ECHO 第一版 |
| `echo_perf`，`6530f3e93` | 直接索引表、radix 分桶选择、UB 缓存、减少原子操作 | 本文讲解的优化版 |

`echo_perf` 延续第一版 ECHO 的缓存规则，优化其实现成本。
它与旧哈希 Plan 的淘汰策略并非完全相同，因此两者在连续多轮运行后的 miss 数也可能不同。
这是一份结合现有框架的 ECHO 思路实现，尚未实现原 ECHO 系统中的 Indexer 融合预取等全部功能。

## 2. 它在整条推理链路中的位置

下面是逻辑数据依赖，实际调度还受到框架流和事件依赖约束：

```text
P 端计算历史 KV ── PD / MemFabric 传输 ──→ D 端 Host KV 池
                                                  ↑
D 端当前层计算新 KV ── 既有 offload 写回路径 ──────────┘

Indexer 根据自己的索引 K 等数据计算 TopK
                    ↓
ECHO Plan：检查命中、安排槽位、生成 miss 清单
                    ↓
Transfer：把 miss 对应的 KV 从 Host 池搬到 NPU resident buffer
                    ↓
Sparse Flash Attention：根据槽位编号读取 resident KV
```

这里要分清各部分的职责：

| 部分 | 负责什么 |
| --- | --- |
| Indexer | 选择当前 query 需要关注的历史 token 位置 |
| ECHO Plan | 管理哪些 KV 已驻留、哪些需要搬运，以及目的槽位 |
| Transfer | 按 Plan 的结果实际搬运 K/V 字节 |
| SFA | 使用准备好的稀疏 KV 计算 attention |
| MTP/投机调度 | 生成和验证草稿，决定接受哪些 token，以及下一轮从哪里继续 |
| MemFabric | 在现有框架中提供 Host 池、地址映射和 PD 数据传输能力 |

Plan 不计算 KV，也不决定草稿接受或拒绝。Transfer 负责 Host→resident 的加载；
当前层新 KV 写回 Host 的工作仍由已有 offload 路径负责。
读取源数据之前，框架必须保证 PD 传输或当前层写回已经满足依赖，Plan 本身不负责等待远端网络传输。

## 3. 从 Python 到两个设备 kernel

主要调用链是：

```text
sfa_kv_offload.py
  ├─ 计算逐 token 可见长度和稳定前缀
  └─ manager.onload_topk_kv(...)
       └─ _onload_topk_kv_simt(...)
            └─ SparseKVEcho.plan_and_transfer(...)
                 └─ torch.ops._C_ascend.npu_sparse_kv_echo_plan_transfer(...)
                      ├─ EXEC_NPU_CMD(aclnnSparseKvEchoPlan, ...)
                      └─ EXEC_NPU_CMD(aclnnSparseKvTransfer, ...)
```

Python 调用一次组合自定义算子，C++ 适配层依次提交两个 ACLNN 算子。
这是两个设备 kernel 的顺序执行，不是把 Plan 与 Transfer 融合成一个设备 kernel。

具体源码入口：

| 文件 | 建议关注的内容 |
| --- | --- |
| [sparse_kv_echo.py](../../../vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_echo.py) | `SparseKVEcho`、持久 tensor 初始化、组合调用与复用调用 |
| [sparse_kv_offload_manager.py](../../../vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py) | `_initialize_echo_backend`、`_onload_topk_kv_simt` |
| [sfa_kv_offload.py](../../../vllm_ascend/attention/sfa_kv_offload.py) | `_compute_decode_token_lengths`、稳定前缀、resident views |
| [sparse_kv_echo_torch_adpt.h](sparse_kv_echo_torch_adpt.h) | `npu_sparse_kv_echo_plan_transfer` 的两次 ACLNN 提交 |
| [torch_binding.cpp](../../torch_binding.cpp) | 算子 schema、可修改 tensor 的 alias 声明 |
| [torch_binding_meta.cpp](../../torch_binding_meta.cpp) | Meta 实现注册 |

`uses_simt_lru`、`_onload_topk_kv_simt` 是框架沿用的字段与函数名称。
在这个分支的 A5 后端，它们连接的是 `SparseKVEcho`，并不表示仍在调用旧哈希 Plan。

## 4. 读代码前先认识这些数组

用 `R` 表示最大 query 行数，`C` 表示每行容量 `capacity`，`T` 表示 `topk`，
`M` 表示 `max_token`。token 在本文中通常指请求中的逻辑位置，不是词表 token ID。

### 4.1 为什么是逐行，而不是简单地逐 request

普通 decode 时，一个 request 通常对应一行 query。
MTP 验证或多 token decode 时，同一 request 可以对应多行 query，每行的可见范围和 TopK 可能不同。

当前实现按 query 行保存驻留状态，即每行有自己的 `C` 个槽位。
同一请求的不同 MTP 行不会自动共享同一份 resident 槽位。
行与请求的联系由 `token_to_req` 和 `req_ids` 确定；行复用给其他请求时，需要重置旧状态。

### 4.2 输入、持久状态与输出

| 名称 | 主要形状/类型 | 含义 |
| --- | --- | --- |
| `topk_indices` | `[R,T]`，int32 | Indexer 选择的逻辑 token 位置，允许包含无效项 |
| `token_to_req` | `[R]`，int32 | 每行属于哪个请求索引 |
| `req_ids` | `[maxRequests]`，int64 | 请求身份，用于区分行上是否换了请求 |
| `visible_seq_lens` | `[R]`，int32 | 每行允许访问的 token 位置上界，不包含该上界 |
| `stable_prefix_lens` | `[R]`，int32 | 该位置之前的 KV 可视为稳定历史，后缀可能被重写 |
| `block_table` | `[maxRequests,maxNumBlocks]`，int32 | 请求逻辑 block 到 Host 池物理 block 的映射 |
| `active_rows` | 单元素 int32 tensor | 本轮实际处理多少行，kernel 在设备侧读取 |
| `last_req_ids` | `[R]`，int64 | 当前每行缓存记录属于哪个请求 |
| `slot_to_token` | `[R,C]`，int32 | 每个 resident 槽位当前存的是哪个逻辑 token；`-1` 表示空 |
| `slot_priorities` | `[R,C+1]`，int64 | 前 `C` 项是槽位优先级，最后一项是该行的逻辑时钟 |
| `current_slots` | `[R,T]`，int32 | 最终供 SFA 使用的槽位编号；无效项为 `-1` |
| `miss_count` | `[R]`，int32 | 每行需要加载的唯一 token 数 |
| `miss_tokens` / `miss_slots` | 各 `[R,T]`，int32 | 待搬 token 与目的槽位，只有前 `miss_count` 项有效 |

持久状态保存在 NPU tensor 中。`SparseKVEcho.__init__()` 在初始化阶段分配这些 tensor；
反复运行 Plan 时更新内容，保持这些状态缓冲区的地址稳定。

`slot_priorities` 中的 clock 是调用累积的缓存时钟，不是 token 位置，也不是序列长度。
当前 ECHO 规则是：空槽优先级为 `-1`，本轮命中项更新为 `clock+1`，新加载项更新为 `clock+2`，
本轮结束后 clock 增加 2。优先淘汰数值较小的槽位，平局时优先较小的槽位编号。
同一轮多个命中项优先级相同，所以它不是对每次单独访问都精确排序的严格 LRU。

### 4.3 临时 workspace

`compact_workspace` 的一个 AIV 分片在 GM 中预留：

```text
tokenMap[M] | victims[C] | missPositions[T] | 对齐填充
```

- `tokenMap`：直接用逻辑 token 位置作为数组下标，暂存首次 TopK 位置或 resident 槽位。
- `victims`：本轮将被新 KV 占用的槽位。
- `missPositions`：唯一 miss 在原 TopK 数组中的位置。

GM workspace 字节数为：

```text
min(max_rows, 64) × align_up(M + C + T, 32) × 4
```

这里的对齐单位是 int32 元素，因此对应 128 字节对齐的分片长度。
分片按 AIV 分配并在各行、各层的顺序调用间复用；它不是每层一张持久的 `M` 大小映射表。
这一复用依赖现有顺序提交方式，不应让共享同一 workspace 的 Plan 在不同 stream 上无依赖地并发运行。

## 5. 一次 Plan 调用具体做什么

核心文件是 [sparse_kv_echo_plan_apt.cpp](op_kernel/sparse_kv_echo_plan_apt.cpp)。
先看 `ProcessRuntimeRow()`，再沿它调用的函数向下读。

### 5.1 重置请求状态，使可能过期的后缀失效

`ResetResident()` 检查每个槽位：

```cpp
if (reset || token < 0 || token >= stablePrefix || token >= maxToken) {
  row.slotToToken[slot] = INVALID_TOKEN;
  row.priority[slot] = -1;
}
```

`reset` 表示该行的请求身份发生变化，需要全部作废。
`token >= stablePrefix` 表示这个位置可能在投机流程中被重写，不能仅凭“位置编号相同”继续认为命中。

这里先作废的是缓存管理记录，不会立即把 resident buffer 里的整段 KV 字节清零。
以后需要这个位置时，Plan 会产生 miss，由 Transfer 覆盖对应槽位的数据。

### 5.2 验证 TopK，准备直接索引表

`IsSourceTokenValid()` 检查 token 是否满足以下条件：

- 位置非负、小于 `maxToken`、小于该行 `visible_seq_lens`。
- 行对应的请求索引合法。
- 逻辑 block 在 block table 范围内，且物理 block 在 Host 池范围内。

无效项不产生搬运任务，最终 `current_slots` 保留 `-1`。
这一检查约束传入 Plan 的数据，并不意味着 Indexer 在正常情况下必须产生未来位置。

代码暂时把有效 token 位置写进 `current_slots`，直到最后一步才把它改成槽位编号。
只将有效 TopK 触及的 `tokenMap[token]` 原子写成 `INT32_MIN`，不遍历清空整个长度为 `M` 的数组。

### 5.3 选举第一次出现的位置，发布命中

`MarkAndCompactMisses()` 首先执行：

```cpp
asc_atomic_max(row.tokenMap + token, -static_cast<int32_t>(pos) - 1);
```

例如 token 50 同时出现在 TopK 的位置 1 和 2，两个线程分别提交 `-2` 和 `-3`。
atomic max 最终保留 `-2`，代表第一次出现的位置 1。后续只给 token 50 生成一条 miss 记录。

注意不同数组中 `-1` 的含义：`slot_to_token=-1` 表示空槽；
`tokenMap=-1` 在这个中间阶段表示 TopK 的位置 0。映射表的空标记是 `INT32_MIN`。

接着，`PublishResidentHits()` 遍历有效 resident 槽位，通过映射表找到可能对应的 TopK 位置，
并反向确认：

```cpp
if (pos < row.topk && row.tokens[pos] == token) {
  row.tokenMap[token] = static_cast<int32_t>(slot);
  row.priority[slot] = hitPriority;
}
```

反向确认很关键。例如 `tokenMap[70]` 残留了上一行的 `-2`，但当前 TopK 第 1 项是 50，
`50 != 70`，因此不会把这条残留记录当作命中。

从全空状态开始，每轮都只为唯一 miss 分配不同槽位，因此有效 resident token 保持唯一。
不同槽位线程发布命中时写的是不同的映射表项和优先级，可以使用普通写入。
调用方不能传入随意构造、同一 token 占多个有效槽位的损坏状态来替代这一不变量。

### 5.4 用前缀和压紧 miss

命中项的映射已经变成非负槽位编号。
未命中的 token 仍保留首次出现位置的负编码，只有被选中的那次出现会产生 `missing=1`。

例如：

```text
missing 标记：  [0, 1, 0, 1, 0]
exclusive scan：[0, 0, 1, 1, 2]
压紧后的位置： [1, 3]
miss_count：   2
```

`BlockExclusiveScan()` 先做 warp 内扫描，再计算各 warp 的基址。
它得到每个 miss 应写入 `missPositions` 的位置，从而避免多个线程争用一个全局计数器。
当前生产配置是 2048 个线程、每个 warp 32 个线程；warp 数由配置推导，没有在循环中写死 64。

### 5.5 选择将被占用的槽位

`SelectVictims()` 处理普通非空行；请求刚切换或稳定前缀为零时，所有槽位已经清空，
直接取槽位 `0..misses-1`，不再执行淘汰选择。

普通行先由 `GatherPriorityStats()` 统计空闲数、非命中 resident 的最小/最大优先级。
命中槽位已更新为 `clock+1`，不会作为淘汰候选。

选择分为三种情况：

| 条件 | 处理 |
| --- | --- |
| 空闲槽位足够 | 选择所需数量的最低编号空槽 |
| 空闲不足，剩余候选优先级相同 | 优先使用空槽，再按槽位编号处理平局 |
| 候选优先级不同 | 使用 radix 分桶找出第 k 小优先级 |

容量要求 `capacity >= topk`。设唯一命中数为 h，唯一 miss 数为 m，
有 `h+m <= topk <= capacity`，因此至少有 m 个非命中槽位可用，不需要淘汰当前 TopK 的命中项。

### 5.6 radix 分桶怎样替代二分扫描

原 `echo` 对 clock 的数值范围二分。每问一次“优先级不大于 mid 的槽位够不够”，
都要重新扫描整个 capacity 并做一次汇总。

现在 `EvictionThreshold()` 按字节缩小候选范围。
一个字节有 256 种取值，因此使用 256 桶直方图。下面用十六进制展示一个需要选择两个槽位的例子：

```text
候选优先级：0x1201、0x1205、0x12FF、0x1300、0x13FF

高字节分桶：0x12 桶有 3 个，0x13 桶有 2 个
第 2 小的值一定在 0x12 桶中

只对 0x12 桶按低字节细分：0x01、0x05、0xFF
第 2 小的值是 0x05，因此完整阈值是 0x1205
```

代码先跳过最小值与最大值相同的高字节，随后每轮细化一个字节。
64 位优先级最多需要 8 轮。这里没有把 clock 截断为 32 位，也不依赖 24 位时间戳上限。
每轮累计被排除的更小桶数量，因此最终同时知道“阈值”和“严格小于阈值的数量”。

`RadixPass()` 使用分片直方图分散原子更新；生产配置中是 8 份，每份 256 个 int32 桶。
`SelectHistogramBucket()` 汇总并用前缀和找到包含目标排名的桶。
这只是选择所需阈值，没有给所有槽位做完整排序。

`CollectVictims()` 随后输出所有严格小于阈值的槽位，再补齐阈值上的平局槽位。
每一组内部按槽位编号排列，与原 ECHO 的结果顺序一致。

### 5.7 写回 miss 清单，把 TopK 转成槽位

`AssignMisses()` 为每个唯一 miss 写入：

```text
slot_to_token[slot] = token
slot_priorities[slot] = clock + 2
miss_tokens[miss] = token
miss_slots[miss] = slot
tokenMap[token] = slot
```

同步后，所有有效 TopK 项都能通过 `tokenMap` 得到最终槽位。
这时 `current_slots` 从临时 token 位置变成供 SFA 使用的槽位编号。

一个完整例子，假设这些 token 均可见且源 block 有效：

```text
capacity = 6，topk = 5，clock = 10
slot_to_token = [10, 20, 30, 40, -1, -1]
priority      = [ 2,  6,  4,  8, -1, -1]
输入 TopK     = [20, 50, 50, 10, -1]

20 命中 slot 1，10 命中 slot 0
50 虽然出现两次，但只有一个唯一 miss，分配空闲 slot 4

输出 current_slots = [1, 4, 4, 0, -1]
输出 miss_count    = 1
有效 miss_tokens   = [50]
有效 miss_slots    = [4]
```

随后 Transfer 只加载一次 token 50 的 KV。
Plan 去重的是搬运任务；`current_slots` 仍保留两个指向 slot 4 的条目，没有删除原 TopK 中的重复项。

## 6. Transfer 怎样找到 Host 源地址

源码见 [sparse_kv_transfer_apt.cpp](../sparse_kv_transfer/op_kernel/sparse_kv_transfer_apt.cpp)。
它遍历有效 miss，根据请求的 block table 计算来源：

```text
logical_block  = token / block_size
offset_in_block = token % block_size
physical_block = block_table[request, logical_block]

source = 本层 Host KV 基址
       + physical_block × block_size × token字节数
       + offset_in_block × token字节数

destination = 本层 resident 基址
            + (row × capacity + slot) × token字节数
```

`host_cache_bases` 虽然名字中带 Host，但传给设备 kernel 的数值必须是设备可访问的地址。
manager 初始化 A5 后端时传入的是 `dvas_k_bases/dvas_v_bases`，CPU 侧使用的 HVA 与这里的 DVA 要区分。
ECHO 不改变现有 MemFabric 池的地址所有权与 PD 传输策略。

Transfer 把 miss 任务分配到 AIV，用 `DataCopyPad`、两个 16 KiB 的队列缓冲进行字节搬运。
它沿用已有 AscendC 搬运实现，没有使用 Plan 的 2048 线程 SIMT 结构。
本次优化提交未改变 Transfer；在相同初始状态和 TopK 序列下，优化前后 ECHO 的搬运安排应一致。

## 7. MTP 在这条链中怎样生效

### 7.1 两个长度从哪里来

在 `sfa_kv_offload.py` 中，`_compute_decode_token_lengths()` 按每个 query 的位置计算可见 KV 长度。
设一个请求本轮包含 q 个 query，本轮最终 KV 长度为 L，第 j 个 query 的局部偏移为 j，
则使用的逻辑是：

```text
visible_seq_len(j) = L - q + j + 1
stable_prefix_len = max(L - q, 0)
```

例如 L=103、q=3：

```text
本轮 query 位置：    [100, 101, 102]
visible_seq_lens：  [101, 102, 103]
stable_prefix_lens：[100, 100, 100]
```

位置从 0 开始，因此第一行的 101 表示可读取位置 0..100。
稳定前缀 100 表示位置 0..99 属于本轮之前的稳定历史；缓存中 100 及以后的旧记录需要失效。
这些值由当前调度元数据决定，不是固定写死在算子里的常量。

### 7.2 草稿被拒绝后，旧缓存怎么处理

假设某轮处理了位置 100、101、102，后续只确认了位置 100，需要从位置 101 重新计算。
位置 101 上新的 token 可能与旧草稿不同，所以它的新 KV 也可能不同。

```text
上一轮 resident：位置 101 → 旧草稿 B 的 KV
下一轮 Host 池： 位置 101 → 新 token X 的 KV
```

如果只检查“resident 是否包含位置 101”，会错误地命中 B。
当下一轮稳定前缀为 101 时，Plan 先作废 resident 中位置 >=101 的旧记录。
之后即使 TopK 仍选择位置 101，也会产生 miss，从已更新的 Host 池重新加载 X 的 KV。

这不是“投机失败后由 Transfer 把 KV 移出去”。当前机制是框架负责后续 KV 的重算与写回，
Plan 作废可能过期的缓存记录，再按需加载；物理槽位可被后续数据覆盖。

SFA 使用的 `resident_seq_lens` 也必须是逐 query 的真实可见长度。
`capacity=4096` 只是存储槽位数量，不能替代序列长度。

### 7.3 skip_topk 为什么还有区别

manager 中的判断是：

```python
reuse_previous_plan = skip_topk and layer_id != self.mtp_layer_id
```

普通 backbone 的复用层保持相同的行映射，沿用 owner Plan 的 miss/slot 结果，
只用本层 Host KV 基址和 resident 目标执行 Transfer。
复用的是搬运安排，不是把不同层的 KV 数据当作同一份数据。

MTP 层在草稿步骤之间可能压紧或重排行，因此即使复用了 Indexer 结果，也重新执行 Plan。
当前实现保留了这个分支，避免将上一轮行号对应的驻留状态直接用于新的行映射。

## 8. UB、GM 与动态内存预算

这里 Plan 的持久状态和大 workspace 位于 NPU 全局内存，代码中用 `__gm__` 指针访问。
UB 是 kernel 执行期间使用的片上临时空间；把优先级和索引表放进 UB，不等于把整个 Host KV 池放进 UB。

配置集中在 [sparse_kv_echo_plan_config.h](op_kernel/sparse_kv_echo_plan_config.h)：

```cpp
constexpr uint32_t PLAN_THREADS = 2048U;
constexpr uint32_t PLAN_WARP_COUNT = PLAN_THREADS / PLAN_WARP_SIZE;
constexpr uint32_t PLAN_UB_BUDGET_BYTES = 112U * 1024U;
```

生产配置的固定部分如下：

| 区域 | 大小 |
| --- | ---: |
| Scan 和控制区 | 544 字节 |
| 8 份、每份 256 桶的 int32 直方图 | 8192 字节 |
| warp/block 的 free/min/max 统计 | 1568 字节 |
| 固定部分合计 | 10304 字节 |

后面按条件附加两块缓存：

1. 优先级快照：`capacity × 8` 字节，按 32 字节对齐；装不下就从 GM 读取。
2. 直接索引表：`maxToken × 4` 字节，按 32 字节对齐；剩余空间不足时使用 GM 分片中的表。

| capacity | maxToken | 优先级快照 | 直接索引表 | 动态 UB 总量 |
| ---: | ---: | --- | --- | ---: |
| 4096 | 8192 | UB | UB | 75840 字节，约 74.06 KiB |
| 4096 | 262144 | UB | GM | 43072 字节，约 42.06 KiB |
| 8192 | 16384 | UB | GM | 75840 字节，约 74.06 KiB |
| 16384 | 32768 | GM | GM | 10304 字节，约 10.06 KiB |

`maxToken` 会影响直接索引表的大小与存储位置，但不是超出 UB 就无法运行。
回退后仍使用同一套精确选择规则。

Host tiling 调用公共函数 `LocalMemoryBytes(capacity, maxToken)`，并通过 `SetDynUBufSize()` 声明空间。
Kernel 使用同一份头文件判断缓存位置，避免 Host 按 GM 路径分配、Kernel 却按 UB 路径访问的分歧。
112 KiB 是本实现选择的显式动态 UB 预算，不是“Ascend 950 所有使用方式下只有 112 KiB UB”的硬件结论。

## 9. 并行划分、同步和图模式

Plan 的 block 数由最大行数、64 的配置上限和设备 AIV 数共同限制。
一个 AIV 处理一行，行数多时按 `rowStart + n × rowStride` 继续处理其他行。
行内由 2048 个 SIMT 线程协作，槽位和 TopK 按线程步长遍历，不存在跨 AIV 的缓存状态汇总。

主要写后读关系均通过 block barrier 连接：

```text
初始化 TopK/map → 选举首次位置 → 发布 resident 命中
→ 压紧 miss → 选择 victims → 写入新映射 → 输出 current_slots
```

`BlockExclusiveScan()` 内保留两个 barrier，调用者读完 rank/total 后再同步，才能复用 Scan 区域。
本版删除的是这条链中重复的同步，不是取消写后读依赖。
整个算子还依赖 Plan、Transfer、SFA 之间正确的流顺序。

图模式方面，本分支具备设备侧状态更新、稳定的状态 tensor 地址、设备侧 `active_rows`、
Torch mutation schema 和 Meta 实现等基础条件。Plan 不通过 Python 回读 miss 数或淘汰阈值。

但是，这不等于完整 Attention 主链已经验证入图。
本次优化没有调整 `mla_forward` 等图切点，也没有改造整个 SFA 元数据准备链。
Python manager/SFA 仍有自己的准备与调用开销；不能把 Plan 的设备优化直接等同于消除 profile 中全部 Free 时间。
完整捕获和 replay 是否正确、是否提高吞吐，仍需用实际框架版本与请求负载验证。

## 10. 性能改善有什么依据

这一版重点减少重复全行扫描、GM 原子操作，以及重复的 block barrier。
全命中时不进入淘汰选择；新请求的空行也可以直接分配槽位。

为了比较实现成本，固定如下样例：

```text
capacity=4096，topk=2048，1946 个唯一 hit，102 个唯一 miss
resident 已满，clock=20000
2150 个非命中 resident 的 priority 为 20000-4096+s（s=0..2149）
其余命中项的 priority 更新为 20001
```

| 源码层面的指标 | ECHO 第一版 | echo_perf |
| --- | ---: | ---: |
| 阈值细化轮数 | 14 轮二分 | 2 轮 radix |
| 整行显式 block barrier，按 2048 线程计算 | 82 | 27 |
| 建表与命中更新的原子调用 | 14234 次，均在 GM | 4096 次，位置随直接索引表而定 |
| 新增的直方图原子调用 | 无 | 2374 次，位于 UB |

当 `maxToken=8192` 时，新版的 4096 次映射原子调用也在 UB；
长上下文装不下直接索引表时，这 4096 次仍在 GM。

旧版阈值计数阶段包括 16 次全行逻辑读取，合计 512 KiB。
新版优先级 UB 路径先加载一份 32 KiB 快照，后续选择读取 UB。
这不代表整个 kernel 的 GM 访问只有 32 KiB，也不能把逻辑字节数当成实际 HBM 流量。

radix 仍有直方图清零、原子更新、归约和同步成本；UB 使用增多也可能影响实际资源安排。
上述变化给出了减少开销的理由，但尚不能给出加速倍数，也不能断言一定快于原来的哈希 SIMT Plan。

## 11. 本地测试证明了什么

在仓库根目录执行：

```bash
python tests/ut/kv_offload/test_sparse_kv_echo.py -v
```

实现提交 `6530f3e93` 的 3 项测试已通过：

| 测试 | 主要覆盖 |
| --- | --- |
| `test_actual_row_code` | 抽取实际 kernel 行处理代码，以 32/64 个 CPU 线程模拟原子、shuffle、barrier，并用 ASAN/UBSAN 检查 |
| `test_production_ub_layout` | 使用真实 2048 线程配置头，检查固定尺寸及容量边界的预算与对齐 |
| `test_workspace_layout` | 检查 Python 分配的 GM workspace 大小 |

CPU 参考算法独立排序优先级，核对淘汰集合和输出顺序，还验证 miss 清单、状态及模拟加载后的 KV payload。
用例包含重复/无效 TopK、无效请求/block、脏 workspace、全命中、请求换行、MTP 后缀重写、
多层结果复用、零 active 行，以及 UB/GM 的四种缓存组合。
还覆盖了 64 位优先级分布与 UB 临界容量两侧。

源码见 [Python 测试入口](../../../tests/ut/kv_offload/test_sparse_kv_echo.py)
和 [C++ CPU 执行模型](../../../tests/ut/kv_offload/sparse_kv_echo_host.cpp)。

这些检查支持算法和索引边界的正确性判断，但 CPU 模型不是 A5 仿真器。
它没有验证 CANN 地址空间编译、真实 2048 线程资源使用、Transfer DMA、SFA 数值结果及图 replay。
先前的静态同步脚本未识别这里的 SIMT builtin，因此“脚本没有报错”也不能当成同步正确性的证明。

## 12. 建议怎样阅读和上机验证

读代码可以按下面的顺序：

1. `sparse_kv_echo.py`：先看数据怎么分配，认清 state、workspace、输出。
2. `ProcessRuntimeRow()`：掌握一行完整的处理顺序。
3. `ResetResident()`、`MarkAndCompactMisses()`、`PublishResidentHits()`：理解有效性、命中与重复 token。
4. `GatherPriorityStats()`、`EvictionThreshold()`、`RadixPass()`、`CollectVictims()`：理解淘汰规则与优化。
5. `AssignMisses()` 和 Transfer：看 token 位置怎样变成 resident 槽位和真实 KV 数据。
6. SFA 的长度计算及 manager 的 `skip_topk` 分支：把 MTP 和跨层复用接回整条流程。
7. Host tiling、公共配置头、Torch schema/Meta：检查工程接入和资源约束。

在服务器验证时，先完整重编本分支的 Host tiling 和 kernel，使用该服务器已经验证过的 A5 构建配置。
不要将旧版 tiling 库与新版 kernel 混装，因为动态 UB 的需求已经改变。

建议先在同一组初始状态和 TopK 序列下比较 `echo` 与 `echo_perf` 的输出和状态，
再比较单算子时间，最后验证 PD、MTP 和图模式端到端行为。
尺寸至少覆盖冷行、全命中、满缓存少量 miss、大量 miss、长上下文和 UB/GM 切换边界。
MTP 至少覆盖部分接受、全部拒绝以及请求行重排。

最终性能应同时看 Plan 时间、Transfer 时间、Indexer→SFA 整段时间和推理吞吐，
避免仅凭某个 kernel 更短或某条 stream 空白减少就作结论。

## 修订记录

| 日期 | 修订人 | 内容 |
| --- | --- | --- |
| 2026-10-07 | hjh（AI 辅助整理） | 基于 `6530f3e93` 增加中文代码说明，解释调用链、MTP、radix/UB 优化和验证边界 |
