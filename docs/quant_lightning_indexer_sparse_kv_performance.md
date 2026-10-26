# QuantLightningIndexerSparseKv 性能对比

比较融合算子与原 `torch_npu.npu_quant_lightning_indexer` +
`torch.ops._C_ascend.npu_sparse_kv_plan` 的合计耗时。目标为 Ascend 950、CANN 9.1.0，
使用已经通过正确性测试的算子包和 GLM-5.2 C8 张量规格，无需模型权重。
这次修改仅涉及 Python 测试脚本，已有正确性验证环境更新代码后无需重编算子。

计时范围包含 TopK、hit/miss 判断、LRU 和 miss 元数据更新，以及保留给 IndexCache 的
TopK 输出；不包含 DDR load、attention、模型其他层或端到端吞吐。

## 执行命令

在已配置的 950 环境、仓库根目录执行。默认设备 0，可添加 `--device N`；
CANN 安装位置不同时传 `--cann-root PATH`。每次使用不同的输出目录。

```bash
# 先快速验证性能脚本：8 个配置，含 batch 32、MTP 6 和冷/预热缓存
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite benchmark --batches 1,32 --seq-lens 140000 --mtps 1,6 \
  --warmup 3 --rounds 2 --iterations 3 --output test-results/perf-check

# 重点 140k：batch 每个整数 1～32，MTP 1/2/4/6，冷/预热缓存，共 256 个配置
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite benchmark --seq-lens 140000 --output test-results/perf-140k-eager

# 默认完整矩阵：额外包含 4096 和 32768 长度，共 768 个配置
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite benchmark --output test-results/perf-full-eager

# 140k ACLGraph 对比：两条路径分别捕获完整计算，记录 replay 耗时
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite benchmark --seq-lens 140000 --execution graph \
  --output test-results/perf-140k-graph

# 更多长序列与 planner capacity：可按需要缩小 batch 或 MTP 列表
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite benchmark --batches 1,2,4,8,16,32 \
  --seq-lens 16384,65536,131072,140000,143360 --mtps 1,2,4,6 \
  --capacities 4096,8192 --output test-results/perf-long-capacity

# 只检查矩阵与参数，不导入 torch/NPU；Mac 也可执行
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite benchmark --dry-run --output test-results/perf-plan
```

`--batches`、`--seq-lens`、`--mtps`、`--capacities` 支持逗号列表及闭区间，
例如 `--batches 1-32`。默认参数如下：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| batches | `1-32` | 请求数，每个整数都覆盖 |
| seq-lens | `4096,32768,140000` | 每个请求的上下文长度；140k 指 140000 token |
| mtps | `1,2,4,6` | 每请求 query 行数，1 为普通 decode；总行数为 batch × MTP |
| capacities | `2048` | 可选 2048/4096/7168/7169/8192，包含 UB/GM scratch 边界 |
| cache-states | `warm,cold` | planner 驻留状态，可单独指定其中一种 |
| execution | `eager` | 可选 `graph`，两种模式分别比较，不混合汇总 |
| warmup | `10` | 每条路径、每个配置的非计时预热次数 |
| rounds | `6` | 配对轮数，路径顺序交替 |
| iterations | `20` | 每轮每条路径的计时调用数 |
| seed | `2026` | 合成输入随机种子 |

例如 batch 32、MTP 6 对应 `T=192`，query 为 `[192,32,128]`。
每个请求使用独立、倒序映射的物理 key blocks；不会因请求共享同一组 KV 页而夸大缓存复用。
batch 32、140000 token 的 FP8 key 本身约 574 MB，另需 scale、planner 状态和算子 workspace。
矩阵一次只持有一个配置，图对象在测完该配置后释放。

## 计时口径与公平性

每个配置先用相同输入检查两条路径的 TopK 和全部七项 planner 状态，并用 CPU oracle
验证状态更新。graph 模式还会检查捕获后的 replay 与 eager 结果一致。校验失败则不输出该配置的耗时。

- `cold`：planner 从初始空驻留状态开始，每次调用都会重新恢复此状态。
- `warm`：先执行一次基线路径，保存更新后的驻留/LRU 快照，每次调用恢复该快照。
  MTP 的非稳定后缀可能继续 miss，因此同时报告实际 `hit_rate` 和 miss 数，不假定全 hit。
- 两条路径使用独立状态张量、同一份只读快照。每次调用前用 `copy_` 恢复七项语义状态，
  同步完成后才开始计时；`compact_workspace` 为算子自行初始化的 scratch，不复制。
  状态复制会影响硬件缓存，但两条路径采用相同过程。cold/warm 不表示清空或预热 HBM/L2。
- 预热、输入生成、状态恢复、正确性检查、结果写盘和 trace 均位于计时区间外。
  算子句柄和基线 rank3 key scale 视图在计时前绑定。
- 每轮按 fused→baseline、baseline→fused 交替执行；每次调用都有独立的 start/end Event，
  等待 end 完成后读数。默认 6 轮保证先执行次数相等；也允许奇数轮，JSON 保留
  `round_order` 及两条路径各自先执行的次数。
- Event 时间是 NPU 时间线上的调用区间，eager 下可能包含主机下发间隙，不等于逐 kernel
  时间之和。另报主机 wall 时间，明确包含 Event 记录和同步等待的测量开销。
- graph 模式分别捕获“一次融合算子”与“QLI 后接 SparseKvPlan”，保持输入、输出和状态地址稳定。
  捕获、恢复均不计时，只测 replay；捕获失败会报错，不会用 eager 时间替代。

默认每个配置、每条路径收集 6 × 20 = 120 次调用。
先计算每轮 20 次调用的平均值，再在 6 个轮均值上计算 median、P90、min/max、标准差。
**P90 是轮均值的 P90，不是请求级尾延迟。** 加速比定义为基线轮均值中位数除以融合轮均值中位数，
大于 1 表示融合更快；另保留同轮比值，便于判断顺序或波动影响。不跨不同形状计算“整体加速比”。

## 输出与失败处理

| 文件 | 内容 |
| --- | --- |
| `plan.json` | 完整矩阵、采样配置、计时口径 |
| `environment.json` | Git revision、dirty 状态、tracked diff 哈希、设备、torch/torch_npu、扩展位置、CANN 版本文件 |
| `results.csv` | 每配置一行，适合直接导入表格或绘图 |
| `results.json` | 相同结果及逐次耗时、原始轮均值、执行顺序、同轮加速比 |
| `summary.md` | 逐配置耗时、加速比和失败项 |
| `status.json` | planned/running/complete/failed、计划与已完成数量 |

重点查看 CSV 的以下字段：

- `batch,seq_len,mtp,query_rows,capacity,cache_state,execution`
- `fused_round_mean_median_ms`、`baseline_round_mean_median_ms`
- `event_speedup`，以及两条路径的 `*_round_mean_p90_ms`、`*_round_mean_stdev_ms`
- `fused_wall_round_mean_median_ms`、`baseline_wall_round_mean_median_ms`、`wall_speedup`
- `miss_count_total,selected_count_total,hit_rate,correctness_validated,status`

每完成一个配置就原子更新结果文件。失败会记录错误并返回非零，之前完成的数据保留；
未执行的配置不伪造耗时。中断后不要把 `status=running` 的部分数据当作完整矩阵。
已有 `results.json` 的目录拒绝覆盖；缩小未完成参数范围后使用新的输出目录继续执行。
`--dry-run` 只产生计划与状态文件，不生成性能数据。

单点 trace 继续使用原入口，现已使用同一计时实现，默认 warm 状态：

```bash
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite profile --batch 32 --seq-len 140000 --mtp 6 --capacity 4096 \
  --rounds 6 --iterations 20 --trace --output test-results/perf-trace
```

`profile/profile.json` 保留旧的 scalar 字段，`profile/results.json` 保存完整统计。
trace 在额外的独立 eager 调用中采集；即使计时选择 graph，trace 仍明确标记为 eager。
可检查融合路径的 mixed kernel 与基线的 QLI、SparseKvPlan 两段，以及其间空隙。

Mac 上只能验证矩阵、统计、文件输出、CLI 和计时流程的测试桩。
真实性能数据、ACLGraph 捕获支持和加速比以你在 950 上运行的结果为准。

2026-10-26 本地验证：集成 `--suite static` 66 项通过，包含新增性能脚本的矩阵、
统计、错误记录、cold/warm 恢复、计时顺序、图清理与输出生命周期回归。
10 个 Python 文件通过 Ruff lint/format、AST/编译检查，140k dry-run 生成 256 个配置。
仓库级 `bash format.sh ci` 因本机未安装 `pre-commit` 未能执行；上述 Ruff 为独立检查。
