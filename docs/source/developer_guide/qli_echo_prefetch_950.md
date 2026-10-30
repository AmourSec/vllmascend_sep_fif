# GLM-5.2：950 上的 indexer 预取

本改动把 [ECHO §5.1、§5.3](https://www.usenix.org/system/files/osdi26-liu-guangda.pdf)
中的 decode 阈值预取接到现有 sparse KV offload 路径。miss 检查、LRU、slot
分配和 DRAM→HBM 搬运仍由 `SparseKVSimtLru` 的
`npu_sparse_kv_plan_transfer` 完成，没有引入第二套缓存管理算法。

本实现尚未在 NPU 950 上编译、执行或测量。Mac 上的检查只能验证可移植逻辑、
接口和 Python 语法，不能证明 CANN 编译通过、设备同步正确或性能提升。

## 算法与执行顺序

每个 indexer 层维护自己的第 k 大分数 EMA。首次运行或没有可用历史时使用
`+inf`，本轮不做预测预取；精确 top-k 完成后初始化历史。EMA 的平滑系数是
0.5，下一轮阈值为 `0.5 * previous + 0.5 * current_kth`。

新增 `npu_qli_prefetch_stage` 对应的原生算子位于相邻仓库
`ops-transformer/attention/quant_lightning_indexer_prefetch`，直接引用
`ops-transformer/attention/quant_lightning_indexer/op_kernel/arch35` 的 Cube、
加权 ReLU 归约和精确 top-k 实现。关闭预取时继续走
`torch_npu.npu_quant_lightning_indexer`，不以 vLLM 内的同名实验分支作基线。
两个阶段共同扫描一遍 K：

1. 阶段 0 计算前半段 128-token tile，保存分数，并收集 `score > threshold`
   且位于稳定前缀内的 token。候选总数不超过
   `min(2048, ceil(seq_len / indexer_prefetch_budget_divisor))`；剩余位置填 `-1`。
2. 预取流等待阶段 0，使用现有 Plan/Transfer 将候选 KV 搬入该层的热 buffer。
3. 计算流执行阶段 1，计算剩余 tile，然后对完整分数运行原来的精确 top-k。
4. 计算流等待预取流完成，再用精确 top-k 执行现有 Plan/Transfer，补齐 miss，
   最后执行 attention。

```text
计算流：当前 KV 写回 → 前半段 indexer → 后半段 indexer + 精确 top-k → 等待 → 最终召回 → attention
                                     │                                ↑
预取流：                             └→ 原有 SIMT Plan + Transfer ─────┘
```

候选可以漏选，也可以包含最终 top-k 以外的 token。正确性来自最终的精确选择和
完整召回，不能把预测阈值当成 attention 的选择规则。候选规划与最终规划严格串行
访问同一层 LRU 状态，最终规划会覆盖候选阶段的临时 slot 输出。

这是 ECHO 算法在现有独立管理算子接口上的分阶段实现：候选筛选集成在 indexer
中，搬运仍为独立算子。它使用一次阶段边界建立流依赖，没有复现 CUDA warp
specialization 的逐 tile 搬运流水线。两条流只提供并发执行条件；实际重叠需要
设备 profiler 证明。新内核为管理算子预留部分计算资源，代价也需要一起测量。

## 必须保持的约束

- 中间分数按 query 行保存，不能沿用跨行复用的 per-core 临时区，否则阶段 1
  会读取另一行覆盖后的前半段分数。
- arch35 中间分数是可排序的 BF16 位编码。必须还原成浮点分数后更新 EMA。
  top-k 输出没有排序保证，不能取最后一个索引对应的分数作为第 k 大值。
- 预取只选择 `token < stable_prefix_len` 的历史 KV，不提前读取当前生成 token
  或可能被 MTP 重写的尾部。
- GLM-5.2 共享 indexer 的层继续复用精确 top-k；开启该功能后，每层独立运行
  现有 Plan/Transfer。只复用 owner 层的最终 miss 列表会漏掉已被 owner 预取、
  但尚未存在于其他层热 buffer 中的数据。
- 该路径使用持久 buffer 和设备侧数据依赖，热路径不读取 NPU tensor 到 CPU。
  流必须在最终召回前汇合，graph capture/replay 也必须覆盖这组依赖。
- 沿用调度器的有效请求连续排列、补齐行的 sequence length 为 0 的约定；分页表
  由调度器提供，输入数值须满足原 quant indexer 的有限数值约定。

## 范围与开关

目标是 GLM-5.2 的标准 FP8 C8 indexer，BF16 weights、FP32 scale，
64 个 query head、head dimension 128、
top-k 2048、TND query 和 paged KV，运行于 Ascend 950 的 rank-local SIMT offload
后端。功能默认关闭。

在已有启动命令的 `--additional-config` 中合并以下字段，保留原来的 PD、模型、
量化和缓存配置：

```json
{
  "sparse_kv_offload_config": {
    "enabled": true,
    "topk_buffer_size": 4096,
    "use_fused_overlap": false,
    "indexer_prefetch": true,
    "indexer_prefetch_budget_divisor": 32
  }
}
```

`indexer_prefetch_budget_divisor` 必须为正整数。增大会减少候选搬运量，减小会增加
预取机会及 cache 污染；最终 top-k 的大小始终为 2048。请先用默认值建立基线。

纯 decode 且每个 request 只有一个 query token 时使用新路径。prefill、混合批次、
MTP/多 token decode、非 FP8 或不支持的维度保持原 indexer 路径。该改动不扩展
GLM-Next 压缩 KPool indexer，也不改变现有 PD offload 对 prefill 的限制。不能与
`use_fused_overlap` 同时开启。

中间 score buffer 的空间约为
`max_num_topk_rows * align(max_model_len, 128) * 2` 字节；它跨层复用，EMA 历史
按层保存。比较内存利用率时需要计入这部分 HBM。

## 在 950 上构建

使用当前已验证的 CANN、TorchNPU、MemFabric 和 vLLM 环境，不要为了本改动升级
它们。复制两个仓库到同一父目录；以下命令在该父目录的 **Bash** 中执行。
先激活原来的 Python/CANN 环境。两套源码分别构建，vLLM 中不再编译重复的预取内核。

```bash
set -eo pipefail
OPS_ROOT="$PWD/ops-transformer"
VLLM_ASCEND_ROOT="$PWD/vllmascend_sep_fif"
QLI_INSTALL="$OPS_ROOT/_install_qli_prefetch"
QLI_RESULTS="$VLLM_ASCEND_ROOT/benchmarks/results/qli-prefetch-950"
test -f "$OPS_ROOT/build.sh"
test -f "$VLLM_ASCEND_ROOT/setup.py"
mkdir -p "$QLI_RESULTS"
npu-smi info | tee "$QLI_RESULTS/npu-smi.txt"
python -m pip freeze > "$QLI_RESULTS/python-packages.txt"

cd "$OPS_ROOT"
bash build.sh --pkg --soc=ascend950 \
  --ops=quant_lightning_indexer,quant_lightning_indexer_prefetch \
  --vendor_name=qli_prefetch -j16 \
  2>&1 | tee "$QLI_RESULTS/build-ops.log"
bash "build_out/cann-ops-transformer-qli_prefetch_linux-$(uname -m).run" \
  --install-path="$QLI_INSTALL"
source "$QLI_INSTALL/vendors/qli_prefetch_transformer/bin/set_env.bash"

cd "$VLLM_ASCEND_ROOT"
# setup.py 在 SOC_VERSION 未设置时通过 npu-smi 检测实际型号。
# 如果已有该变量，确认其值属于当前 950 机器。
export COMPILE_CUSTOM_KERNELS=1
python -m pip install -v -e . --no-build-isolation --no-deps \
  2>&1 | tee "$QLI_RESULTS/build-vllm-ascend.log"
```

`ops-transformer` 的 `--soc=ascend950` 是芯片系列；vLLM 构建所用的 `SOC_VERSION`
可以是探测出的具体 SKU。请不要把其他机器遗留的 `ascend910*` 值带入本次构建。
多机部署需要在各个 D worker 上安装同样的两个构建产物，并在启动前 source 新 vendor
环境。旧 Python worker 必须重启，ACL 函数地址和库搜索路径会被缓存。

检查原生导出和 Torch 注册。仅看到 Torch schema 不代表原生 kernel 已安装：

```bash
nm -D "$QLI_INSTALL/vendors/qli_prefetch_transformer/op_api/lib/libcust_opapi.so" \
  | grep -E 'aclnnQuantLightningIndexer(Prefetch)?(GetWorkspaceSize)?$'
python - <<'PY'
import ctypes
import os
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.utils import bootstrap_custom_op_env
bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C
from vllm_ascend._build_info import __device_type__
assert __device_type__ == 'A5', __device_type__
matches = []
for root in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':'):
    if not root:
        continue
    lib = Path(root) / 'op_api/lib/libcust_opapi.so'
    if not lib.is_file():
        continue
    handle = ctypes.CDLL(str(lib))
    if hasattr(handle, 'aclnnQuantLightningIndexerPrefetch'):
        assert hasattr(handle, 'aclnnQuantLightningIndexerPrefetchGetWorkspaceSize')
        matches.append(str(lib.resolve()))
assert len(matches) == 1, f'预取符号缺失或重复: {matches}'
assert 'qli_prefetch_transformer' in matches[0], matches
print('prefetch native library:', matches[0])
print(torch.ops._C_ascend.npu_qli_prefetch_stage)
print('baseline:', torch_npu.npu_quant_lightning_indexer)
PY
```

若检查发现旧的 `vllm_ascend/_cann_ops_custom` 也导出预取符号，说明仍在加载上一次
构建的旧包。先确认本轮 editable build 完整成功，再重新 source 环境并重启 Python；
不要带着两个同名原生实现继续比较。

## 正确性与性能验收

先运行新增算子测试，再对同一批 FP8/BF16 输入比较原生基线与分阶段实现：

```bash
cd "$VLLM_ASCEND_ROOT"
python -m pytest -v \
  --confcutdir=tests/e2e/nightly/single_node/ops/singlecard_ops \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_qli_prefetch.py \
  2>&1 | tee "$QLI_RESULTS/operator-tests.log"

for threshold in cold warm low; do
  python benchmarks/benchmark_qli_prefetch.py \
    --rows 2 --seq-len 32769 --threshold-mode "$threshold" --seed 41 \
    2>&1 | tee "$QLI_RESULTS/op-${threshold}.log"
done

for rows in 1 8 32; do
  for length in 8192 32768 65536 131072; do
    for repeat in 1 2 3; do
      python benchmarks/benchmark_qli_prefetch.py \
        --rows "$rows" --seq-len "$length" --threshold-mode warm --seed 41 \
        2>&1 | tee "$QLI_RESULTS/op-r${rows}-s${length}-${repeat}.log"
    done
  done
done
```

950 上不应因缺少 `torch_npu` 而跳过 NPU 测试。`cold` 使用无历史的无限阈值；
`warm` 使用上一轮精确 kth；`low` 使用负无限阈值以覆盖候选预算边界。
微基准核对原生 TorchNPU QLI 的 top-k（允许边界同分项），并报告
`deployed_compute_us`、`staged_compute_us` 和 `compute_overhead_ratio`。
比值小于 1 表示 indexer 计算部分更快；它不包含 SIMT KV 搬运，不能证明完整
KV offload 获得加速。不要加 `--verify` 运行 NPU 测试，该选项只执行 CPU 参考。

完整模型对比请使用相同模型权重、输入、并发、生成长度、量化、热 buffer 容量和
graph 设置，仅切换 `indexer_prefetch`：

1. **选取与数据正确性**：比较新旧 indexer 的有效 top-k 集合；分数相同的边界
   token 需按 tie 规则核对。比较最终 slot 对应的 KV 字节和 attention 输出。
   覆盖空/短序列、128 tile 边界、16K top-k 合并边界、多个请求、batch reorder、
   请求退出后行复用、共享 indexer 层及连续 graph replay。
2. **完整召回**：分别制造阈值过高（零候选）、过低（达到预算）、热命中和冷 miss。
   任何候选误差都不能造成最终 top-k 缺失，attention 前不能有未完成搬运。
3. **共享层**：先让 owner 层预取一组最终选中的 KV，再检查共享层收到各自的
   payload；不能只检查索引和 slot 元数据。
4. **基线矩阵**：至少测试 8K/32K/64K/128K context，及 1/8/32 等机器可承受的
   并发。先 eager，再 graph；记录预热轮数、测试轮数和硬件/软件版本。
5. **测量内容**：记录端到端 ITL p50/p95、输出 tokens/s、HBM 峰值、indexer 总耗时、
   候选/最终 miss 数、搬运字节数和实际重叠区间。使用 NPU event 计时并在测量边界
   synchronize，排除首次编译、初始化和 graph capture。

若 profiler 中 Plan/Transfer 与 suffix indexer 串行执行，即使 Python 使用了两条
stream，也应记为“未实现硬件重叠”。需检查目标 SKU 的 AIC/AIV 数量、资源占用、
调度限制与分阶段启动开销后，再决定如何调整分核或候选预算。

## 完整 GLM-5.2：关闭/开启预取的 A/B 命令

复用已经跑通的 PD 拓扑、MemFabric 配置和模型启动脚本。下面唯一需要用户提供的
部署信息是模型路径、对外服务地址、served model 名称，以及原 D 节点启动脚本。
不要把单机普通 `vllm serve` 当作已有 PD offload 的等价替代。

先把 D 节点原有 `--additional-config` 的完整 JSON 保存为
`$QLI_RESULTS/base-additional.json`，然后生成两份配置。已有其他字段全部保留：

```bash
export QLI_RESULTS
python - <<'PY'
import copy
import json
import os
from pathlib import Path
root = Path(os.environ['QLI_RESULTS'])
base = json.loads((root / 'base-additional.json').read_text())
assert base['sparse_kv_offload_config']['enabled'] is True
for mode in ('off', 'on'):
    config = copy.deepcopy(base)
    sparse = config['sparse_kv_offload_config']
    sparse['use_fused_overlap'] = False
    sparse['indexer_prefetch'] = mode == 'on'
    sparse['indexer_prefetch_budget_divisor'] = 32
    (root / f'{mode}.json').write_text(json.dumps(config, indent=2))
PY
```

在原 D 节点启动脚本中，将原有 `--additional-config` 参数替换为这一项：

```bash
--additional-config "$(cat "$CONFIG_PATH")"
```

两组都移除 `--speculative-config`，因为当前预取路径不支持 MTP；保持相同的
`--max-model-len`、热 buffer 容量、TP/DP、prefix caching 和量化设置。先在两组中
都加 `--enforce-eager` 完成正确性检查，再都移除它测试原 graph 配置。
测试 32K 输入加 256 输出时，`max-model-len` 至少为 33024。P 节点保持原设置，
不要给 P 节点加入 sparse offload 配置。

以下客户端脚本会保存输出文本和性能 JSON。请在客户端设置实际值；`BASE_URL`
指向 PD 对外入口，`MODEL_PATH` 是客户端可读取 tokenizer 的本地模型目录。
参数按 [vLLM 0.26 bench serve](https://docs.vllm.ai/en/v0.26.0/cli/bench/serve/)
核对；可用 `vllm bench serve --help` 检查现有环境的版本。

```bash
export BASE_URL='http://127.0.0.1:8000'   # 替换为 PD 对外入口
export MODEL_PATH='/models/GLM-5.2-C8'     # 替换为实际模型目录
export SERVED_MODEL='glm-5.2'             # 与服务端 --served-model-name 一致
cat > "$QLI_RESULTS/run_clients.sh" <<'SH'
#!/usr/bin/env bash
set -eo pipefail
label="${1:?传入 off 或 on}"
: "${QLI_RESULTS:?}" "${BASE_URL:?}" "${MODEL_PATH:?}" "${SERVED_MODEL:?}"
common=(
  --backend vllm --base-url "$BASE_URL" --endpoint /v1/completions
  --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" --served-model-name "$SERVED_MODEL"
  --dataset-name random --random-range-ratio 0 --seed 2026
  --temperature 0 --ignore-eos --request-rate inf --num-warmups 4
  --random-output-len 256 --percentile-metrics ttft,tpot,itl
  --metric-percentiles 50,95,99 --save-result --save-detailed --result-dir "$QLI_RESULTS"
)
vllm bench serve "${common[@]}" \
  --random-input-len 8192 --num-prompts 8 --max-concurrency 1 \
  --result-filename "${label}-correctness.json" \
  2>&1 | tee "$QLI_RESULTS/${label}-correctness.log"
for length in 8192 32768; do
  for concurrency in 1 8 32; do
    for repeat in 1 2 3; do
      vllm bench serve "${common[@]}" \
        --random-input-len "$length" --num-prompts 128 --max-concurrency "$concurrency" \
        --result-filename "${label}-s${length}-c${concurrency}-r${repeat}.json" \
        2>&1 | tee "$QLI_RESULTS/${label}-s${length}-c${concurrency}-r${repeat}.log"
    done
  done
done
SH
```

分别启动两组，不能在同一组正在测量时运行另一组。`your_decode_start.sh` 是你已有的
启动脚本，需替换路径；它应在启动所有 D worker 时使用当前 `CONFIG_PATH` 和 vendor 环境。
服务端在独立终端运行，确认服务 ready 后在客户端终端执行下一条：

```bash
# 服务端终端：先用关闭预取的配置启动。
source "$QLI_INSTALL/vendors/qli_prefetch_transformer/bin/set_env.bash"
CONFIG_PATH="$QLI_RESULTS/off.json" bash /absolute/path/your_decode_start.sh

# 客户端终端：关闭预取的正确性与性能数据。
bash "$QLI_RESULTS/run_clients.sh" off

# 按原部署方式停止并重启 D 节点，改为开启预取。
source "$QLI_INSTALL/vendors/qli_prefetch_transformer/bin/set_env.bash"
CONFIG_PATH="$QLI_RESULTS/on.json" bash /absolute/path/your_decode_start.sh

# 客户端终端：开启预取的数据。
bash "$QLI_RESULTS/run_clients.sh" on
```

比较同一输入的生成文本及输出长度，并汇总三次运行的中位数：

```bash
python - <<'PY'
import json
import os
import statistics
from pathlib import Path
root = Path(os.environ['QLI_RESULTS'])
def read(name, expected):
    value = json.loads((root / name).read_text())
    assert value['completed'] == expected and value.get('failed', 0) == 0, name
    assert not any(value.get('errors', [])), name
    return value
off = read('off-correctness.json', 8)
on = read('on-correctness.json', 8)
assert off['input_lens'] == on['input_lens'], '输入长度不一致'
assert off['output_lens'] == on['output_lens'], '输出长度不一致'
assert len(off['generated_texts']) == len(on['generated_texts']) == 8
different = [i for i, (a, b) in enumerate(zip(off['generated_texts'], on['generated_texts'])) if a != b]
assert not different, f'输出文本不一致，需要定位请求 {different}'
print('PASS: 8 个确定性请求的输出文本及长度一致')
print('ISL,concurrency,off_tok_s,on_tok_s,throughput_gain_pct,off_itl_p50_ms,on_itl_p50_ms,off_itl_p95_ms,on_itl_p95_ms')
for length in (8192, 32768):
    for concurrency in (1, 8, 32):
        metrics = {}
        for mode in ('off', 'on'):
            samples = [read(f'{mode}-s{length}-c{concurrency}-r{r}.json', 128) for r in (1, 2, 3)]
            metrics[mode] = [statistics.median(s[key] for s in samples)
                             for key in ('output_throughput', 'median_itl_ms', 'p95_itl_ms')]
        a, b = metrics['off'], metrics['on']
        assert a[0] > 0
        print(f'{length},{concurrency},{a[0]:.3f},{b[0]:.3f},{100*(b[0]/a[0]-1):.2f},'
              f'{a[1]:.3f},{b[1]:.3f},{a[2]:.3f},{b[2]:.3f}')
PY
```

文本一致是模型级回归检查，不代替逐层 KV payload/attention 输出核对。若文本不一致，
先重复两次 off/off 排除已有非确定性，再结合算子测试检查 top-k 边界同分项、共享层
payload 和流依赖，不应仅凭文本差异认定预取搬运错误。

## 确认模型实际走了预取路径

正式计时不启用 profiler。另开一次短测试，在 D 节点原启动命令加上：

```bash
--profiler-config.profiler=torch \
--profiler-config.torch_profiler_dir=/tmp/qli-prefetch-profile
```

使用 D 节点管理 API 开关采集，请将地址改成该 D 服务地址；不要假设 PD 代理转发
`start_profile`。多 D 服务需分别开启采集。随后通过正常 PD 对外入口发送请求：

```bash
DECODE_ADMIN_URL='http://127.0.0.1:8100'  # 替换为 D 服务地址
curl --fail -X POST "$DECODE_ADMIN_URL/start_profile"
vllm bench serve --backend vllm --base-url "$BASE_URL" --endpoint /v1/completions \
  --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" --served-model-name "$SERVED_MODEL" \
  --dataset-name random --random-input-len 8192 --random-output-len 64 \
  --random-range-ratio 0 --seed 2026 --temperature 0 --ignore-eos \
  --num-prompts 2 --max-concurrency 1
curl --fail -X POST "$DECODE_ADMIN_URL/stop_profile"
grep -R -l -E 'QuantLightningIndexerPrefetch|quant_lightning_indexer_prefetch|npu_qli_prefetch_stage' \
  /tmp/qli-prefetch-profile --include='*.json' --include='*.csv'
```

on trace 必须出现预取算子，off trace 不应出现。若 on 中没有，检查 MTP、BF16
weights、FP8 K、纯 decode batch 和形状限制是否导致回退。最后在时间线上核实候选
Plan/Transfer 与 stage 1 的交叠，以及最终 Plan/attention 前的等待；名称检索只能
证明调用存在，不能证明硬件重叠。

## Mac 上可运行的检查

以下检查不依赖 CANN、TorchNPU 或 vLLM，仅需要 CPU PyTorch、pytest 和 C++ 编译器：

```bash
python3 tests/ut/attention/test_indexer_prefetch.py
python3 tests/ut/attention/test_indexer_prefetch_cpu.py
python3 -m pytest --confcutdir=tests/e2e/nightly/single_node/ops/singlecard_ops \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_qli_prefetch.py -k 'not npu'
clang++ -std=c++17 -Wall -Wextra -Werror \
  ../ops-transformer/attention/quant_lightning_indexer_prefetch/op_kernel/tests/qli_prefetch_helpers_test.cpp \
  -o /tmp/qli_prefetch_helpers_test
/tmp/qli_prefetch_helpers_test
```

它们覆盖 EMA 初始化/请求重排、工作区切片、生产 scale 布局、调用顺序、
分阶段边界和 BF16 编码。完整模型的双流 Plan/Transfer、共享层 KV 内容和 graph
replay 仍需按上面的 950 验收矩阵验证。

## 参考

- [ECHO：Fused Lossless Prefetching，§5](https://www.usenix.org/system/files/osdi26-liu-guangda.pdf)
- [AIKG NPU 架构能力核对 skill](https://github.com/AmourSec/aikg/blob/master/skills/npu-arch-capability-check/SKILL.md)
- vLLM 仓库现有 `sparse_kv_plan`、`sparse_kv_transfer` 是缓存管理接口依据。
- `ops-transformer/attention/quant_lightning_indexer/op_kernel/arch35` 是计算基线。
