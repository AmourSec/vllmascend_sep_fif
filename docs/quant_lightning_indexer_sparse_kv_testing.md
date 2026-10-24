# GLM-5.2 C8 indexer 与 KV LRU 融合算子测试

目标环境为 Ascend 950 系列、CANN 9.1.0，以及当前 `clean_simt` 已能运行的
vLLM / torch / torch_npu 环境。本次开发环境为 macOS，未执行 CANN 编译或 NPU 测试。

## 实现范围

新算子为 `torch.ops._C_ascend.npu_quant_lightning_indexer_sparse_kv`。
同一次 mixed AIC/AIV kernel launch 中，每行最终 TopK 合并、补齐完成后，
由同一个 AIV 直接读取 UB 索引，执行原有 SIMT SparseKvPlan 的命中判断、
slot 分配、LRU 和 miss 元数据更新，然后保留 IndexCache 所需的 TopK 写回。
planner 不再从 HBM 重读 TopK 索引，也不再等待整个 QLI 的全核阶段屏障。
`npu_sparse_kv_transfer` 仍由 attention 前的原有位置执行，从 DDR 装载 miss KV。

适用路径为标准 DSA 的 A5 LI C8 decode / speculative decode。共享 TopK 层、
MTP 的共享索引及 row compaction 继续遵循原有 Plan 复用规则。prefill、混合批、
非 FP8 indexer 和已有 fused-overlap 路径保留原有实现。
这里的 C8 指 indexer Q/K FP8；当前 offload 的主 KV latent / RoPE 仍为 BF16。

算子源代码及其所需 upstream 头文件都在本仓库内，编译不依赖兄弟目录的
`ops-transformer`。来源版本和修改记录见
[`csrc/attention/quant_lightning_indexer_sparse_kv/README.md`](../csrc/attention/quant_lightning_indexer_sparse_kv/README.md)。

## 模型规格与用例

规格依据 [GLM-5.2 官方 config.json](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)，
dtype/layout 依据本仓库 A5 `DeviceOperator.indexer_select_post_process` 的实际调用。
测试使用这些规格生成确定性输入，无需下载模型权重。

| 项目 | 规格 |
| --- | --- |
| Query | FP8 E4M3FN `[T, 32, 128]`，T 为所有请求的 query token 总数 |
| Paged key | FP8 E4M3FN `[blocks, 128, 1, 128]` |
| Weights | BF16 `[T, 32]` |
| Query scale | FP32 `[T, 32]` |
| Key scale | FP32 `[blocks, 128]`，也测试尾部 singleton 维度 |
| TopK | 2048；原始 token index 输出为 int32 `[T, 1, 2048]` |
| Request batch | 1、2、4、8、16、32 |
| MTP query span | 1、2、4、6，包含 ragged batch |
| Context | 1999、2048、2049、2177，以及 16k、64k、128k、140000、143360 |
| Resident capacity | 2048、4096、8192，以及 UB / GM scratch 的边界容量 |

正确性检查包括原 `torch_npu.npu_quant_lightning_indexer` + 独立 `npu_sparse_kv_plan`
的差分比较，以及独立 CPU 状态机对所有七项 planner 状态/输出的逐项比较。
还覆盖重复 token、LRU 顺序、重复调用、请求复用、MTP 回滚、无效 planner 页、
padding、graph capture/replay、非法 shape/dtype 和有确定分数间隔的 TopK 选择。
刻意构造的同分样本按各自产生的合法索引检查状态，避免把未规定的 tie 顺序当作正确性要求。
内联路径另外检查总 query 行数 64/65 的分块切换、同一 AIV 多行复用、
跨多个 TopK trunk 的最终合并，以及混合空序列和全空序列的单次状态更新。
行数 64/65 与容量 2048/7168/7169 交叉覆盖，包括动态 UB 最大的
`T=65、capacity=7168` 组合及相邻 GM fallback。

## 在 950 上构建

在与当前 clean_simt 相同的已配置容器或环境内，pull 修改并重新构建扩展。
先 source 本机 CANN 9.1.0 的 `set_env.sh`，然后在仓库根目录运行：

```bash
set -o pipefail
python -m pip install --no-build-isolation --no-deps -v -e . 2>&1 | tee qli-build.log
```

使用现有构建流程识别真实 950 型号；如环境原本设置了 `SOC_VERSION`，保持实际型号值。
构建日志应列出 `quant_lightning_indexer_sparse_kv`。请保留编译器的资源报告：
该 mixed kernel 的最大动态 UB 预算为 212768 bytes，需确认静态 UB、SIMT stack/spill
和 DataCache 最终满足 CANN 9.1.0 约束。源码静态预算不能替代编译器检查。

最终 TopK 的 merge、score、临时区合并为可复用 slab，planner 复用已完成计算的
临时数据区，索引仍保留在独立 UB 区。scan 额外占用 544 bytes；容量 2048/4096
继续使用 UB scratch，容量 8192 沿用 GM scratch。原有 AIC/MTE ping-pong
缓冲区不参与复用。典型预算如下，需与编译资源报告对照：

| Capacity | `T <= 64` 动态 UB | `T > 64` 动态 UB | Planner scratch |
| --- | --- | --- | --- |
| 2048 | 151072 | 187424 | UB |
| 4096 | 151840 | 188192 | UB |
| 7168（112 KiB 边界） | 176416 | 212768 | UB |
| 8192 | 151072 | 187424 | GM |

## 执行测试

以下命令都在仓库根目录执行，默认设备为 0，可添加 `--device N`。
脚本检查目标设备、原版 QLI 和新扩展是否存在，保存环境信息、日志与 JUnit XML；
缺少依赖、缺少算子、没有选中测试或测试被跳过均会返回非零退出码。

```bash
# 快速真实算子差分，先检查安装与最小 MTP 路径
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite smoke

# 本次内联修改专项：行归属、UB 复用、空序列、长序列和 scratch 边界
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite inline

# 完整正确性矩阵，包含 batch 32 + 140k + ragged MTP，以及 CPU/runtime 单元测试
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite full

# 单独重跑长上下文与 batch 32 压力用例
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite stress

# 正确性检查通过后对比融合和原两算子路径，导出时间与各自 NPU trace
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite profile --batch 32 --seq-len 140000 --mtp 6 --capacity 4096 --iterations 20 --trace
```

结果默认写入 `test-results/qli-<时间>/`，可用 `--output DIR` 指定目录。
`--cann-root PATH` 可指定本机 CANN 根目录；默认读取已 source 的 `ASCEND_HOME_PATH`，
否则使用 `/usr/local/Ascend/cann`。脚本读取版本文件的 `version` 字段并要求为 9.1.0，
不从 torch_npu 版本号推断 CANN 版本。安装信息路径依据
[CANN 9.1.0 安装指南](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/softwareinst/instg/instg_0054.html)。

profile 输出为已预热 planner 状态下的算子级时间，范围仅为 QLI + Plan，
不包含 DDR Transfer 或整模型 attention 延迟。检查 trace 中融合路径是否为一个
`QuantLightningIndexerSparseKv` mixed kernel、基线是否为 QLI 与 SparseKvPlan 两个 kernel。
对比应使用相同设备、输入、capacity 和执行模式。本次 macOS 开发未测得性能结果。
建议分别以 `--capacity 2048`、`4096`、`8192` 重跑 profile；消除索引重读并不直接
证明端到端提速，同一 AIV 提前执行 planner 也会影响 QLI 流水调度。

CPU oracle、内联源码约束和编译定义回归也可以在无 NPU 环境单独执行。
编译定义回归需要本机 C++17 编译器（`clang++`、`g++` 或 `c++`），检查
template key 头文件能否暴露 device tiling 类型、24 个字段的类型/顺序/偏移，
以及生成的 tiling key 宏是否与 planner 配置命名空间冲突。
这些检查使用主机编译器和最小模板宏桩，不执行 CANN 内核，不能代替 CANN 编译或 950 正确性测试：

```bash
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite static
```

完整算子测试通过后，再用已有 clean_simt 的 GLM-5.2 启动配置进行模型级回归，
覆盖启用 MTP、关闭 MTP 及已有 graph 模式。单算子结果不能替代模型端到端精度与吞吐验证。

## 本地验证记录（2026-10-24，macOS）

- `--suite static` 实际运行：12 项通过，包括 4 个独立 CPU planner oracle 测试、
  4 个内联源码约束检查、2 个 tiling 定义检查和 2 个命名空间编译回归。
- 针对 `4fd25e2fb` 的编译反馈，删除未使用的错误 include，补齐 kernel 侧 tiling
  结构体及 template key 引用；迁移 `d944f3ca2` 的命名空间修复，并适配共享 planner
  和融合算子中的新增引用。主机编译回归已复现改名前的冲突，并在修复后通过。
- 本次编译修复涉及的 4 个 Python 文件通过 AST、字节码和 Ruff 检查。
- 19 个新增或修改的 Python 文件：AST、字节码编译及 Ruff lint/format 检查通过。
- Torch schema、8 个 mutable alias、22 个 ACLNN 输入及 14 个属性顺序检查通过。
- vendored 源码保留原 scoring、TopK 分块和排序算法；本次有意修改 vector/kernel
  的 UB 分配、最终行消费点和空序列行归属，修改清单见算子 README。
  共享 planner 使用不同地址空间的行输入指针，LRU 算法保持原语义。
  六个 planner 算法函数与 clean_simt 基线逐函数比对通过，差异仅为行索引指针偏移。
- 实际执行新增 CMake install 规则，vendor、共享 planner 和许可证正确进入安装目录。
  此项将编译函数替换为空操作，仅验证打包，不代表 CANN 编译通过。
- `git diff --check`、构建 shell 语法、测试入口和 profiler 的 `--help` 检查通过；
  无 torch_npu 时测试入口返回失败，CANN 版本检查拒绝不匹配或缺失的版本。
- 独立内核架构、运行时/绑定及测试审查完成；发现的源码安装依赖、ACLNN 状态校验、
  padded query、非 128 block size 回退和多请求页表测试问题已修正。

未执行：CANN 编译、950 单算子正确性/性能、模型端到端回归。
本机缺少 vLLM / torch_npu，完整运行时测试也未执行；相关 import skip 不计为通过。
Python LSP 类型检查器未安装，未将 Python 语法检查等同于类型检查。
本地 clang 缺少 CANN 头文件和 AscendC 专用类型，无法完成内核 C++ 类型检查。
本次运行 `bash format.sh ci` 因未安装 `pre-commit` 而退出；上述 Ruff 检查为单独执行。
