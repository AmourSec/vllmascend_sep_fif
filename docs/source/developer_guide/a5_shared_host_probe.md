# A5 共享 Host KV 池：方案与独立验证

本实验基于 SIMTv3 修复分支，面向 Ascend 950/A5、CANN 9.2 环境。
它提供独立工具，不改变当前每个 TP rank 使用本地池的生产路径。
目标是找出共享池的可行基础，再决定是否接入框架；不能由 API 名称推断整条链路已经可用。

## 1. 官方资料带来的判断

官方 Host 内存指南描述了将普通 `malloc/mmap` 内存注册为 pinned、mapped 的路径，
并指出非锁页内存在 Linux 5.10 及更低版本上的限制。
因此，“共享 mmap + 各进程自己的设备注册”值得优先验证。
这仍是从文档推导出的候选方案：文档并未直接保证同一物理页在多个进程、多张 A5 上重复注册一定成功。
参见 [Host 内存管理](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/programug/acldevg/runtime_doc_dev_0006.html)。

`HostGetDevicePointer` 的结果用于设备核函数访问，不应拿来代替 CPU HVA 做普通内存操作。
工具始终保存两个地址：共享 CPU 数据和 URMA 目的注册使用 HVA，核函数参数使用该进程查询到的 DVA。
不会要求 HVA/DVA 数值相等，也不会把 rank 0 的 DVA 广播给其他设备。
参见 [设备地址约束](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclcppdevg_03_0100.html)。

CANN 9.2 的 V2 物理内存导出/导入文档列出了 950PR/950DT 支持，并要求进程白名单、资源存活及跨卡访问配置。
这是第二条候选路线；接口存在不等于任意 Host 内存属性都受当前驱动支持。
参见 [导出 V2](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclcppdevg_03_2006.html)、
[导入 V2](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclcppdevg_03_2008.html)、
[PID 白名单](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclcppdevg_03_2007.html)。

物理内存申请接口能指定 Host NUMA 属性。
VMM 探针明确请求 `HOST_NUMA`，再查询实际属性；若实际不属于 Host 内存则失败，不会偷偷改成 HBM IPC。
但本探针不把映射的 VMM 地址当作 CPU HVA，没有提供它与 URMA Host 目的注册之间的生产接入。
参见 [物理内存申请](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclcppdevg_03_0112.html)。

不同版本及网页缓存中，部分 950DT 支持表存在差异。记录实际 CANN、驱动、SoC 和接口返回码，
不要仅凭头文件里有符号，或旧版网页里的一个支持标记宣布可用。

## 2. 候选路线和取舍

| 路线 | 预计收益 | 待验证的关键问题 | 工具入口 |
| --- | --- | --- | --- |
| 重新验证现有 MemFabric SHARED 路径 | 复用已有导出/导入和分片拉取，运行代码改动可能最少 | A5 导入映射、旧统一地址约定、各 rank 的 DVA 查询 | `mf-shared` |
| 一份 POSIX 共享内存，每个 rank 单独注册映射 | 一份物理 Main KV；各卡仍直接读取 Host，不增加 payload 中转 | 多进程重复注册、设备访问、改写可见性、URMA 对共享区域的注册 | `posix-mapped` / `posix-pinned-mapped` |
| Host VMM 物理句柄跨进程导入 | 明确共享同一份物理 allocation，映射由驱动管理 | HOST_NUMA/P2P 属性、白名单、跨卡访问，以及后续 CPU HVA 的获取 | `vmm-host` |
| 共享主池 + 每 rank 小型本地 staging | 主历史 KV 只保留一份，可减少大池复制 | 多一跳 payload 拷贝、staging 大小、图模式和 CPU 调度成本 | 本轮只评估，未实现 |

优先次序：先试原 MF 共享能力和 POSIX 注册，两者均不满足时再看 VMM。
staging 是兼容性备选，不预设它比当前直接 Host→resident 更快。
无论选择哪条共享路线，PD Main KV 可以继续各 rank 分工填充，最终只填满一份共享池。

## 3. 工具做什么

目录：`tools/a5_shared_host_probe/`。核心依赖只有 Python 标准库和服务器 CANN；
`mf-shared`/URMA 才需要已安装的 MemFabric。无需安装 vLLM Ascend、Torch 扩展或整网模型。

Python 父进程不加载 CANN，使用 spawn 启动独立子进程。每个子进程选择自己的逻辑设备并持有自己的映射。
POSIX 实验只共享文件及偏移，不跨进程传递可解引用的指针。日志记录 HVA、DVA、backing inode、设备号。

每次实验依次检查：

1. 所有 rank 读取初始完整模式，逐字节核对。
2. 各 rank 写不同片段，完成设备 stream 同步后，每个 rank 再读取完整模式。
3. rank 0 改写最后一页，其他 rank 必须读到新内容。这是改写可见性测试，不是完整 MTP 验收。
4. POSIX/MF 模式再从 CPU 改写，设备必须读到新内容。
5. 可选增加一张 P 卡，通过真实 DEVICE_URMA 将 HBM 数据分片拉入共享 HVA，再由各 D 卡读取核对。
6. 检查清理返回码；D 注册/连接在 P 源 allocation 释放前清理。

`pinned` 是各 rank 私有池正对照，因此每个池单独初始化、写完整模式。
`cpu` 只检查真实多进程共享和控制协议，结果标为 `CPU_ONLY`。
`runtime` 路径仅验证 Runtime 拷贝视图，结果为 `RUNTIME_COPY_ONLY`，不能充当 AIV 成功证据。

AIV 路径包括普通 SIMT、`asc_ldcg/asc_stcg` 的缓存策略对照和通过 UB 中转的 MTE copy。
普通 SIMT 与生产 Transfer 的直接 GM 访问更接近；另两条用来区分访存路径和缓存可见性问题。
不能仅因 CG 版本通过，就把缓存策略作为生产修复直接提交。

POSIX 共享区前后有 Host canary；AIV copy 的 HBM 缓冲区前后也有 canary。
所有比较覆盖完整有效 payload，模式随位置和轮次变化，不把未使用容量里的 NaN 比例当作正确性结论。
这些 canary 不代替设备 memcheck，也不证明没有任何越界。

## 4. 构建与首轮测试

从仓库根目录执行。设备号按服务器空闲的逻辑卡替换。
CANN 建议先按服务器安装路径加载 `set_env.sh`。

```bash
bash tools/a5_shared_host_probe/build.sh --jobs=2
# 非标准安装布局可显式指定包含 include/acl、asc/include、lib64 的目录：
# bash tools/a5_shared_host_probe/build.sh --cann-root=/path/to/aarch64-linux

python3 tools/a5_shared_host_probe/probe.py --backend cpu --devices 0,1 \
  --output /dev/shm/a5_cpu.jsonl

python3 tools/a5_shared_host_probe/probe.py --backend pinned --devices 0 --path simt \
  --output /dev/shm/a5_pinned.jsonl

python3 tools/a5_shared_host_probe/probe.py --backend posix-pinned-mapped --devices 0,1 --path simt \
  --output /dev/shm/a5_shared_simt.jsonl
```

本工具默认仅测试 256 KiB。按 `max(4096, 系统页大小)` 对齐；`--bytes` 必须为该对齐量乘 rank 数的倍数，最多 1 GiB。
这也覆盖鲲鹏系统使用较大基础页的情况；实际对齐量会写入结果元数据。
先 2 卡，成功后再用 `--devices 0,1,2,3` 和更大尺寸；不用一开始申请生产规模的数百 GiB。
发生设备异常时停止当前路径、收集日志并按现场方式恢复设备，不要盲目连续跑所有候选。
工具仅管理自己启动的子进程，不清理其他服务。

同一块共享内存可以比较两种注册标志：

```bash
python3 tools/a5_shared_host_probe/probe.py --backend posix-mapped --devices 0,1 --path simt \
  --output /dev/shm/a5_shared_mapped_only.jsonl
python3 tools/a5_shared_host_probe/probe.py --backend posix-pinned-mapped --devices 0,1 --path simt-cg \
  --output /dev/shm/a5_shared_cg.jsonl
python3 tools/a5_shared_host_probe/probe.py --backend posix-pinned-mapped --devices 0,1 --path mte \
  --output /dev/shm/a5_shared_mte.jsonl
```

`pinned` 使用 `aclrtMallocHost + MAPPED`，避免把已锁页分配和普通 mmap 内存的注册标志混用。
上述标志对照不是允许在生产中忽略任何注册错误。

## 5. 复查原 MemFabric SHARED

这个模式复用已安装 MF 的真实 `Scene.SHARED`，只有 rank 0 分配 1 GiB。
`offload.malloc()` 与生产 `offload.empty()` 的地址来源相同，省去构造 Torch Tensor。
先按旧框架的 HVA/DVA 同值约定测试，再单独测试各 rank 的 `get_dva()`。

```bash
python3 tools/a5_shared_host_probe/probe.py --backend mf-shared --devices 0,1 --path simt \
  --shared-store-url tcp://127.0.0.1:31991 --shared-address identity \
  --output /dev/shm/a5_mf_shared_identity.jsonl
python3 tools/a5_shared_host_probe/probe.py --backend mf-shared --devices 0,1 --path simt \
  --shared-store-url tcp://127.0.0.1:31992 --shared-address get-dva \
  --output /dev/shm/a5_mf_shared_lookup.jsonl
```

这两个模式有意对照旧地址约定；不是把 identity 当作新方案的正确前提。
共享入口当前的 `get_dva()` 有本 rank 池范围检查，非拥有者查询 rank 0 地址可能失败。
若 identity 可用、lookup 失败，应修共享地址转换契约，不能直接认定底层不支持共享。
若在 `initialize` 阶段失败，应先定位底层 reserve/export/import/map 的首个错误，不归咎于 SIMT。

## 6. VMM Host 物理共享候选

```bash
python3 tools/a5_shared_host_probe/probe.py --backend vmm-host --devices 0,1 --path simt \
  --numa-node 0 --output /dev/shm/a5_vmm_host.jsonl
# 确认普通页路径后，才按机器页配置试 --huge。
```

rank 0 请求 Host NUMA 物理内存并按查询到的粒度对齐，导出 V2 默认句柄，
使用 `aclrtDeviceGetBareTgid` 得到的实际进程 ID 设置白名单；各进程保留自己的映射和句柄。
跨卡时查询并开启相应 peer access，源物理内存在其他进程结束之前存活。
不会在同一进程同时 export/import，也不会关闭白名单来绕过失败。

这条路径没有直接解引用 VMM 指针，只通过 Runtime/AIV 验证 Host 物理内存的共享。
成功不代表已经解决了 CPU 可访问 HVA 或 URMA 注册，因此 CLI 禁止将它直接与 URMA 测试组合。
NUMA ID 必须按实际服务器配置选取；P2P 属性不支持、页资源不足或跨卡不支持都会报告失败，不回退 HBM。

## 7. 真实 URMA 分片拉取

需要额外一张 P 卡。下面用 P=2、D=0/1；IP、两个 store 端口按现场替换，确保是未占用的测试端口。
source 设备不能与任何 D 设备相同，避免设备同卡跨进程通信限制干扰判断。

```bash
python3 tools/a5_shared_host_probe/probe.py --backend posix-pinned-mapped --devices 0,1 --path simt \
  --urma-source-device 2 --bind-ip 192.0.2.10 --store-url tcp://192.0.2.10:31993 \
  --timeout 90 --output /dev/shm/a5_shared_urma.jsonl
```

替换示例 IP 后再运行，`192.0.2.10` 是文档占位地址。
每个 D rank 向自身引擎注册自己的 HVA，分别拉取 P HBM 的一段，随后全部读取完整模式。
`urma_pull_then_aiv_read` 比较通过才证明这个小实验的传输和设备读链路闭合。
同样可用 `pinned` 做复制池对照，或用 `mf-shared --shared-store-url ...` 检查原池的通信兼容性。
此处是同机多卡数据通路测试，不能代替双节点网络、真实 KV layout、生产规模注册和整网 PD 验收。

## 8. 如何据结果决定改造

| 结果 | 下一步 |
| --- | --- |
| 本地 pinned 正对照失败 | 先定位环境、API 或探针构建，不评价共享方案 |
| CPU 共享通过，HostRegister 失败 | 查注册标志、对齐、内核/驱动、重复注册支持；保留当前本地池 |
| 注册成功，AIV 初次读取失败 | 查该设备 DVA 与访问路径；注册返回成功不能充当访问证据 |
| 初次读取通过，CPU/peer 改写后失败 | 查缓存和跨设备可见性；比较 SIMT、CG、MTE，不能直接接整网 |
| AIV 完整通过，URMA 注册/拉取失败 | 共享算子访问与通信注册未闭合，继续定位 MF/HCOMM 的地址和区域注册 |
| MF identity 通过，但 get_dva 失败 | 有希望复用现有共享池；优先修正共享地址查询和上层本地地址表 |
| VMM 共享通过 | 继续研究 CPU Host 映射和 URMA 注册；暂不宣称生产 Host 池已可替换 |
| 多卡 AIV 与 URMA 均通过 | 才改 MF 的内存所有权/映射接口、vLLM 地址表及写回规则，并保留原分片拉取 |

整网接入时还要把 A5、共享池选择和 SIMT 后端选择解耦；当前 `uses_simt_lru` 与 rank-local 标志绑定。
共享 Main KV 不意味着共享各卡的 resident HBM、LRU 状态或 Indexer 分片。
需要固定输入比较 token/KV 内容，并覆盖 TP1/TP4、MTP 后缀改写、图 replay、请求并发及释放。
小块正确后才逐步测大池注册耗时、NUMA 带宽和端到端收益；本工具的总 elapsed 包含 IPC 与 CPU 比较，不是性能基准。

## 9. 本机验证范围

本机具备 CANN 9.2 beta2 x86 开发包，没有 A5 设备。
已检查 Host C++ 对真实 SDK 头文件的语法，CPU 多进程共享测试可本地运行。
毕昇混合编译遇到本机 glibc 头文件 `_Float128` 与设备编译目标不兼容，因此 AIV 产物和所有设备路径仍需在服务器编译、运行。
这不是已经验证的 A5 共享池实现，不修改现有部署的软件包或生产配置。
