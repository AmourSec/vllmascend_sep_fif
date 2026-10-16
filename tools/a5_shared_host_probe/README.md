# A5 共享 Host KV 池独立探针

用于选择后续内存池改造方案，不修改生产 manager、SIMT Plan/Transfer 或 MemFabric。

从仓库根目录执行，不需要安装当前 vLLM Ascend 分支：

```bash
# 不需要 NPU/CANN 的 CPU 共享及控制协议检查。
python3 tools/a5_shared_host_probe/probe.py --backend cpu --devices 0,1

# 在已加载 CANN 9.2 环境的服务器上构建小型独立库。
bash tools/a5_shared_host_probe/build.sh --jobs=2

# 先做单卡本地 pinned/MAPPED 正对照，再做多进程共享。
python3 tools/a5_shared_host_probe/probe.py --backend pinned --devices 0 --path simt
python3 tools/a5_shared_host_probe/probe.py --backend posix-pinned-mapped --devices 0,1 --path simt
```

`--devices` 使用当前运行环境的逻辑卡号。默认只有 256 KiB 数据，POSIX 模式额外有两页 Host canary；
`mf-shared` 则需要一套 1 GiB 的 MemFabric 池。

完整方案、测试矩阵、URMA 命令及结果解释见
[A5 共享 Host 池验证方案](../../docs/source/developer_guide/a5_shared_host_probe.md)。

任何 `FAIL` 都不代表已定位唯一根因；`CPU_ONLY` 或 `RUNTIME_COPY_ONLY` 的成功不代表 AIV 访问成功。
