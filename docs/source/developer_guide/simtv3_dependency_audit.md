# SIMTv3 dependency audit: A5 / CANN 9.2 candidate environment

This audit applies to the SIMTv3 fork based on
`10de9206981b23350ed29bb9fc2df8ea23ac7323`, with vLLM v0.28.0 at
`2cf0a6915ce544dc493a0990f2ea38d81601128a`. It is not a new validated release
matrix for all Ascend hardware.

## Observed environment and evidence

The container image is
`vllm-ascend:dev-26.2.0.day20261011-A5-py311-openEuler24.03-lts-aarch64`.
It provides Python 3.11.10, CANN 9.2.0, Torch 2.10.0, TorchNPU
2.10.0.post7, and Triton Ascend 3.6.0+dev20261007182438.

The server-side audit supplied with this patch reports the following checks.
These results were not rerun on the local PC applying this patch:

- Full vLLM Ascend custom-op/C++ compilation, with an explicit
  `SOC_VERSION=ascend950dt_9582`, succeeded.
- vLLM 0.28.0+empty and vLLM Ascend were installed from the pinned sources.
- The MemFabric clean fork at `4b5bfe2490a7e0b9bc7214743411cb173c288945`
  was built and installed as version 1.2.1. Its AICPU payload and AccOffload
  library were built and installed for A5 / CANN 9.2.
- Source imports, configuration imports, MemFabric interface presence, and a
  FastAPI 0.136.3 / Starlette 1.8.0 CPU TestClient check passed.

The checks did not initialize an NPU, execute a model, or exercise Triton or
custom-op device kernels. Successful compilation and imports do not establish
end-to-end runtime compatibility.

## Disposition of the five reported conflicts

| Dependency | Old Ascend requirement | Installed version | Disposition |
| --- | --- | --- | --- |
| FastAPI | `<0.124.0` | `0.136.3` | Align build and runtime metadata to vLLM 0.28: `>=0.133.0,<0.137.0`; add `starlette>=1.0.1` in both places |
| MemFabric Hybrid | `==1.2.0` | `1.2.1`, custom clean fork | Require `==1.2.1`; retain the exact fork revision in installation records |
| Memcache Hybrid | `==1.2.0`, mandatory | `1.3.0`, inherited from the image | Make the separate MemcacheConnector backend an optional `memcache` extra, retaining its existing `==1.2.0` contract |
| TorchNPU | `==2.10.0.post4` | `2.10.0.post7` | Keep the existing declaration until device runtime validation is available; do not downgrade the running image stack during this audit |
| Triton Ascend | `==3.2.2` | `3.6.0+dev20261007182438` | Keep the existing declaration until Triton device-kernel validation is available; do not broadly relax the requirement to suppress a warning |

vLLM 0.28's FastAPI lower bound and the old Ascend upper bound have no
intersection. A constraints file alone cannot repair that contradiction;
the package declarations must change and the installed metadata must be
regenerated. `--no-deps` preserves installed packages, but does not make
incompatible metadata compatible.

The sparse KV path uses MemFabric APIs including `offload.get_dva`,
`Scene.LOCAL`, and the `device_urma` protocol. A package named
`memfabric-hybrid` with version 1.2.1 is not sufficient evidence that it
contains this fork. Build and install the wheel from the exact revision above
before installing Ascend. Do not substitute an unrelated PyPI wheel or
automatically replace it with the image's 1.3.0 package.

MemcacheConnector imports `memcache_hybrid` when its backend is initialized;
it is not the backend used by this SIMT PD configuration. Plain installation
no longer installs or changes Memcache. Selecting `.[memcache]` explicitly
adds the existing 1.2.0 requirement; this audit does not validate Memcache 1.3.0.
The optional backend's own transitive dependencies still need resolution and
testing if that backend is selected.

## Applying the metadata changes to an existing server installation

Review and apply the patch to the exact base commit before rebuilding. Preserve
local changes and existing compiled artifacts. Use the original virtual
environment, source paths, and explicit compilation target. For example, after
the reviewed changes are present in the checkout:

```bash
(
  set -e
  source /workspace/env_simtv3.sh
  cd /workspace/src/vllmascend_sep_fif

  export SOC_VERSION=ascend950dt_9582
  export COMPILE_CUSTOM_KERNELS=1
  export VLLM_BATCH_INVARIANT=0
  export MAX_JOBS=64
  export OPS_CPU_NUMBER=64
  export CMAKE_BUILD_PARALLEL_LEVEL=64
  export OMP_NUM_THREADS=1
  export TMPDIR=/workspace/tmp/vllmascend_build
  export TMP="$TMPDIR"
  export TEMP="$TMPDIR"
  mkdir -p "$TMPDIR" /workspace/logs

  LOG="/workspace/logs/reinstall_dependency_metadata_$(date +%Y%m%d_%H%M%S).log"
  echo "LOG=$LOG"
  if python -m pip install -v -e . --no-build-isolation --no-deps > "$LOG" 2>&1; then
    tail -20 "$LOG"
  else
    tail -80 "$LOG"
    exit 1
  fi
)
```

This uses the existing offline compilation path and may perform an incremental
native build. There is no model start or device test in the command. It does
not install replacement TorchNPU, Triton, or MemFabric packages.

Collect the remaining declaration conflicts without importing any device
backend:

```bash
source /workspace/env_simtv3.sh
python -m pip check > /workspace/logs/pip_check_after_dependency_audit.txt 2>&1
PIP_CHECK_RC=$?
cat /workspace/logs/pip_check_after_dependency_audit.txt
echo "pip check exit code: $PIP_CHECK_RC"

python - <<'PY'
from importlib import metadata
from packaging.requirements import Requirement

for name in ("vllm", "vllm-ascend", "memfabric-hybrid", "memcache-hybrid",
             "fastapi", "starlette", "torch", "torch-npu", "triton-ascend"):
    try:
        print(name, "->", metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name, "-> not installed")

print("\nDeclared Ascend dependency differences (default extra):")
for text in metadata.requires("vllm-ascend") or []:
    requirement = Requirement(text)
    if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
        continue
    try:
        version = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        print("MISSING:", text)
        continue
    if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
        print("MISMATCH:", text, "| installed:", version)
PY
```

Do not assume that the total `pip check` output contains only the five original
issues. A `--system-site-packages` environment also exposes dependencies of
other image packages. In particular, inspect Memcache's own requirements if it
remains installed. Moving an unrelated backend to an extra does not repair
that backend's internal dependency graph.

## Remaining release gates

The patch resolves three specific declarations. TorchNPU and Triton conflicts
remain intentionally open, so this environment is not yet a conflict-free,
validated delivery environment.

When NPUs become available, validate the exact image stack with custom-op
loading, kernels used by the selected model, ordinary PD inference, SIMT
offload, graph replay, and MTP. Include long contexts that exercise sparse
selection and LRU misses, accepted speculative tokens, and multi-rank PD
transfers. Reassess device version declarations from those results, and keep
environment-specific exact pins rather than inferring support for all newer
versions. If that validation fails, choose a complete compatible image stack
instead of independently downgrading one binary component.

Dependency consistency, successful offline build/import, and model runtime
correctness are separate acceptance criteria. All three must be recorded
before delivery.
