#!/usr/bin/env python3
"""Check compiled QLI decode keys before installing or testing an Ascend 950 OPP.

Usage: python3 check_kernel_metadata.py /path/to/binary/ascend950/bin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Final, NamedTuple, TypedDict

EXPECTED_KEYS: Final = {"int8": 50528770, "float8_e4m3fn": 50537508}


class InputInfo(TypedDict):
    dtype: str


class SupportInfo(TypedDict):
    inputs: list[InputInfo]


class KernelEntry(TypedDict, total=False):
    tilingKey: int | str
    kernelName: str


class KernelDocument(TypedDict):
    binFileName: str
    binFileSuffix: str
    supportInfo: SupportInfo
    kernelList: list[KernelEntry]


class KernelMetadata(NamedTuple):
    dtype: str
    keys: frozenset[int]
    binary: Path


class MetadataError(ValueError):
    pass


def read_metadata(path: Path) -> KernelMetadata:
    document: KernelDocument = json.loads(path.read_text(encoding="utf-8"))
    inputs = document["supportInfo"]["inputs"]
    dtype = inputs[0]["dtype"]
    if dtype not in EXPECTED_KEYS or inputs[1]["dtype"] != dtype:
        raise MetadataError(f"unexpected Q/K dtypes: {dtype}, {inputs[1]['dtype']}")
    keys = set()
    for entry in document["kernelList"]:
        # CANN can encode the key in kernelName instead of an explicit field.
        key = entry.get("tilingKey")
        if key is None:
            key = entry["kernelName"].rsplit("_", 1)[-1]
        keys.add(int(key))
    binary_name = document["binFileName"] + document["binFileSuffix"]
    if Path(binary_name).name != binary_name:
        raise MetadataError(f"expected a local binary filename, got {binary_name}")
    return KernelMetadata(dtype, frozenset(keys), path.parent / binary_name)


def check_root(root: Path) -> bool:
    root = root.resolve()
    print(f"[qli-kernel-check] root={root}")
    normal_dtypes: set[str] = set()
    errors: list[str] = []
    for path in sorted(root.rglob("LightningIndexerQuant_*.json")):
        if path.parent.name != "lightning_indexer_quant":
            continue
        if not any(part.startswith("ascend950") for part in path.parts):
            continue
        try:
            metadata = read_metadata(path)
            print(f"[qli-kernel-check] json={path} dtype={metadata.dtype} keys={sorted(metadata.keys)}")
            print(f"[qli-kernel-check] binary={metadata.binary}")
            if not path.stem.endswith("_relocatable"):
                normal_dtypes.add(metadata.dtype)
            expected = EXPECTED_KEYS[metadata.dtype]
            if expected not in metadata.keys:
                errors.append(f"{path}: missing compiled tilingKey={expected}; observed={sorted(metadata.keys)}")
            digest = hashlib.sha256(metadata.binary.read_bytes()).hexdigest()
            print(f"[qli-kernel-check] binary_sha256={digest}")
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            errors.append(f"{path}: {error}")
    for dtype in EXPECTED_KEYS:
        if dtype not in normal_dtypes:
            errors.append(f"{root}: missing normal metadata for {dtype}")
    for error in errors:
        print(f"[qli-kernel-check] ERROR: {error}", file=sys.stderr)
    return not errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Build bin directory or installed OPP root")
    arguments = parser.parse_args()
    results = [check_root(root) for root in arguments.roots]
    if all(results):
        print("[qli-kernel-check] PASS: metadata contains both decode keys; runtime selection is not checked")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
