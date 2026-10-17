from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parents[2] / "check_kernel_metadata.py"


def _metadata(root: Path, dtype: str, keys: list[int | str], *, relocatable: bool = False) -> Path:
    folder = root / "ascend950" / "lightning_indexer_quant"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"LightningIndexerQuant_{dtype}" + ("_relocatable" if relocatable else "")
    document = {
        "binFileName": name,
        "binFileSuffix": ".o",
        "coreType": "MIX",
        "taskRation": "tilingKey",
        "supportInfo": {
            "inputs": [{"name": "query", "dtype": dtype}, {"name": "key", "dtype": dtype}],
            "tilingKey": ["50528770", "50537508"],
        },
        "kernelList": [{"kernelName": f"{name}_{key}", "tilingKey": key} for key in keys],
    }
    path = folder / f"{name}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    path.with_suffix(".o").write_bytes(b"metadata-test-object")
    return path


def _check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(CHECKER), str(root)], capture_output=True, text=True)


def test_metadata_accepts_compiled_keys_and_reports_binary_paths(tmp_path: Path) -> None:
    _metadata(tmp_path, "int8", [50528770])
    fp8 = _metadata(tmp_path, "float8_e4m3fn", ["50537508"])
    _metadata(tmp_path, "float8_e4m3fn", [50537508], relocatable=True)
    result = _check(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(fp8) in result.stdout
    assert str(fp8.with_suffix(".o")) in result.stdout
    assert "50537508" in result.stdout


def test_metadata_rejects_fp8_key_missing_from_its_own_kernel_list(tmp_path: Path) -> None:
    _metadata(tmp_path, "int8", [50528770, 50537508])
    fp8 = _metadata(tmp_path, "float8_e4m3fn", [50528770])
    _metadata(tmp_path, "float8_e4m3fn", [50537508], relocatable=True)
    result = _check(tmp_path)
    assert result.returncode == 1
    assert str(fp8) in result.stderr
    assert "missing compiled tilingKey=50537508" in result.stderr


def test_metadata_rejects_missing_normal_fp8_binary(tmp_path: Path) -> None:
    _metadata(tmp_path, "int8", [50528770])
    _metadata(tmp_path, "float8_e4m3fn", [50537508], relocatable=True)
    result = _check(tmp_path)
    assert result.returncode == 1
    assert "missing normal metadata for float8_e4m3fn" in result.stderr


def test_metadata_rejects_missing_object_file(tmp_path: Path) -> None:
    _metadata(tmp_path, "int8", [50528770])
    fp8 = _metadata(tmp_path, "float8_e4m3fn", [50537508])
    fp8.with_suffix(".o").unlink()
    result = _check(tmp_path)
    assert result.returncode == 1
    assert str(fp8.with_suffix(".o")) in result.stderr


@pytest.mark.parametrize("content", ["{", '{"binFileName": "LightningIndexerQuant_broken"}'])
def test_metadata_rejects_unreadable_kernel_schema(tmp_path: Path, content: str) -> None:
    _metadata(tmp_path, "int8", [50528770])
    fp8 = _metadata(tmp_path, "float8_e4m3fn", [50537508])
    fp8.write_text(content, encoding="utf-8")
    result = _check(tmp_path)
    assert result.returncode == 1
    assert str(fp8) in result.stderr


def test_metadata_does_not_use_other_soc_or_operator_to_supply_fp8_key(tmp_path: Path) -> None:
    _metadata(tmp_path, "int8", [50528770])
    fp8 = _metadata(tmp_path, "float8_e4m3fn", [50537508])
    other = tmp_path / "ascend910b" / "lightning_indexer_quant"
    other.mkdir(parents=True)
    fp8.rename(other / fp8.name)
    result = _check(tmp_path)
    assert result.returncode == 1
    assert "missing normal metadata for float8_e4m3fn" in result.stderr


def test_metadata_rejects_empty_root(tmp_path: Path) -> None:
    result = _check(tmp_path)
    assert result.returncode == 1
    assert "missing normal metadata" in result.stderr


def test_metadata_accepts_key_encoded_in_kernel_name(tmp_path: Path) -> None:
    _metadata(tmp_path, "int8", [50528770])
    fp8 = _metadata(tmp_path, "float8_e4m3fn", [50537508])
    document = json.loads(fp8.read_text(encoding="utf-8"))
    del document["kernelList"][0]["tilingKey"]
    fp8.write_text(json.dumps(document), encoding="utf-8")
    result = _check(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_metadata_checks_both_roots_when_first_one_fails(tmp_path: Path) -> None:
    built, installed = tmp_path / "build", tmp_path / "installed"
    _metadata(built, "int8", [50528770])
    _metadata(installed, "int8", [50528770])
    fp8 = _metadata(installed, "float8_e4m3fn", [50537508])
    result = subprocess.run([sys.executable, str(CHECKER), str(built), str(installed)], capture_output=True, text=True)
    assert result.returncode == 1
    assert str(fp8) in result.stdout
    assert "missing normal metadata for float8_e4m3fn" in result.stderr
