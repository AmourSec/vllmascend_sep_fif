from __future__ import annotations

import subprocess
from pathlib import Path


REPOSITORY = Path(__file__).parents[3]


def _shell_scripts() -> list[Path]:
    output = subprocess.run(
        ("git", "ls-files", "-z", "--", "csrc"),
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout
    return sorted(
        REPOSITORY / path
        for path in (Path(item) for item in output.decode().split("\0") if item)
        if path.suffix == ".sh"
    )


def test_csrc_shell_scripts_use_lf_and_parse_with_bash() -> None:
    scripts = _shell_scripts()
    assert scripts

    for script in scripts:
        relative = script.relative_to(REPOSITORY)
        attribute = subprocess.run(
            ("git", "check-attr", "eol", "--", str(relative)),
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert attribute.endswith("eol: lf"), f"{relative}: {attribute}"
        contents = script.read_bytes()
        assert b"\r" not in contents, f"CR byte found in {script}"
        result = subprocess.run(
            ("bash", "-n", str(script)),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"
