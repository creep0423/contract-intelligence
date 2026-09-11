"""在有无 Git 的环境中生成可复现的应用版本标识。"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def resolve_app_version(project_root: Path) -> tuple[str, str]:
    if (project_root / ".git").exists():
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=False
        )
        sha = result.stdout.strip()
        if result.returncode == 0 and sha:
            return sha, "git"
    digest = hashlib.sha256()
    roots = [project_root / "app.py", project_root / "qa_core" / "contracts", project_root / "qa_core" / "retrieval", project_root / "qa_core" / "api"]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.py")))
    for path in sorted(files):
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return f"source-sha256:{digest.hexdigest()[:16]}", "source_fingerprint"
