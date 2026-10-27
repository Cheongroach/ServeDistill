#!/usr/bin/env bash
set -euo pipefail
repo="$(git rev-parse --show-toplevel)"
destination="${1:-${repo}-doc}"
git fetch origin doc
if [[ -d "${destination}/.git" || -f "${destination}/.git" ]]; then
  test "$(git -C "${destination}" branch --show-current)" = "doc"
else
  if git show-ref --verify --quiet refs/heads/doc; then
    git worktree add "${destination}" doc
  else
    git worktree add -b doc "${destination}" origin/doc
  fi
fi
python3 - "${repo}" "${destination}" <<'PY'
import json
import os
from pathlib import Path
import sys
repo, doc = map(lambda value: Path(value).resolve(), sys.argv[1:])
target = repo / "ServeDistill.code-workspace"
value = {
    "folders": [{"name": "main · runnable code", "path": "."},
                {"name": "doc · research notes", "path": os.path.relpath(doc, repo)}],
    "settings": {"files.exclude": {"**/__pycache__": True}},
}
if not target.exists() or json.loads(target.read_text()) != value:
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
print(target)
PY
