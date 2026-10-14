"""Static repository checks that do not require TensorFlow or the dataset."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REQUIRED_FILES = (
    ".gitignore",
    "README.md",
    "requirements.txt",
    "requirements-dev.txt",
    "stream_segmentation.py",
    "autonomous-driving-scene-segmentation.ipynb",
)


def validate_repo(repo: Path) -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        if not (repo / relative).is_file():
            errors.append(f"missing required file: {relative}")

    script = repo / "stream_segmentation.py"
    if script.is_file():
        try:
            ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        except SyntaxError as exc:
            errors.append(f"stream_segmentation.py syntax error: {exc}")

    notebook_path = repo / "autonomous-driving-scene-segmentation.ipynb"
    if not notebook_path.is_file():
        return errors
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"notebook is not valid JSON: {exc}")
        return errors

    if notebook.get("nbformat", 0) < 4:
        errors.append("notebook must use nbformat 4 or newer")
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            ast.parse(source, filename=f"notebook cell {index}")
        except SyntaxError as exc:
            errors.append(f"notebook cell {index} syntax error: {exc}")
        if cell.get("execution_count") is not None:
            errors.append(f"notebook cell {index} has stale execution_count")
        if cell.get("outputs"):
            errors.append(f"notebook cell {index} has stale outputs")
        for forbidden in ("/kaggle/", "/home/", "C:\\\\"):
            if forbidden in source:
                errors.append(f"notebook cell {index} contains local-only path {forbidden!r}")
        if "os.remove" in source or "shutil.rmtree" in source:
            errors.append(f"notebook cell {index} deletes dataset/workspace files")

    return errors


def main() -> int:
    repo = Path(__file__).resolve().parent
    errors = validate_repo(repo)
    if errors:
        print("Project validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("Project validation passed (static checks only; no dataset/model execution performed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
