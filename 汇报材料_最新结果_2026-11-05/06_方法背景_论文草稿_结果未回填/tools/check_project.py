"""Check local LaTeX dependencies, citations, labels, and delimiter balance.

This is a source check, not a substitute for compiling and inspecting a PDF.
No third-party dependencies or experiment imports are required.
"""
from __future__ import annotations

import re
from pathlib import Path


def uncomment(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sources = sorted(p for p in root.rglob("*.tex") if "vendor" not in p.parts)
    errors: list[str] = []
    labels: dict[str, Path] = {}
    refs: set[str] = set()
    citations: set[str] = set()
    for source in sources:
        rel = source.relative_to(root)
        text = uncomment(source.read_text(encoding="utf-8-sig"))
        for name in re.findall(r"\\input\{([^}]+)\}", text):
            target = root / (name if Path(name).suffix else name + ".tex")
            if not target.is_file():
                errors.append(f"{rel}: missing input {name}")
        for label in re.findall(r"\\label\{([^}]+)\}", text):
            if label in labels:
                errors.append(f"{rel}: duplicate label {label} (also {labels[label]})")
            labels[label] = rel
        refs.update(re.findall(r"\\(?:eqref|ref|autoref)\{([^}]+)\}", text))
        for group in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", text):
            citations.update(key.strip() for key in group.split(","))
        stack: list[str] = []
        for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", text):
            if kind == "begin":
                stack.append(name)
            elif not stack or stack.pop() != name:
                errors.append(f"{rel}: unmatched environment end {name}")
        if stack:
            errors.append(f"{rel}: unclosed environments {stack}")
        # Ignore escaped literal braces and escaped backslashes.
        clean = re.sub(r"\\(?:[{}]|\\)", "", text)
        depth = 0
        for char in clean:
            depth += (char == "{") - (char == "}")
            if depth < 0:
                errors.append(f"{rel}: closing brace before opening brace")
                break
        if depth:
            errors.append(f"{rel}: unbalanced braces ({depth:+d})")
    for ref in sorted(refs - labels.keys()):
        errors.append(f"undefined label {ref}")
    bib = root / "references.bib"
    if bib.is_file():
        keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib.read_text(encoding="utf-8-sig")))
        for key in sorted(citations - keys):
            errors.append(f"undefined citation {key}")
    else:
        errors.append("missing references.bib")
    if errors:
        print("Source check failed:\n" + "\n".join("- " + error for error in errors))
        return 1
    print(f"Source check passed: {len(sources)} TeX files, {len(labels)} labels, "
          f"{len(citations)} citation keys; inputs and delimiters resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
