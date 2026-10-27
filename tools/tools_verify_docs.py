"""Read-only final check: every image referenced by the two reports exists, and
no stale RECR-D/HFWPF figures from before the fix are left behind."""
import re
from pathlib import Path

DOCS = [Path("results/REPORT.md"),
        Path("results/导师汇报_实验结果汇总.md")]

for doc in DOCS:
    text = doc.read_text(encoding="utf-8")
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    missing = [r for r in refs if not (doc.parent / r).exists()]
    print(f"{doc.name}: {len(refs)} image refs, {len(missing)} missing")
    for m in missing:
        print(f"    MISSING {m}")

# Any figure in figures_report that predates the last regeneration?
import datetime
figs = sorted(Path("results/figures_report").glob("*.png"))
newest = max(f.stat().st_mtime for f in figs)
print(f"\nfigures_report: {len(figs)} files")
for f in figs:
    age_h = (newest - f.stat().st_mtime) / 3600
    flag = "  <-- STALE (>2h older than newest)" if age_h > 2 else ""
    ts = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M")
    print(f"  {f.name:<44} {ts}{flag}")
