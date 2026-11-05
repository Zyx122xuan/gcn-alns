# IEEE LaTeX draft for Overleaf

This is an English manuscript draft for the GCN-ALNS project. It uses the
standard `IEEEtran` journal class in two-column format. It is not yet tailored
to the additional requirements of a specific IEEE journal or conference.

## Open in Overleaf

1. Upload `gcn_alns_ieee_overleaf_draft.zip` using **New Project -> Upload Project**.
2. Set **Main document** to `main.tex`.
3. Use **pdfLaTeX** and the normal **BibTeX/latexmk** build. TikZ figures are
   source files and need no shell escape or external drawing software.
4. Replace the author/affiliation placeholder in `main.tex`.
5. Edit the individual files in `sections/`; edit citations in `references.bib`.

For a conference, change `\documentclass[journal]{IEEEtran}` to
`\documentclass[conference]{IEEEtran}` and adapt the author block and page
limit to the venue. A named IEEE venue can require its own template, so check
the venue instructions before submission.

## Manuscript status

- Introduction, related work, system model, mathematical formulation, method,
  pseudocode, experimental design, limitations, and a provisional conclusion
  are written in English.
- Two editable TikZ diagrams explain the recovery budget and training/search
  architecture.
- Final comparative results are explicitly marked as pending. The abstract
  makes no empirical improvement claim. Historical results from the old
  verifier and partial test results are not presented as final findings.
- Completed setup facts are distinguished from test outcomes: teacher input
  roles are 60/15; retained demonstration instances are 26/4; the independent
  test set has 90 instances. Validation selected A2-off before testing.
- `macros.tex` controls editorial notes with `\draftnotestrue` or
  `\draftnotesfalse`. Pending scientific results remain visible in either
  mode; hiding notes does not make the draft submission-ready.

## Paper scope (updated 2026-10-28)

The manuscript presents one validation-selected GCN-ALNS model. Its three
research questions cover the main comparison, component analysis, and budget
efficiency (19,200 planned records). A protection-mode prediction head is not
a separate research contribution. The independent mode-head research question
and model A/B result subsection have been removed.

The running pipeline keeps its original 20,550-record plan, including 1,350
model-comparison diagnostic records. These records and alternative checkpoints
remain in the development archive. They do not create an additional manuscript
experiment or replace the checkpoint already frozen by validation.

## Results integration

The independent rerun finished on 2026-11-04 (Beijing time). The interpretation
audit is recorded in the repository at
`analysis/independent_v1/INTERPRETATION_ZH.md`. Two legacy protection-policy
ablation labels were found not to enforce globally homogeneous modes; the
draft now describes them as repair-mode preferences and marks the limitation.
The final results narrative still needs integration.

Twelve corrected main-study figures are available in the repository under
`results/independent_v1/test/figures_corrected/` (PNG, SVG, plotted CSV, and
provenance). Use these rather than the original cost-breakdown, mixed-study
scalability, or run-weighted overall plots. The corrected figures are packaged
separately as `results/independent_v1/test/figures_corrected.zip`.

The optional Python helper `tools/update_results.py` reads completed snapshots
from `results/independent_v1`. It does not run an experiment, change a model,
or write to the experimental results tree. Run it locally from the repository:

```powershell
python paper/ieee_draft/tools/update_results.py --results-root results/independent_v1
```

It refuses incomplete three-scale main results and mismatched provenance. It
updates only the paper's generated main-result tables and an export manifest.
The abstract, interpretation, confidence intervals, ablation discussion, and
conclusion still require an author review after the full rerun. See
`notes/author_notes.md` for the remaining scientific editing tasks.

## Build locally

With a standard TeX Live/MiKTeX installation:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The dependency-free source checker can also be run from the repository:

```powershell
python paper/ieee_draft/tools/check_project.py
```

It checks local inputs, citation keys, labels, environments, and braces. It
does not replace compilation or visual inspection.

The complete source project requires only standard LaTeX packages available on
Overleaf. No experiment source file, configuration, model, or registered study
plan is modified by this paper project.

## Template sources

- IEEE author templates: https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/
- IEEE/Overleaf templates: https://www.overleaf.com/org/ieee
- IEEEtran class and bibliography style: https://ctan.org/pkg/ieeetran

The class and bibliography style are supplied by the standard TeX installation
on Overleaf; this project does not modify IEEEtran or bundle third-party class
files. Reference-checking links are in `notes/reference_verification.md`.

## Verification status

Local source checks resolve the inputs, citations, labels, environments, and
braces. This machine has no LaTeX compiler, and the optional portable compiler
download did not finish. No PDF has been compiled or visually verified here.
Compile `main.tex` in Overleaf and inspect equation widths, table placement,
and figure/pseudocode floats before treating the layout as final.
