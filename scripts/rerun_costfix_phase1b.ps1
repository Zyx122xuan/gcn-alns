# Phase-1b: finish the tags that did not complete (main_medium resume,
# main_hard_medium, ablation) + aggregate + plots. Lower workers to avoid the
# resource spike that killed the first main_medium run. Resume-safe.
# Paths are relative to the repo root, not this script's folder.
Set-Location (Split-Path -Parent $PSScriptRoot)
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
$WORKERS = 30

New-Item -ItemType Directory -Force -Path results/logs | Out-Null
$MASTER = "results/logs/phase1b_master.log"
function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    $line | Tee-Object -FilePath $MASTER -Append
}

$HARD_ALGOS = @("HEFT","PEFT","No-FT","Reactive-Only","Full-Proactive",
                "Critical-Path Backup","Criticality-Based Hybrid","Reliability-Aware",
                "ALNS","GCN-ALNS")

function Run-Step($name, [scriptblock]$cmd) {
    Log "START  $name"
    $t0 = Get-Date
    & $cmd *>&1 | Tee-Object -FilePath "results/logs/$name.log" -Append
    $dt = [int]((Get-Date) - $t0).TotalSeconds
    Log "DONE   $name (exit=$LASTEXITCODE, ${dt}s)"
}

Log "=== PHASE 1b BEGIN (workers=$WORKERS) ==="

# main_medium: resume (1381/1500 already done -> only the rest run)
Run-Step "main_medium" {
    python -m gcnalns.experiments.run_main --instances data/instances/medium `
        --out results --workers $WORKERS --repeats 10 --max-iter 1000 --tag main_medium
}
Run-Step "main_hard_medium" {
    python -m gcnalns.experiments.run_main --instances data/instances/hard_medium `
        --out results --workers $WORKERS --repeats 10 --max-iter 1000 --tag main_hard_medium `
        --algorithms $HARD_ALGOS
}
Run-Step "ablation" {
    python -m gcnalns.experiments.run_ablation `
        --instances data/instances/small data/instances/medium `
        --out results --workers $WORKERS --repeats 5 --max-iter 1000
}
Run-Step "aggregate" {
    python -m gcnalns.eval.aggregate --raw results/raw_json --out results/csv
}
Run-Step "plots" {
    python -m gcnalns.viz.plots --csv results/csv --out results/figures
}

Log "=== PHASE 1b COMPLETE ==="
