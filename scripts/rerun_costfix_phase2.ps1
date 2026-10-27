# Phase-2 re-run after cost-model fix: large + hard_large only.
# Resume-safe: each tag skips already-completed (tag,algo,iid,rep) items,
# so an interruption (e.g. Windows auto-restart) can be recovered by re-running this script.
# Config matches the original large runs exactly:
#   main_large      : full 14 algorithms (incl. external baselines), repeats=5, max-iter=800
#   main_hard_large : 10-algo hard set (no external), repeats=5, max-iter=800, workers=8
# Paths are relative to the repo root, not this script's folder.
Set-Location (Split-Path -Parent $PSScriptRoot)
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"

# main_large: 500-task solves are memory-heavy; keep workers modest to avoid OOM.
$WORKERS_LARGE = 12
# hard_large: bumped 8->16 to accelerate the remaining stochastic solves while
# the machine is idle (ablation finished). Resume-safe; original run used 8.
$WORKERS_HARD  = 16

New-Item -ItemType Directory -Force -Path results/logs | Out-Null
$MASTER = "results/logs/phase2_master.log"
function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    $line | Tee-Object -FilePath $MASTER -Append
}

# 8 built-in baselines + ALNS + GCN-ALNS (hard-tier algorithm set; no external baselines)
$HARD_ALGOS = @("HEFT","PEFT","No-FT","Reactive-Only","Full-Proactive",
                "Critical-Path Backup","Criticality-Based Hybrid","Reliability-Aware",
                "ALNS","GCN-ALNS")

function Run-Step($name, [scriptblock]$cmd) {
    Log "START  $name"
    $t0 = Get-Date
    & $cmd *>&1 | Tee-Object -FilePath "results/logs/$name.log"
    $dt = [int]((Get-Date) - $t0).TotalSeconds
    Log "DONE   $name (exit=$LASTEXITCODE, ${dt}s)"
}

Log "=== PHASE 2 BEGIN (large=$WORKERS_LARGE, hard=$WORKERS_HARD) ==="

# --- main large (full 14 algorithms, repeats=5, iter=800) ---
Run-Step "main_large" {
    python -m gcnalns.experiments.run_main --instances data/instances/large `
        --out results --workers $WORKERS_LARGE --repeats 5 --max-iter 800 --tag main_large
}

# --- main hard large (10-algo hard set, repeats=5, iter=800, workers=8) ---
Run-Step "main_hard_large" {
    python -m gcnalns.experiments.run_main --instances data/instances/hard_large `
        --out results --workers $WORKERS_HARD --repeats 5 --max-iter 800 --tag main_hard_large `
        --algorithms $HARD_ALGOS
}

# --- final aggregate + figures (now with all three scales) ---
Run-Step "aggregate" {
    python -m gcnalns.eval.aggregate --raw results/raw_json --out results/csv
}
Run-Step "plots" {
    python -m gcnalns.viz.plots --csv results/csv --out results/figures
}

Log "=== PHASE 2 COMPLETE ==="
