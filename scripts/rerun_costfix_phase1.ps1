# Phase-1 re-run after cost-model fix (excludes large / hard_large; those are Phase 2).
# Resume-safe: each tag skips already-completed (tag,algo,iid,rep) items.
# Paths are relative to the repo root, not this script's folder.
Set-Location (Split-Path -Parent $PSScriptRoot)
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
$WORKERS = 30

New-Item -ItemType Directory -Force -Path results/logs | Out-Null
$MASTER = "results/logs/phase1_master.log"
function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    $line | Tee-Object -FilePath $MASTER -Append
}

# 8 built-in baselines + ALNS + GCN-ALNS (hard-tier algorithm set)
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

Log "=== PHASE 1 BEGIN (workers=$WORKERS) ==="

# --- main tiers (full 14 algorithms) ---
Run-Step "main_small" {
    python -m gcnalns.experiments.run_main --instances data/instances/small `
        --out results --workers $WORKERS --repeats 10 --max-iter 1000 --tag main_small
}
Run-Step "main_hard_small" {
    python -m gcnalns.experiments.run_main --instances data/instances/hard_small `
        --out results --workers $WORKERS --repeats 10 --max-iter 1000 --tag main_hard_small `
        --algorithms $HARD_ALGOS
}

# --- model A/B (A2-off=premodefix vs A2-on=current default) ---
Run-Step "model_ab" {
    python -m gcnalns.experiments.run_model_ab --instances data/instances/small `
        --out results --workers $WORKERS --repeats 10 --max-iter 1000 `
        --old-model results/gcn_model_premodefix.pt --new-model results/gcn_model.pt --with-alns
}

# --- iteration-budget sweep (small) ---
Run-Step "budget" {
    python -m gcnalns.experiments.run_budget --instances data/instances/small `
        --out results --workers $WORKERS --repeats 10
}

# --- medium tiers ---
Run-Step "main_medium" {
    python -m gcnalns.experiments.run_main --instances data/instances/medium `
        --out results --workers $WORKERS --repeats 10 --max-iter 1000 --tag main_medium
}
Run-Step "main_hard_medium" {
    python -m gcnalns.experiments.run_main --instances data/instances/hard_medium `
        --out results --workers $WORKERS --repeats 10 --max-iter 1000 --tag main_hard_medium `
        --algorithms $HARD_ALGOS
}

# --- ablation (14 variants, small+medium, GCN-ALNS only) ---
Run-Step "ablation" {
    python -m gcnalns.experiments.run_ablation `
        --instances data/instances/small data/instances/medium `
        --out results --workers $WORKERS --repeats 5 --max-iter 1000
}

# --- final aggregate + figures ---
Run-Step "aggregate" {
    python -m gcnalns.eval.aggregate --raw results/raw_json --out results/csv
}
Run-Step "plots" {
    python -m gcnalns.viz.plots --csv results/csv --out results/figures
}

Log "=== PHASE 1 COMPLETE ==="
