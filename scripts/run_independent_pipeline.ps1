param(
    [string]$Python = "python",
    [int]$Workers = 8,
    [int]$LargeWorkers = 12
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = Join-Path (Get-Location) "src"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
& $Python -u scripts/rerun_independent_v1.py --workers $Workers --large-workers $LargeWorkers
exit $LASTEXITCODE
