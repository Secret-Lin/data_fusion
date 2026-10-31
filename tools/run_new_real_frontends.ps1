[CmdletBinding()]
param(
    [ValidateSet('m3t','megapose')][string[]]$Frontends = @('m3t','megapose'),
    [string[]]$Sessions = @('stable-0','stable-1','stable-2','Test-0','Test-1','Test-2','Test-3','Test-4'),
    [string]$DataRoot = 'data/laboratory',
    [string]$SeedParent = 'artifacts/new_real_full',
    [string]$SeedRunName = 'results_0',
    [string]$CalibrationDir = 'artifacts/new_real_calibration/results_1',
    [string]$ReferenceDir = 'artifacts/new_real_reference/results_1',
    [switch]$WithAblation,
    [switch]$LeaveOneOut,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ($env:CONDA_DEFAULT_ENV -ne 'mr_nav') { throw 'Activate conda mr_nav first.' }
Push-Location $repoRoot
try {
    # Check every source/seed before beginning expensive tracking.
    foreach ($name in $Sessions) {
        & python -m tools.preflight_real_ots --session (Join-Path $DataRoot $name) --calibration-dir $CalibrationDir --reference-dir $ReferenceDir
        if ($LASTEXITCODE -ne 0) { throw "Input check failed: $name" }
        $seed = Join-Path (Join-Path $SeedParent $name) $SeedRunName
        & python -m tools.run_real_frontend --frontend $Frontends[0] --session (Join-Path $DataRoot $name) --seed-run-dir $seed --full-sequence --calibration-dir $CalibrationDir --reference-dir $ReferenceDir --check-only
        if ($LASTEXITCODE -ne 0) { throw "Seed check failed: $name" }
    }
    if ($DryRun) { Write-Host 'All data/seed checks passed; no tracking or fusion started.'; return }
    $failed = @()
    foreach ($frontend in $Frontends) {
        foreach ($name in $Sessions) {
            $parent = "artifacts/new_real_$frontend"
            $sessionParent = Join-Path $parent $name
            $before = @(Get-ChildItem -LiteralPath $sessionParent -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
            $seed = Join-Path (Join-Path $SeedParent $name) $SeedRunName
            & python -u -m tools.run_real_frontend --frontend $frontend --session (Join-Path $DataRoot $name) --seed-run-dir $seed --full-sequence --calibration-dir $CalibrationDir --reference-dir $ReferenceDir --output-parent $parent
            if ($LASTEXITCODE -ne 0) { $failed += "$frontend/$name tracking"; continue }
            $created = @(Get-ChildItem -LiteralPath $sessionParent -Directory | Where-Object { $_.Name -match '^results_\d+$' -and $_.FullName -notin $before })
            if ($created.Count -ne 1) { throw "Expected one new run in $sessionParent; do not run duplicate batches concurrently." }
            $run = $created[0].FullName
            & python -m tools.check_real_ots_outputs $run
            if ($LASTEXITCODE -ne 0) { $failed += "$frontend/$name output validation"; continue }
            if ($WithAblation) {
                $ablationArgs = @('-u','-m','tools.run_real_ots_ablation','--run-dir',$run,'--full-sequence',
                                 '--calibration-dir',$ReferenceDir,'--singleton-guard','--output-parent',"artifacts/new_real_${frontend}_ablation")
                if ($LeaveOneOut) { $ablationArgs += '--leave-one-out' }
                & python @ablationArgs
                if ($LASTEXITCODE -ne 0) { $failed += "$frontend/$name ablation" }
            }
        }
    }
    if ($failed.Count) { throw "Failed jobs (partial outputs retained): $($failed -join ', ')" }
    Write-Host 'Complete: artifacts/new_real_<frontend>/<session>/results_#'
    if ($WithAblation) { Write-Host 'Ablations: artifacts/new_real_<frontend>_ablation/<session>/results_#' }
} finally { Pop-Location }
