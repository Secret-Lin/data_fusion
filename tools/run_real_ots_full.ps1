[CmdletBinding()]
param(
    [string[]]$Sessions = @('data/laboratory/20260915_155343', 'data/laboratory/20260915_155901'),
    [string]$OutputParent = '',
    [string]$CalibrationDir = '',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputParent) { $OutputParent = Join-Path $repoRoot 'artifacts/real_ots_full' }
if (-not $CalibrationDir) { $CalibrationDir = Join-Path $repoRoot 'artifacts/ots_probe_analysis/results_6' }
if ($env:CONDA_DEFAULT_ENV -ne 'mr_nav') {
    $condaCommand = Get-Command conda.exe -ErrorAction SilentlyContinue
    $condaExe = if ($condaCommand) { $condaCommand.Source } else { 'conda.exe' }
    $activation = & $condaExe shell.powershell activate mr_nav
    if ($LASTEXITCODE -ne 0) { throw 'Conda activation failed.' }
    Invoke-Expression ($activation -join "`n")
}
if ($env:CONDA_DEFAULT_ENV -ne 'mr_nav') { throw 'mr_nav must be active.' }
Push-Location $repoRoot
try {
    # Check BOTH inputs before beginning an expensive GPU run.
    foreach ($session in $Sessions) {
        & python -m tools.preflight_real_ots --session $session --calibration-dir $CalibrationDir
        if ($LASTEXITCODE -ne 0) { throw "Input check failed: $session" }
    }
    if ($DryRun) {
        Write-Host "Input checks passed. No tracking started. Output parent: $OutputParent"
        return
    }
    $failed = @()
    foreach ($session in $Sessions) {
        # One process / one session: avoids GPU competition and releases model memory.
        & python -u -m tools.run_real_ots_smoke --session $session --full-sequence --calibration-dir $CalibrationDir --output-parent $OutputParent
        if ($LASTEXITCODE -ne 0) {
            $failed += $session
            Write-Warning "Run or post-evaluation failed: $session. Partial outputs are retained."
        }
    }
    if ($failed.Count) { throw "Failed sessions: $($failed -join ', ')" }
    Write-Host "Completed both inference and evaluation. Results: $OutputParent"
} finally {
    Pop-Location
}
