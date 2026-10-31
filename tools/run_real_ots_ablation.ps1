[CmdletBinding()]
param(
    [string[]]$RunDirs = @(
        'artifacts/real_ots_full/20260915_155343/results_0',
        'artifacts/real_ots_full/20260915_155901/results_0'
    ),
    [string]$OutputParent = 'artifacts/real_ots_ablation',
    [string]$CalibrationDir = 'artifacts/ots_probe_analysis/results_6',
    [switch]$LeaveOneOut,
    [switch]$SingletonGuard,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
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
    foreach ($dir in $RunDirs) {
        foreach ($file in @('manifest.json','observations.jsonl','input_timeline.json')) {
            if (-not (Test-Path -LiteralPath (Join-Path $dir $file))) { throw "Missing cache: $dir/$file" }
        }
        $manifest = Get-Content -LiteralPath (Join-Path $dir 'manifest.json') -Raw | ConvertFrom-Json
        if ($manifest.status -ne 'completed') { throw "Incomplete frontend run: $dir" }
        $sessionName = Split-Path -Leaf $manifest.session
        $reference = Join-Path $CalibrationDir "${sessionName}_reference_for_native_mesh.npz"
        if (-not (Test-Path -LiteralPath $reference)) { throw "Missing evaluation reference: $reference" }
        Write-Host "Cache ready: $dir ($($manifest.n_observations) observations)"
    }
    if ($DryRun) { Write-Host 'No fusion or frontend started.'; return }
    $failed = @()
    foreach ($dir in $RunDirs) {
        $pythonArgs = @('-u','-m','tools.run_real_ots_ablation','--run-dir',$dir,'--full-sequence','--output-parent',$OutputParent,'--calibration-dir',$CalibrationDir)
        if ($LeaveOneOut) { $pythonArgs += '--leave-one-out' }
        if ($SingletonGuard) { $pythonArgs += '--singleton-guard' }
        & python @pythonArgs
        if ($LASTEXITCODE -ne 0) { $failed += $dir; Write-Warning "Some variants failed; inspect $OutputParent" }
    }
    if ($failed.Count) { throw "Failed experiments: $($failed -join ', ')" }
    Write-Host "All cached experiments complete: $OutputParent"
} finally {
    Pop-Location
}
