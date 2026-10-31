[CmdletBinding()]
param(
    [string[]]$Sessions = @('stable-0','stable-1','stable-2','Test-0','Test-1','Test-2','Test-3','Test-4'),
    [string]$RunParent = 'artifacts/new_real_full',
    [string]$RunName = 'results_0',
    [string]$ReferenceDir = 'artifacts/new_real_reference/results_1',
    [string]$OutputParent = 'artifacts/new_real_ablation',
    [switch]$LeaveOneOut,
    [switch]$SingletonGuard,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Push-Location $repoRoot
try {
    $runs = @($Sessions | ForEach-Object { Join-Path (Join-Path $RunParent $_) $RunName })
    Write-Host 'Test-5 is excluded by default: independent OTS time matching is unresolved, not a frontend failure.'
    & (Join-Path $PSScriptRoot 'run_real_ots_ablation.ps1') -RunDirs $runs -CalibrationDir $ReferenceDir `
        -OutputParent $OutputParent -LeaveOneOut:$LeaveOneOut -SingletonGuard:$SingletonGuard -DryRun:$DryRun
} finally { Pop-Location }
