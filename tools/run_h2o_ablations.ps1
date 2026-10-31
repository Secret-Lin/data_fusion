[CmdletBinding()]
param(
    [string]$SubjectRoot = "data/h2o/subject1",
    [string]$OutputRoot = "",
    [string]$Seeds = "0",
    [string]$Scenarios = "annotations",
    [string]$Variants = "full,no_adaptive,no_huber,no_gate,no_robust,no_grouping,arrival_grouping,no_integrity,reverse_order,weak_initial_prior,no_temporal,no_history,no_online_calibration,quality_consensus",
    [string]$Take = "",
    [string]$LegacyResults = "",
    [int]$MaxFrames = 0,
    [int]$MaxTakes = 0,
    [double]$Fps = 30.0,
    [switch]$LeaveOneOut,
    [switch]$Resume,
    [switch]$SummarizeOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
if ($env:CONDA_DEFAULT_ENV -ne "mr_nav") {
    $condaCommand = Get-Command conda.exe -ErrorAction SilentlyContinue
    $condaExe = if ($null -ne $condaCommand) { $condaCommand.Source } else {
        "conda.exe"
    }
    if (-not (Test-Path -LiteralPath $condaExe)) {
        throw "Cannot find conda.exe. Activate mr_nav before running this script."
    }
    $activation = & $condaExe shell.powershell activate mr_nav
    if ($LASTEXITCODE -ne 0) { throw "Conda activation failed" }
    Invoke-Expression ($activation -join "`n")
}
if ($env:CONDA_DEFAULT_ENV -ne "mr_nav") { throw "mr_nav must be active" }
if (($Resume -or $SummarizeOnly) -and [string]::IsNullOrWhiteSpace($OutputRoot)) {
    throw "Resume and SummarizeOnly require OutputRoot."
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonArgs = @(
    "-u", "-m", "experiments.h2o.ablation",
    "--subject-root", $SubjectRoot,
    "--seeds", $Seeds, "--scenarios", $Scenarios, "--variants", $Variants,
    "--max-frames", $MaxFrames.ToString(), "--max-takes", $MaxTakes.ToString(),
    "--fps", $Fps.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)
if ($OutputRoot) { $pythonArgs += @("--output", $OutputRoot) }
if ($Take) { $pythonArgs += @("--take", $Take) }
if ($LegacyResults) { $pythonArgs += @("--legacy-results", $LegacyResults) }
if ($LeaveOneOut) { $pythonArgs += "--leave-one-out" }
if ($Resume) { $pythonArgs += "--resume" }
if ($SummarizeOnly) { $pythonArgs += "--summarize-only" }
if ($DryRun) { $pythonArgs += "--dry-run" }

Push-Location -LiteralPath $repoRoot
try {
    Write-Host "Environment: $env:CONDA_DEFAULT_ENV"
    & python @pythonArgs
    if ($LASTEXITCODE -ne 0) { throw "H2O ablation failed (exit code $LASTEXITCODE)." }
}
finally {
    Pop-Location
}
