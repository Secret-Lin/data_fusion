[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [string]$Output = "",
    [string]$OutputParent = "",
    [string]$Takes = "",
    [int]$MaxFrames = 0,
    [int]$MaxTakes = 0,
    [string]$FusionConfig = "",
    [switch]$LeaveOneOut,
    [double]$Fps = 30.0,
    # Empty uses the Python registry, so new variants cannot be silently missed.
    [string]$Variants = ""
)

$ErrorActionPreference = "Stop"
if ($Output -and $OutputParent) { throw "Use Output or OutputParent, not both" }
if ($env:CONDA_DEFAULT_ENV -ne "mr_nav") {
    $condaCommand = Get-Command conda.exe -ErrorAction SilentlyContinue
    $condaExe = if ($null -ne $condaCommand) { $condaCommand.Source } else {
        "conda.exe"
    }
    if (-not (Test-Path -LiteralPath $condaExe)) { throw "Cannot find conda.exe" }
    $activation = & $condaExe shell.powershell activate mr_nav
    if ($LASTEXITCODE -ne 0) { throw "Conda activation failed" }
    Invoke-Expression ($activation -join "`n")
}
if ($env:CONDA_DEFAULT_ENV -ne "mr_nav") { throw "mr_nav must be active" }

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonArgs = @(
    "-u", "-m", "tools.run_h2o_foundationpose_ablation",
    "--run-dir", $RunDir,
    "--max-frames", $MaxFrames.ToString(),
    "--max-takes", $MaxTakes.ToString(),
    "--fps", $Fps.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)
if ($Variants) { $pythonArgs += @("--variants", $Variants) }
if ($LeaveOneOut) { $pythonArgs += "--leave-one-out" }
if ($Output) { $pythonArgs += @("--output", $Output) }
if ($OutputParent) { $pythonArgs += @("--output-parent", $OutputParent) }
if ($Takes) { $pythonArgs += @("--takes", $Takes) }
if ($FusionConfig) { $pythonArgs += @("--fusion-config", $FusionConfig) }

Push-Location -LiteralPath $repoRoot
try {
    & python @pythonArgs
    if ($LASTEXITCODE -ne 0) { throw "H2O cached-frontend ablation has failures (exit code $LASTEXITCODE); inspect summary.json failed_methods/failed_takes and per-variant logs." }
}
finally {
    Pop-Location
}
