[CmdletBinding()]
param(
    [string]$SubjectRoot = "data/h2o/subject1",
    [string]$OutputRoot = "",
    [double]$Fps = 30.0,
    [int]$MaxFrames = 0,
    [int]$MaxTakes = 0
)

$ErrorActionPreference = "Stop"
if ($env:CONDA_DEFAULT_ENV -ne "mr_nav") {
    $condaCommand = Get-Command conda.exe -ErrorAction SilentlyContinue
    $condaExe = if ($null -ne $condaCommand) { $condaCommand.Source } else {
        "conda.exe"
    }
    if (-not (Test-Path -LiteralPath $condaExe)) { throw "Activate conda mr_nav first." }
    $activation = & $condaExe shell.powershell activate mr_nav
    if ($LASTEXITCODE -ne 0) { throw "Conda activation failed" }
    Invoke-Expression ($activation -join "`n")
}
if ($env:CONDA_DEFAULT_ENV -ne "mr_nav") { throw "mr_nav must be active" }
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonArgs = @("-u", "-m", "experiments.h2o.consistency", "--subject-root", $SubjectRoot,
    "--fps", $Fps.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--max-frames", $MaxFrames.ToString(), "--max-takes", $MaxTakes.ToString())
if ($OutputRoot) { $pythonArgs += @("--output", $OutputRoot) }
Push-Location -LiteralPath $repoRoot
try {
    Write-Host "Environment: $env:CONDA_DEFAULT_ENV; no artificial noise"
    & python @pythonArgs
    if ($LASTEXITCODE -ne 0) { throw "H2O consistency check failed (exit $LASTEXITCODE)" }
}
finally { Pop-Location }
