[CmdletBinding()]
param(
    [string]$Take = "",
    [switch]$AllTakes,
    [string]$SubjectRoot = "data/h2o/subject1",
    [string]$MeshSource = "data/h2o/object.zip",
    [string]$Output = "",
    [string]$OutputParent = "",
    [string]$MeshCache = "",
    [string]$Cameras = "all",
    [double]$Fps = 30.0,
    [string]$FusionConfig = "",
    [int]$MaxFrames = 0,
    [int]$MaxTakes = 0,
    [ValidateSet("oracle_gt", "mask_dir")][string]$InitMode = "oracle_gt",
    [string]$MaskRoot = "",
    [int]$RegisterIterations = 5,
    [int]$TrackIterations = 2,
    [ValidateRange(0, 64)][int]$IoWorkers = 4,
    [ValidateRange(1, 64)][int]$PrefetchFrames = 4,
    [ValidateRange(0, 64)][int]$TorchThreads = 4,
    [switch]$ScoreTracking,
    [switch]$DepthFitQuality
)

$ErrorActionPreference = "Stop"
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
if ($Output -and $OutputParent) { throw "Use Output or OutputParent, not both" }

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonArgs = @(
    "-u", "-m", "tools.run_h2o_foundationpose",
    "--subject-root", $SubjectRoot,
    "--mesh-source", $MeshSource,
    "--cameras", $Cameras,
    "--fps", $Fps.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--max-frames", $MaxFrames.ToString(),
    "--max-takes", $MaxTakes.ToString(),
    "--init-mode", $InitMode,
    "--register-iterations", $RegisterIterations.ToString(),
    "--track-iterations", $TrackIterations.ToString(),
    "--io-workers", $IoWorkers.ToString(),
    "--prefetch-frames", $PrefetchFrames.ToString(),
    "--torch-threads", $TorchThreads.ToString()
)
if ($Take) { $pythonArgs += @("--take", $Take) }
if ($AllTakes) { $pythonArgs += "--all-takes" }
if ($ScoreTracking) { $pythonArgs += "--score-tracking" }
if ($DepthFitQuality) { $pythonArgs += "--depth-fit-quality" }
if ($MeshCache) { $pythonArgs += @("--mesh-cache", $MeshCache) }
if ($MaskRoot) { $pythonArgs += @("--mask-root", $MaskRoot) }
if ($Output) { $pythonArgs += @("--output", $Output) }
if ($OutputParent) { $pythonArgs += @("--output-parent", $OutputParent) }
if ($FusionConfig) { $pythonArgs += @("--fusion-config", $FusionConfig) }

Push-Location -LiteralPath $repoRoot
try {
    & python @pythonArgs
    if ($LASTEXITCODE -ne 0) { throw "H2O FoundationPose run failed (exit code $LASTEXITCODE)." }
}
finally {
    Pop-Location
}
