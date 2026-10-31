[CmdletBinding()]
param(
    [string]$SubjectRoot = "data/h2o/subject1",
    [string]$OutputRoot = "",
    [double]$Fps = 30.0,
    [int]$Seed = 7,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Enable-MrNavEnvironment {
    if ($env:CONDA_DEFAULT_ENV -eq "mr_nav") {
        return
    }

    $condaCommand = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($null -ne $condaCommand) {
        $condaExe = $condaCommand.Source
    }
    else {
        $condaExe = "conda.exe"
    }

    if (-not (Test-Path -LiteralPath $condaExe)) {
        throw "Cannot find conda.exe. Activate the mr_nav environment manually."
    }

    $activation = & $condaExe shell.powershell activate mr_nav
    Invoke-Expression ($activation -join "`n")
    if ($env:CONDA_DEFAULT_ENV -ne "mr_nav") {
        throw "Failed to activate conda environment mr_nav"
    }
}

Enable-MrNavEnvironment

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$subjectPath = [System.IO.Path]::GetFullPath($SubjectRoot)
$outputPath = if ($OutputRoot) { [System.IO.Path]::GetFullPath($OutputRoot) } else { "" }

if (-not (Test-Path -LiteralPath $subjectPath -PathType Container)) {
    throw "H2O subject directory does not exist: $subjectPath"
}

$takes = @()
$groups = Get-ChildItem -LiteralPath $subjectPath -Directory | Sort-Object Name
foreach ($group in $groups) {
    $groupTakes = Get-ChildItem -LiteralPath $group.FullName -Directory |
        Where-Object { $_.Name -match '^\d+$' } |
        Sort-Object { [int]$_.Name }

    foreach ($take in $groupTakes) {
        $validCameras = Get-ChildItem -LiteralPath $take.FullName -Directory |
            Where-Object {
                $_.Name -match '^cam\d+$' -and
                (Test-Path -LiteralPath (Join-Path $_.FullName "cam_pose")) -and
                (
                    (Test-Path -LiteralPath (Join-Path $_.FullName "obj_pose_rt")) -or
                    (Test-Path -LiteralPath (Join-Path $_.FullName "obj_pose_RT"))
                )
            }
        if ($validCameras.Count -eq 0) {
            Write-Warning "Skipping take without valid cameras: $($take.FullName)"
            continue
        }

        $takes += [pscustomobject]@{
            Group = $group.Name
            Take = $take.Name
            Path = $take.FullName
            CameraCount = $validCameras.Count
        }
    }
}

if ($takes.Count -eq 0) {
    throw "No valid H2O takes found below $subjectPath"
}

if (-not $outputPath) {
    $resultsParent = Split-Path -Parent $subjectPath
    $resultIndices = @(Get-ChildItem -LiteralPath $resultsParent -Directory |
        Where-Object { $_.Name -match '^results_\d+$' } |
        ForEach-Object { [int]$_.Name.Substring(8) })
    $nextIndex = if ($resultIndices.Count) { ($resultIndices | Measure-Object -Maximum).Maximum + 1 } else { 0 }
    $outputPath = Join-Path $resultsParent "results_$nextIndex"
    if (-not $DryRun) {
        # No -Force: a concurrently created result must never be reused.
        New-Item -ItemType Directory -Path $outputPath | Out-Null
    }
}

Write-Host "H2O subject: $subjectPath"
Write-Host "Output root: $outputPath"
Write-Host "Valid takes: $($takes.Count)"
Write-Host "Conda environment: $env:CONDA_DEFAULT_ENV"

if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
}

$completed = 0
$skipped = 0
$failed = 0
$originalLocation = Get-Location

try {
    Set-Location -LiteralPath $repoRoot

    foreach ($item in $takes) {
        $runName = "{0}_{1}" -f $item.Group, $item.Take
        $runOutput = Join-Path $outputPath $runName
        $reportPath = Join-Path $runOutput "report.json"

        if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
            $existingReport = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
            if ($existingReport.synthetic_noise -ne $false) {
                throw "Existing result contains synthetic noise or unknown provenance: $reportPath. Use a new results_N."
            }
            Write-Host "[SKIP] $runName (report.json exists)"
            $skipped += 1
            continue
        }

        $pythonArgs = @(
            "-m", "experiments.h2o.benchmark",
            "--sequence", $item.Path,
            "--cameras", "all",
            "--output", $runOutput,
            "--max-frames", "0",
            "--no-synthetic-noise",
            "--fps", ([string]::Format(
                [System.Globalization.CultureInfo]::InvariantCulture,
                "{0}",
                $Fps
            )),
            "--seed", $Seed.ToString()
        )

        if ($DryRun) {
            Write-Host "[DRY RUN] $runName cameras=$($item.CameraCount)"
            Write-Host "python $($pythonArgs -join ' ')"
            continue
        }

        Write-Host "[RUN] $runName cameras=$($item.CameraCount)"
        & python @pythonArgs
        if ($LASTEXITCODE -ne 0) {
            $failed += 1
            throw "H2O benchmark failed: $runName (exit code $LASTEXITCODE)"
        }
        $completed += 1
    }
}
finally {
    Set-Location -LiteralPath $originalLocation
}

Write-Host "Finished: completed=$completed skipped=$skipped failed=$failed"
if ($DryRun) {
    Write-Host "Dry run only; no result files were created."
}
