[CmdletBinding()]
param(
    [string]$DataRoot = "data/h2o",
    [ValidateNotNullOrEmpty()][ValidateSet(1, 2, 3, 4)][int[]]$Subjects = @(1, 2, 3, 4),
    [string]$OutputParent = "",
    [ValidateRange(0, 64)][int]$IoWorkers = 4,
    [ValidateRange(1, 64)][int]$PrefetchFrames = 4,
    [ValidateRange(0, 64)][int]$TorchThreads = 4,
    [ValidateRange(0, 2147483647)][int]$MaxTakes = 0,
    [ValidateRange(0, 2147483647)][int]$MaxFrames = 0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$data = (Resolve-Path -LiteralPath $DataRoot).Path
$meshSource = Join-Path $data "object.zip"
if (-not (Test-Path -LiteralPath $meshSource -PathType Leaf)) {
    throw "H2O mesh archive is missing: $meshSource"
}
$destination = if ($OutputParent) {
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputParent)
} else {
    Join-Path $data "results_all_subjects"
}
$launcher = Join-Path $PSScriptRoot "run_h2o_foundationpose.ps1"
$jobs = @()
foreach ($number in ($Subjects | Select-Object -Unique)) {
    $name = "subject$number"
    $subjectRoot = Join-Path $data $name
    if (-not (Test-Path -LiteralPath $subjectRoot -PathType Container)) {
        throw "Extract $name first; directory is missing: $subjectRoot"
    }
    $takes = @(foreach ($group in Get-ChildItem -LiteralPath $subjectRoot -Directory) {
        Get-ChildItem -LiteralPath $group.FullName -Directory |
            Where-Object { $_.Name -match '^\d+$' }
    })
    if (-not $takes.Count) { throw "No group/take directories found below $subjectRoot" }
    $jobs += [pscustomobject]@{
        Subject = $name
        Root = $subjectRoot
        Takes = $takes.Count
        OutputParent = Join-Path $destination $name
    }
}

Write-Host "H2O FoundationPose + default fusion: all valid cameras, oracle_gt first-frame mask."
Write-Host "Depth-fit quality enabled; readers=$IoWorkers, prefetch=$PrefetchFrames, torch threads=$TorchThreads."
Write-Host "Limits per subject: MaxTakes=$MaxTakes; per take: MaxFrames=$MaxFrames (0 = all)."
foreach ($job in $jobs) {
    Write-Host "$($job.Subject): $($job.Takes) take directories -> $($job.OutputParent)\results_N"
}
if ($DryRun) {
    Write-Host "Dry run completed: no inference, fusion or output directories created."
    return
}

# Reuse the existing launcher: it activates mr_nav, allocates results_N and
# records per-take observations, fusion, metrics, settings and failures.
$failedSubjects = @()
foreach ($job in $jobs) {
    Write-Host "Starting $($job.Subject) ..."
    $runOptions = @{
        SubjectRoot = $job.Root
        MeshSource = $meshSource
        OutputParent = $job.OutputParent
        AllTakes = $true
        Cameras = "all"
        InitMode = "oracle_gt"
        DepthFitQuality = $true
        IoWorkers = $IoWorkers
        PrefetchFrames = $PrefetchFrames
        TorchThreads = $TorchThreads
        MaxTakes = $MaxTakes
        MaxFrames = $MaxFrames
    }
    try {
        & $launcher @runOptions
    } catch {
        # A failed take is recorded by the existing runner. Keep processing the
        # other subjects and report the failure at the end of this batch.
        $failedSubjects += $job.Subject
        Write-Warning "$($job.Subject): $($_.Exception.Message)"
    }
}
if ($failedSubjects.Count) {
    throw "Batch finished with failures in: $($failedSubjects -join ', '). Review their summary.json and logs under $destination."
}
Write-Host "All requested subjects completed. Results: $destination"
