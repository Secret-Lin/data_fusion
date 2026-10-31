[CmdletBinding()]
param(
    [string]$RunRoot = "data/h2o/results_all_subjects",
    [string]$RunName = "results_0",
    [ValidateNotNullOrEmpty()][ValidateSet(1, 2, 3, 4)][int[]]$Subjects = @(1, 2, 3, 4),
    [string]$OutputParent = "data/h2o/ablation_all_subjects",
    [string]$Takes = "",
    [ValidateRange(0, 2147483647)][int]$MaxTakes = 0,
    [ValidateRange(0, 2147483647)][int]$MaxFrames = 0,
    [switch]$LeaveOneOut = $true,
    [string]$Variants = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
if ($RunName -notmatch '^results_\d+$') { throw "RunName must be results_N" }
$root = (Resolve-Path -LiteralPath $RunRoot).Path
$destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputParent)
$jobs = @()
foreach ($number in ($Subjects | Select-Object -Unique)) {
    $subject = "subject$number"
    $run = Join-Path (Join-Path $root $subject) $RunName
    $summaryPath = Join-Path $run "summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) { throw "Missing completed run: $summaryPath" }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    if ($null -eq $summary.n_takes_requested -or $summary.n_takes_requested -le 0 -or
        $summary.n_takes_failed -ne 0 -or $summary.n_takes_completed -ne $summary.n_takes_requested) {
        throw "Incomplete source run: $run; inspect summary.json before proceeding"
    }
    $jobs += [pscustomobject]@{
        Subject = $subject
        RunDir = $run
        Takes = $summary.n_takes_completed
        OutputParent = Join-Path $destination $subject
    }
}
Write-Host "H2O cached fusion ablations; LeaveOneOut=$LeaveOneOut."
Write-Host "Limits: MaxTakes=$MaxTakes, MaxFrames=$MaxFrames (0 = all); Takes='$Takes'."
foreach ($job in $jobs) {
    Write-Host "$($job.Subject): $($job.Takes) stored takes -> $($job.OutputParent)\results_N"
}
if ($DryRun) {
    Write-Host "Dry run completed: inputs checked; no fusion or output directories created."
    return
}
$launcher = Join-Path $PSScriptRoot "run_h2o_foundationpose_ablation.ps1"
$failedSubjects = @()
foreach ($job in $jobs) {
    $options = @{
        RunDir = $job.RunDir
        OutputParent = $job.OutputParent
        Takes = $Takes
        MaxTakes = $MaxTakes
        MaxFrames = $MaxFrames
        LeaveOneOut = [bool]$LeaveOneOut
        Variants = $Variants
    }
    try {
        & $launcher @options
    } catch {
        $failedSubjects += $job.Subject
        Write-Warning "$($job.Subject): $($_.Exception.Message)"
    }
}
if ($failedSubjects.Count) {
    throw "Ablation batch has failures in $($failedSubjects -join ', '); inspect reports under $destination."
}
Write-Host "All requested cached ablations completed. Results: $destination"
