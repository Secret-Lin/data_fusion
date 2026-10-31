[CmdletBinding()]
param(
    [string]$DataRoot = 'data/h2o',
    [string]$RunName = 'results_0',
    [string]$OutputRoot = 'data/h2o/checkpoint_ablation',
    [ValidateSet('foundationpose', 'm3t', 'megapose')]
    [string[]]$Frontends = @('foundationpose', 'm3t', 'megapose'),
    [ValidateSet(1, 2, 3, 4)][int[]]$Subjects = @(1, 2, 3, 4),
    [switch]$LeaveOneOut = $true,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'run_h2o_all_subjects_ablation.ps1'
$sources = @{
    foundationpose = 'results_all_subjects'
    m3t = 'm3t_all_subjects'
    megapose = 'megapose_all_subjects'
}
$jobs = @()
foreach ($frontend in ($Frontends | Select-Object -Unique)) {
    $options = @{
        RunRoot = Join-Path $DataRoot $sources[$frontend]
        RunName = $RunName
        OutputParent = Join-Path $OutputRoot $frontend
        Subjects = $Subjects
        LeaveOneOut = [bool]$LeaveOneOut
    }
    # Validate every input before launching any computation.
    & $launcher @options -DryRun
    $jobs += [pscustomobject]@{ Frontend = $frontend; Options = $options }
}
if ($DryRun) {
    Write-Host 'All inputs checked. No fusion, tracking or result writes were started.'
    return
}
Write-Host 'Reusing cached poses only. H2O checkpoint fusion default; live defaults unchanged.'
Write-Host "New outputs: $OutputRoot\<frontend>\subjectN\results_N"
$failed = @()
foreach ($job in $jobs) {
    Write-Host "Starting $($job.Frontend) cached ablations."
    try {
        $options = $job.Options
        & $launcher @options
    } catch {
        $failed += $job.Frontend
        Write-Warning "$($job.Frontend): $($_.Exception.Message)"
    }
}
if ($failed.Count) {
    throw "Ablation failures in $($failed -join ', '); inspect per-variant reports under $OutputRoot."
}
Write-Host "All requested cached ablations completed: $OutputRoot"
