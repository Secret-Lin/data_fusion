[CmdletBinding()]
param(
    [string]$DataRoot = 'data/h2o',
    [ValidateSet(1,2,3,4)][int[]]$Subjects = @(1,2,3,4),
    [string]$OutputParent = '',
    [string]$SeedRoot = 'data/h2o/results_all_subjects',
    [string]$SeedRunName = 'results_0',
    [ValidateSet('foundationpose_first_pose','oracle_gt_pose')][string]$InitMode = 'foundationpose_first_pose',
    [string]$Take = '',
    [ValidateRange(0,2147483647)][int]$MaxTakes = 0,
    [ValidateRange(0,2147483647)][int]$MaxFrames = 0,
    [ValidateRange(1,64)][int]$Threads = 4,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$data = (Resolve-Path -LiteralPath $DataRoot).Path
$destination = if ($OutputParent) { $OutputParent } else { Join-Path $data 'm3t_all_subjects' }
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$mesh = Join-Path $data 'object.zip'
if (-not (Test-Path -LiteralPath $mesh)) { throw "Missing H2O mesh: $mesh" }
$jobs = @()
foreach ($number in ($Subjects | Select-Object -Unique)) {
    $subject = "subject$number"
    $root = Join-Path $data $subject
    $seed = Join-Path (Join-Path $SeedRoot $subject) $SeedRunName
    if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw "Missing $root" }
    if ($InitMode -eq 'foundationpose_first_pose' -and -not (Test-Path -LiteralPath (Join-Path $seed 'summary.json'))) {
        throw "Missing cached seed run: $seed"
    }
    $jobs += [pscustomobject]@{Subject=$subject; Root=$root; Seed=$seed; Destination=(Join-Path $destination $subject)}
    Write-Host "$subject -> $destination\$subject\results_N; independent single-camera M3T region+depth"
}
Write-Host "Initialization=$InitMode; first frame only. MaxFrames=$MaxFrames; MaxTakes=$MaxTakes (0=all)."
if ($DryRun) { return }
if ($env:CONDA_DEFAULT_ENV -ne 'mr_nav') {
    $activation = & 'conda.exe' shell.powershell activate mr_nav
    if ($LASTEXITCODE -ne 0) { throw 'Conda activation failed' }
    Invoke-Expression ($activation -join "`n")
}
if ($env:CONDA_DEFAULT_ENV -ne 'mr_nav') { throw 'mr_nav must be active' }
$failed = @()
Push-Location $repo
try {
    foreach ($job in $jobs) {
        $options = @('-u','-m','tools.run_h2o_m3t','--subject-root',$job.Root,'--mesh-source',$mesh,
            '--seed-run-dir',$job.Seed,'--init-mode',$InitMode,'--output-parent',$job.Destination,
            '--max-takes',"$MaxTakes",'--max-frames',"$MaxFrames",'--threads',"$Threads")
        if ($Take) { $options += @('--take',$Take) }
        & python @options
        if ($LASTEXITCODE -ne 0) { $failed += $job.Subject }
    }
} finally { Pop-Location }
if ($failed.Count) { throw "M3T failed subjects: $($failed -join ', '); inspect reports under $destination" }
Write-Host "M3T tracking and default fusion completed under $destination"
