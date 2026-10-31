[CmdletBinding()]
param(
    [string]$DataRoot = 'data/h2o',
    [string]$RunName = 'results_0',
    [string]$OutputParent = 'data/h2o/fusion_baselines',
    [ValidateSet('foundationpose','m3t','megapose')]
    [string[]]$Frontends = @('foundationpose','m3t','megapose'),
    [ValidateSet(1,2,3,4)][int[]]$Subjects = @(1,2,3,4),
    [ValidateRange(1,16)][int]$Workers = 4,
    [string]$Takes = '',
    [ValidateRange(0,2147483647)][int]$MaxTakes = 0,
    [ValidateRange(0,2147483647)][int]$MaxFrames = 0,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^results_[0-9]+$') { throw 'RunName must be results_N' }
if ($env:CONDA_DEFAULT_ENV -ne 'mr_nav') {
    $condaCommand = Get-Command conda.exe -ErrorAction SilentlyContinue
    $condaExe = if ($condaCommand) { $condaCommand.Source } else { 'conda.exe' }
    if (-not (Test-Path -LiteralPath $condaExe)) { throw 'Activate conda mr_nav first' }
    $activation = & $condaExe shell.powershell activate mr_nav
    if ($LASTEXITCODE -ne 0) { throw 'mr_nav activation failed' }
    Invoke-Expression ($activation -join "`n")
}
if ($env:CONDA_DEFAULT_ENV -ne 'mr_nav') { throw 'mr_nav is required' }
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonArgs = @('-u','-m','tools.run_h2o_fusion_baselines','--all-subjects',
    '--data-root',$DataRoot,'--run-name',$RunName,'--output-parent',$OutputParent,
    '--workers',$Workers.ToString(),'--max-takes',$MaxTakes.ToString(),'--max-frames',$MaxFrames.ToString(),
    '--frontends') + $Frontends + @('--subjects') + @($Subjects | ForEach-Object { $_.ToString() })
if ($Takes) { $pythonArgs += @('--takes',$Takes) }
if ($DryRun) { $pythonArgs += '--dry-run' }
# Independent takes are CPU jobs; do not multiply BLAS threads by worker count.
$savedThreads = @{}
foreach ($name in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')) {
    $savedThreads[$name] = [Environment]::GetEnvironmentVariable($name,'Process')
    [Environment]::SetEnvironmentVariable($name,'1','Process')
}
Push-Location -LiteralPath $repoRoot
try {
    & python @pythonArgs
    if ($LASTEXITCODE -ne 0) { throw 'Baseline comparison incomplete; inspect summary.json failed_takes/failed_methods. Existing results were not overwritten.' }
} finally {
    Pop-Location
    foreach ($name in $savedThreads.Keys) { [Environment]::SetEnvironmentVariable($name,$savedThreads[$name],'Process') }
}
