[CmdletBinding()]
param(
    [string]$Session = 'data/laboratory/Test-0',
    [double]$Start = 0,
    [double]$Warmup = 1,
    [double]$Duration = 3,
    [double]$ReceiverHz = 30,
    [double]$MaxPreloadGiB = 2
)
$ErrorActionPreference = 'Stop'
if ($env:CONDA_DEFAULT_ENV -ne 'mr_nav') { throw 'Activate conda mr_nav first.' }
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Push-Location $repoRoot
try {
    & python -u -m tools.benchmark_recorded_realtime --session $Session --start $Start --warmup $Warmup --duration $Duration --receiver-hz $ReceiverHz --max-preload-gib $MaxPreloadGiB
    if ($LASTEXITCODE -ne 0) { throw 'Paced replay failed; inspect its manifest and logs.' }
} finally { Pop-Location }
