[CmdletBinding()]
param([int]$Jobs = 4)
$ErrorActionPreference = 'Stop'
if ($env:CONDA_DEFAULT_ENV -ne 'mr_nav') {
    $activation = & 'conda.exe' shell.powershell activate mr_nav
    if ($LASTEXITCODE -ne 0) { throw 'Conda activation failed' }
    Invoke-Expression ($activation -join "`n")
}
if ($env:CONDA_DEFAULT_ENV -ne 'mr_nav') { throw 'mr_nav must be active' }
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $repo
try {
    & python -m tools.build_m3t --jobs $Jobs
    if ($LASTEXITCODE -ne 0) { throw 'M3T native build failed' }
} finally { Pop-Location }
