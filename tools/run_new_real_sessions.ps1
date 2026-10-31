[CmdletBinding()]
param(
    [string[]]$Sessions = @('stable-0','stable-1','stable-2','Test-0','Test-1','Test-2','Test-3','Test-4'),
    [string]$DataRoot = 'data/laboratory',
    [string]$CalibrationDir = 'artifacts/new_real_calibration/results_1',
    [string]$ReferenceDir = 'artifacts/new_real_reference/results_1',
    [string]$OutputParent = 'artifacts/new_real_full',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ($env:CONDA_DEFAULT_ENV -ne 'mr_nav') {
    throw 'Activate conda mr_nav before starting this script.'
}
Push-Location $repoRoot
try {
    Write-Host 'Default batch: eight prepared sessions. Test-5 is deferred (unresolved OTS clock and interrupted frontend smoke check).'
    $referenceManifest = Get-Content -LiteralPath (Join-Path $ReferenceDir 'manifest.json') -Raw | ConvertFrom-Json
    foreach ($name in $Sessions) {
        if (-not ($referenceManifest.sessions.PSObject.Properties.Name -contains $name)) {
            throw "No reference-preparation record for $name"
        }
        $inputArgs = @('-m','tools.preflight_real_ots','--session',(Join-Path $DataRoot $name),
                       '--calibration-dir',$CalibrationDir,'--reference-dir',$ReferenceDir)
        if (-not $referenceManifest.sessions.$name.reference_ready) {
            $inputArgs += '--allow-missing-reference'
            Write-Warning "$name : OTS reference unresolved. Tracking/fusion will run, but no OTS accuracy will be reported."
        }
        & python @inputArgs
        if ($LASTEXITCODE -ne 0) { throw "Preflight failed: $name" }
    }
    if ($DryRun) { Write-Host 'All input checks passed; no tracking started.'; return }
    $failed = @()
    foreach ($name in $Sessions) {
        $runArgs = @('-u','-m','tools.run_real_ots_smoke','--session',(Join-Path $DataRoot $name),
                     '--full-sequence','--calibration-dir',$CalibrationDir,'--reference-dir',$ReferenceDir,
                     '--output-parent',$OutputParent)
        if (-not $referenceManifest.sessions.$name.reference_ready) { $runArgs += '--skip-reference-evaluation' }
        & python @runArgs
        if ($LASTEXITCODE -ne 0) { $failed += $name; Write-Warning "Failed: $name. Partial outputs retained." }
    }
    if ($failed.Count) { throw "Failed sessions: $($failed -join ', ')" }
    Write-Host "Completed. Outputs: $OutputParent/<session>/results_#. Check evaluation_status in each manifest."
} finally { Pop-Location }
