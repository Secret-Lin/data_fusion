[CmdletBinding()]
param(
    [string]$RunRoot = 'data/h2o/megapose_all_subjects',
    [string]$RunName = 'results_0',
    [ValidateSet(1,2,3,4)][int[]]$Subjects = @(1,2,3,4),
    [string]$OutputParent = 'data/h2o/megapose_ablation_all_subjects',
    [string]$Takes = '',
    [int]$MaxFrames = 0,
    [int]$MaxTakes = 0,
    [string]$Variants = '',
    [switch]$LeaveOneOut = $true,
    [switch]$DryRun
)
$options = @{
    RunRoot=$RunRoot; RunName=$RunName; Subjects=$Subjects; OutputParent=$OutputParent;
    Takes=$Takes; MaxFrames=$MaxFrames; MaxTakes=$MaxTakes; Variants=$Variants;
    LeaveOneOut=[bool]$LeaveOneOut; DryRun=[bool]$DryRun
}
# Existing cached-observation path; MegaPose source names and detector are preserved.
& (Join-Path $PSScriptRoot 'run_h2o_all_subjects_ablation.ps1') @options
