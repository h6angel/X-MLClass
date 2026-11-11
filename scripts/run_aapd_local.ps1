param([int]$TrialProcessId = 0)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = 'C:\Users\hyvonne\miniconda3\envs\xmlclass\python.exe'
$runDirectory = Join-Path $projectRoot 'outputs\aapd_pipeline'
New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null
$statusFile = Join-Path $runDirectory 'runner_status.json'
$logFile = Join-Path $runDirectory 'run.log'
function Write-RunStatus([string]$state, [string]$detail) {
    @{ state = $state; detail = $detail; updated_at = [DateTimeOffset]::Now.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding UTF8
}
try {
    if ($TrialProcessId -gt 0) {
        Write-RunStatus 'waiting_for_trial' "Waiting for trial process $TrialProcessId"
        Wait-Process -Id $TrialProcessId -ErrorAction SilentlyContinue
        if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'outputs\aapd_pipeline_trial\metrics.json'))) {
            throw 'Trial did not produce metrics. Inspect trial error before starting the full run.'
        }
    }
    $env:HF_HUB_DOWNLOAD_TIMEOUT = '600'
    $env:HF_HUB_ETAG_TIMEOUT = '60'
    $env:PYTHONUNBUFFERED = '1'
    Write-RunStatus 'running' 'Initial NLI, ten improvement rounds, then test and local evaluation'
    $process = Start-Process -FilePath $pythonPath -ArgumentList @('-u', 'scripts/aapd_pipeline.py', '--stage', 'all') `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $logFile -RedirectStandardError (Join-Path $runDirectory 'run.stderr.log')
    if ($process.ExitCode -ne 0) { throw "Pipeline exited with code $($process.ExitCode). See run.stderr.log." }
    Write-RunStatus 'complete' 'Finished; results are in metrics.json'
} catch {
    Write-RunStatus 'failed' $_.Exception.Message
    exit 1
}
