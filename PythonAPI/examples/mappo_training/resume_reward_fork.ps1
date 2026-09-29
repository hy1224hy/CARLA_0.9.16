param([int]$StopUpdate = 456)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $projectRoot '../../..')).Path
$pythonExe = Join-Path $repoRoot 'venv/Scripts/python.exe'
$runDir = Join-Path $projectRoot 'logs/R8C1_RoutePaceV1_u356_to406_s42_20260927'
if (-not (Test-Path -LiteralPath (Join-Path $runDir 'checkpoint_manifest.json'))) {
    throw 'Prepared reward fork is missing. Do not launch the historical u600 experiment.'
}
$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(training\.reward_fork|training\.train_|evaluation\.evaluate_|evaluation\.diagnose_|utils\.r8c1_training_session)'
}
if ($active) {
    throw ('Another trainer/evaluator may own CARLA. Inspect PIDs: ' + (($active.ProcessId) -join ','))
}
$opsDir = Join-Path $runDir 'session'
New-Item -ItemType Directory -Force -Path $opsDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_ffff'
$env:PYTHONIOENCODING = 'utf-8'
$launchArgs = @('-u', '-m', 'training.reward_fork', '--run-dir', ('"{0}"' -f $runDir), '--stop-update', "$StopUpdate")
$process = Start-Process -FilePath $pythonExe -ArgumentList $launchArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $opsDir "train_$stamp.out.log") -RedirectStandardError (Join-Path $opsDir "train_$stamp.err.log")
Start-Process -FilePath $pythonExe -ArgumentList @('-u', '-m', 'utils.follow_reward_fork', '--trainer-pid', "$($process.Id)", '--run-dir', ('"{0}"' -f $runDir)) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $opsDir "camera_$stamp.out.log") -RedirectStandardError (Join-Path $opsDir "camera_$stamp.err.log")
[pscustomobject]@{pid=$process.Id; launched_at=(Get-Date).ToString('o'); stop_update=$StopUpdate; run_dir=$runDir; stdout="train_$stamp.out.log"; stderr="train_$stamp.err.log"} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $opsDir 'launch.json') -Encoding utf8
Write-Output "Launched PID=$($process.Id). Inspect logs and checkpoint growth to confirm training."
