param([int]$StopUpdate = 600, [int]$InterruptAfterEpisodes = 0, [switch]$ClearStop)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $projectRoot '../../..')).Path
$pythonExe = Join-Path $repoRoot 'venv/Scripts/python.exe'
$opsDir = Join-Path $projectRoot 'logs/P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224/continuation_u600'
New-Item -ItemType Directory -Force -Path $opsDir | Out-Null
$stopPath = Join-Path $opsDir 'STOP_REQUESTED'
if ($ClearStop -and (Test-Path -LiteralPath $stopPath)) {
    Move-Item -LiteralPath $stopPath -Destination (Join-Path $opsDir ('STOP_ACK_' + (Get-Date -Format 'yyyyMMdd_HHmmss_ffff')))
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_ffff'
$launchArgs = @('-u', '-m', 'utils.r8c1_training_session', 'run', '--stop-update', "$StopUpdate", '--interrupt-after-episodes', "$InterruptAfterEpisodes")
$process = Start-Process -FilePath $pythonExe -ArgumentList $launchArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $opsDir "supervisor_$stamp.out.log") -RedirectStandardError (Join-Path $opsDir "supervisor_$stamp.err.log")
Write-Output "Supervisor launched PID=$($process.Id). Check continuation_u600/session.json and latest supervisor log before assuming training started."
