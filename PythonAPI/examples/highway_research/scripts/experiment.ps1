param(
    [ValidateSet('Preflight','TrainMain','TrainAblations','AuditDev','AuditSelected','AuditTest','AuditRobust','AuditStochastic','Report','TensorBoard')]
    [string]$Action='Preflight',
    [string]$StudyRoot='highwaytest/runs/research1_5seeds_20261111_v1',
    [int]$Jobs=4,
    [int]$Port=6012
)
$ErrorActionPreference='Stop'
$examplesRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$repositoryRoot=Split-Path -Parent (Split-Path -Parent $examplesRoot)
$pythonBinary=Join-Path $repositoryRoot 'venv/Scripts/python.exe'
Push-Location $examplesRoot
try {
    switch($Action) {
        'Preflight' { & $pythonBinary -m highway_research batch --root $StudyRoot --arms main beta mechanism reward --dry-run }
        'TrainMain' { & $pythonBinary -X utf8 -u -m highway_research batch --root $StudyRoot --arms main --jobs $Jobs }
        'TrainAblations' { & $pythonBinary -X utf8 -u -m highway_research batch --root $StudyRoot --arms beta mechanism reward --jobs $Jobs }
        'AuditDev' { & $pythonBinary -m highway_research audit-study --root $StudyRoot --arms main beta mechanism reward --checkpoint latest --split development --cases 40 }
        'AuditSelected' { & $pythonBinary -m highway_research audit-study --root $StudyRoot --arms main --checkpoint selected --split development --cases 40 }
        'AuditTest' { & $pythonBinary -m highway_research audit-study --root $StudyRoot --arms main --checkpoint latest --split test --cases 100 }
        'AuditRobust' { & $pythonBinary -m highway_research audit-study --root $StudyRoot --arms main --labels PPO Mut-PPO --checkpoint latest --conditions nominal idm_short_gap idm_long_gap cautious_lane position_shift --split development --cases 40 }
        'AuditStochastic' { & $pythonBinary -m highway_research audit-study --root $StudyRoot --arms main --labels PPO Mut-PPO --checkpoint latest --execution stochastic --action-seeds 0 101 202 --split development --cases 40 }
        'Report' { & $pythonBinary -m highway_research report --root $StudyRoot }
        'TensorBoard' {
            $absoluteStudy=(Resolve-Path -LiteralPath $StudyRoot).Path
            $logs=Join-Path $absoluteStudy 'logs';New-Item -ItemType Directory -Path $logs -Force | Out-Null
            $tb=Start-Process -FilePath $pythonBinary -ArgumentList @('-m','tensorboard.main','--logdir',"$absoluteStudy\train",'--host','127.0.0.1','--port',"$Port") -WindowStyle Hidden -RedirectStandardOutput "$logs\tensorboard_stdout.log" -RedirectStandardError "$logs\tensorboard_stderr.log" -PassThru
            Write-Output "TensorBoard http://127.0.0.1:$Port PID=$($tb.Id)"
        }
    }
    if($Action -ne 'TensorBoard' -and $LASTEXITCODE -ne 0){throw "Command failed with exit code $LASTEXITCODE"}
} finally { Pop-Location }
