param(
    [string]$Python = '',
    [int]$Steps = 200000,
    [int[]]$Seeds = @(42,43,44),
    [int]$Jobs = 1,
    [string]$Study = 'highwaytest/runs/straight_v1'
)
$ErrorActionPreference = 'Stop'
$examplesRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $examplesRoot
if (-not $Python) { $Python = Join-Path $examplesRoot '../../venv/Scripts/python.exe' }
function Invoke-Straight {
    & $Python -X utf8 -u -m highwaytest.straight @args
    if ($LASTEXITCODE -ne 0) { throw 'Straight experiment command failed; inspect output before continuing.' }
}

# 1. Main matrix (18 runs) + per-seed base initialization.
$configRoot = 'highwaytest/configs/straight_v1'
if (-not (Test-Path -LiteralPath $configRoot)) {
    Invoke-Straight plan --output $configRoot --steps $Steps --seeds @Seeds
}
Invoke-Straight check --configs $configRoot --steps $Steps --seeds @Seeds
foreach ($trainingSeed in $Seeds) {
    $baseRoot = Join-Path $Study ('base_s'+$trainingSeed)
    if (-not (Test-Path -LiteralPath (Join-Path $baseRoot 'driver.pt'))) {
        Invoke-Straight bootstrap --seed $trainingSeed --output $baseRoot
    }
    if (-not (Test-Path -LiteralPath (Join-Path $baseRoot 'repertoire.pt'))) {
        Invoke-Straight bootstrap --seed $trainingSeed --repertoire --output $baseRoot
    }
}

# 2. Training blocks: main, beta, reward-ablation (reused runs are not retrained).
Invoke-Straight batch --configs $configRoot --output (Join-Path $Study 'train/main') --jobs $Jobs --initialize-from-dir $Study
Invoke-Straight batch --configs 'highwaytest/configs/straight_beta_v2' --output (Join-Path $Study 'train/beta') --jobs $Jobs --initialize-from-dir $Study
Invoke-Straight batch --configs 'highwaytest/configs/straight_reward_v2' --output (Join-Path $Study 'train/reward') --jobs $Jobs --initialize-from-dir $Study

# 3. Frozen final test: 100 paired cases per run, trajectory + GIF export.
$runDirs = Get-ChildItem -LiteralPath (Join-Path $Study 'train') -Directory | ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory }
foreach ($runDir in $runDirs) {
    $selected = Join-Path $runDir.FullName 'selected.pt'
    $testDir = Join-Path $runDir.FullName 'test'
    if ((Test-Path -LiteralPath $selected) -and -not (Test-Path -LiteralPath (Join-Path $testDir 'evaluation.json'))) {
        Invoke-Straight evaluate --checkpoint $selected --split test --cases 100 --gif --output $testDir
    }
}

# 4. Aggregate comparison (all 30 runs), strategy pool and consistency matrix (main 18 only).
Invoke-Straight report --runs (Join-Path $Study 'train/main') (Join-Path $Study 'train/beta') (Join-Path $Study 'train/reward') --output (Join-Path $Study 'comparison')
Invoke-Straight pool --runs (Join-Path $Study 'train/main') --output (Join-Path $Study 'pool.json')
Invoke-Straight matrix --runs (Join-Path $Study 'train/main') --output (Join-Path $Study 'matrix.json')
Write-Output ('Pipeline finished. Study: '+$Study+' ; comparison figures, pool.json, matrix.json written.')
