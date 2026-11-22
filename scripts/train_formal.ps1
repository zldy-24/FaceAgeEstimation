param(
    [string]$Output = 'artifacts/phase4',
    [string]$Protocol = 'docs/phase4/protocol.json'
)

$ErrorActionPreference = 'Stop'
$python = '.\.venv\Scripts\python.exe'
$locked = Get-Content -Raw -LiteralPath $Protocol | ConvertFrom-Json
$training = $locked.training
$datasetId = & $python -c "from age_estimation.locking import verify_dataset_lock; print(verify_dataset_lock('data/processed/utkface'))"
if ($LASTEXITCODE -ne 0 -or $datasetId -ne $locked.dataset_lock_id) {
    throw 'The prepared data does not match the formal experiment protocol.'
}

if (Test-Path -LiteralPath $Output) {
    throw "Choose a new experiment directory: $Output"
}
New-Item -ItemType Directory -Path $Output | Out-Null
Copy-Item -LiteralPath $Protocol -Destination (Join-Path $Output 'protocol.json')

$trainArgs = @(
    '--device', $training.device, '--seed', $training.seed, '--batch-size', $training.batch_size,
    '--workers', $training.workers, '--optimizer', $training.optimizer,
    '--lr', [string]$training.lr, '--weight-decay', [string]$training.weight_decay
)
$warmup = Join-Path $Output 'warmup/best.pt'
foreach ($strategy in 'cnn','warmup','frozen','finetune') {
    $branchArgs = @()
    if ($strategy -in 'frozen','finetune') {
        $branchArgs += @('--init-checkpoint', $warmup)
    }
    if ($strategy -eq 'finetune') {
        $branchArgs += @('--backbone-lr', [string]$training.finetune_backbone_lr)
    }
    & $python -m scripts.train --strategy $strategy --epochs $training.epochs.$strategy @trainArgs @branchArgs --output (Join-Path $Output $strategy)
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $strategy, exit code $LASTEXITCODE." }
}
