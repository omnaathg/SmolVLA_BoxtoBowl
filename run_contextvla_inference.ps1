# =============================================================================
# ContextVLA Card-Memory Inference — Windows PowerShell launcher
#
# Usage:
#   .\run_contextvla_inference.ps1                        # voice mode
#   .\run_contextvla_inference.ps1 -Card "ace of hearts"  # start with a specific card
#   .\run_contextvla_inference.ps1 -TextMode              # keyboard instead of mic
#   .\run_contextvla_inference.ps1 -Duration 120          # 2-minute episode
#
# Prerequisites:
#   pip install openai-whisper sounddevice
# =============================================================================

param(
    [string]$Card           = "",
    [string]$PolicyPath     = "omnaathg/contextvla_card_memory",
    [string]$DatasetRepoId  = "omnaathg/so101_card_memory",
    [string]$FollowerPort   = "COM8",
    [string]$Device         = "cuda",
    [int]$Duration          = 60,
    [string]$WhisperModel   = "base",
    [switch]$TextMode
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate virtual environment
$ActivateScript = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    & $ActivateScript
} else {
    Write-Warning "Virtual environment not found at $ActivateScript — using system Python."
}

Write-Host "=== ContextVLA Card-Memory Inference ==="
Write-Host "  Policy:   $PolicyPath"
Write-Host "  Device:   $Device"
Write-Host "  Duration: ${Duration}s"

$PythonArgs = @(
    "$ScriptDir\run_contextvla_card_inference.py",
    "--policy_path",     $PolicyPath,
    "--dataset_repo_id", $DatasetRepoId,
    "--follower_port",   $FollowerPort,
    "--device",          $Device,
    "--duration",        $Duration,
    "--whisper_model",   $WhisperModel
)

if ($Card) {
    $PythonArgs += "--initial_task"
    $PythonArgs += $Card
    Write-Host "  Card:     $Card"
} else {
    Write-Host "  Mode:     voice controlled"
}

if ($TextMode) {
    $PythonArgs += "--text_mode"
    Write-Host "  Input:    keyboard"
}

Write-Host ""

python @PythonArgs
