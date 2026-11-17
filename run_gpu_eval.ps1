# ============================================================
# run_gpu_eval.ps1
# Launches llama3.2:latest on RTX 3050 GPU (port 11435)
# then resumes dsel ablation evaluation from checkpoint.
# ============================================================

$BASE = "C:\Users\Acer\AppData\Local\Programs\Ollama\lib\ollama"
$BLOB = "C:\Users\Acer\.ollama\models\blobs\sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff"
$PORT = 11435
$PROJ = "C:\Users\Acer\OneDrive\Desktop\multiagents\disaster_response_rag"

Write-Host "=== Starting llama3.2:latest on GPU (RTX 3050) at port $PORT ===" -ForegroundColor Green

# Kill any old llama-server on port 11435
Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force

# Start the GPU server in background
$env:PATH = "$BASE;$env:PATH"
$server = Start-Process -FilePath "$BASE\llama-server.exe" `
    -ArgumentList "-m `"$BLOB`" -ngl 99 --port $PORT -c 4096 -np 1 --no-webui" `
    -WorkingDirectory $BASE -PassThru -NoNewWindow -RedirectStandardOutput "$PROJ\logs\llama_server_gpu.log"

Write-Host "llama-server PID: $($server.Id)"
Write-Host "Waiting 5s for model to load..."
Start-Sleep -Seconds 5

# Health check
try {
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$PORT/health" -Method GET -TimeoutSec 5
    Write-Host "Server status: $($resp.status)" -ForegroundColor Green
} catch {
    # /health may not exist — try a quick completion
    Write-Host "Health endpoint not available, trying inference..." -ForegroundColor Yellow
}

# Quick GPU speed test
$body = '{"messages":[{"role":"user","content":"Say: GPU ONLINE"}],"max_tokens":5,"temperature":0}' | ConvertFrom-Json | ConvertTo-Json
$t0 = Get-Date
try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$PORT/v1/chat/completions" `
        -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10
    $elapsed = ((Get-Date) - $t0).TotalSeconds
    Write-Host "GPU speed test: $($r.choices[0].message.content) in ${elapsed}s" -ForegroundColor Cyan
} catch {
    Write-Host "Speed test error: $_" -ForegroundColor Red
}

# Run eval from checkpoint
Write-Host ""
Write-Host "=== Resuming DSEL Ablation Evaluation (from checkpoint) ===" -ForegroundColor Green
Set-Location $PROJ
python -u evaluation/eval_dsel_ablation.py

# When done, stop the GPU server
Write-Host ""
Write-Host "=== Stopping GPU server ===" -ForegroundColor Yellow
Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
Write-Host "Done! Check evaluation/dsel_ablation_raw.csv and dsel_ablation_results.json"
