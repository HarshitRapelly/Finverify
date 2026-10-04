Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " Starting FinVerify AI - Hallucination-Resistant RAG" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Cyan

Set-Location $PSScriptRoot
& ".\.venv\Scripts\streamlit.exe" run app.py
