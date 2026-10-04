@echo off
echo ====================================================
echo Starting FinVerify AI - Hallucination-Resistant RAG
echo ====================================================
cd /d "%~dp0"
call .\.venv\Scripts\activate
streamlit run app.py
pause
