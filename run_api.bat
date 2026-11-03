@echo off
echo ============================================================
echo Starting Restaurant Revenue Prediction REST API...
echo API Docs: http://127.0.0.1:8000/docs
echo ============================================================
"%~dp0.venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
pause
