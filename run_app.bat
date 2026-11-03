@echo off
echo ============================================================
echo Starting Restaurant Revenue Prediction Web Application...
echo ============================================================
echo.
"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" --browser.gatherUsageStats false
pause
