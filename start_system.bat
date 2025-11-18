@echo off
echo Starting AI Insurance Claims Processing System...

echo Starting FastAPI backend server...
cd backend
start /B python main.py

echo Waiting for backend to start...
timeout /t 10 /nobreak

echo Starting Streamlit frontend...
cd ..\frontend
start /B streamlit run ui.py

echo.
echo === SYSTEM READY ===
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:8501
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to continue...
pause
