@echo off
echo Starting Fake News Detection App...
echo.

echo Installing/updating Python dependencies...
pip install -r requirements.txt
echo.

echo Starting backend server...
start "Backend Server" cmd /k "python app.py"
echo Backend server started on http://localhost:5000
echo.

echo Waiting 5 seconds for backend to initialize...
timeout /t 5 /nobreak > nul

echo Moving to frontend directory...
cd frontend
echo.

echo Installing/updating Node.js dependencies...
npm install
echo.

echo Starting frontend development server...
start "Frontend Server" cmd /k "npm run dev"
echo Frontend server will start on http://localhost:3000
echo.

echo Both servers are starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit...
pause > nul
