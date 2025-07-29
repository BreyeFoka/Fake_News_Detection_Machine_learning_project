#!/bin/bash

echo "Starting Fake News Detection App..."
echo

echo "Installing/updating Python dependencies..."
pip install -r requirements.txt
echo

echo "Starting backend server in background..."
python app.py &
BACKEND_PID=$!
echo "Backend server started on http://localhost:5000 (PID: $BACKEND_PID)"
echo

echo "Waiting 5 seconds for backend to initialize..."
sleep 5

echo "Moving to frontend directory..."
cd frontend
echo

echo "Installing/updating Node.js dependencies..."
npm install
echo

echo "Starting frontend development server..."
npm run dev &
FRONTEND_PID=$!
echo "Frontend server started on http://localhost:3000 (PID: $FRONTEND_PID)"
echo

echo "Both servers are running:"
echo "Backend: http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo
echo "Press Ctrl+C to stop both servers..."

# Function to cleanup background processes
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "Servers stopped."
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup INT TERM

# Wait for user interrupt
wait
