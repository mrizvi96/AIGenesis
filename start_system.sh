#!/bin/bash
# AI Insurance Claims Processing System Launch Script

echo "Starting AI Insurance Claims Processing System..."

# Start backend server
echo "Starting FastAPI backend server..."
cd backend
python main.py &
BACKEND_PID=$!

echo "Backend server started with PID: $BACKEND_PID"
echo "API available at: http://localhost:8000"

# Wait for backend to start
sleep 10

# Start frontend
echo "Starting Streamlit frontend..."
cd ../frontend
streamlit run ui.py &
FRONTEND_PID=$!

echo "Frontend started with PID: $FRONTEND_PID"
echo "Frontend available at: http://localhost:8501"

echo ""
echo "=== SYSTEM READY ==="
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:8501"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "To stop the system:"
echo "kill $BACKEND_PID $FRONTEND_PID"
