# Posture Analysis Studio

A real-time machine-vision-based application to monitor user posture and gaze using a webcam.

## Steps to Run

### 1. Backend Setup
```bash
cd backend
python3 -m venv myenv
source myenv/bin/activate  # On Windows use: myenv\Scripts\activate
pip install -r requirements.txt
python app.py
```
*Runs on `http://127.0.0.1:8000`.*

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
*Runs on `http://localhost:5173`.*
