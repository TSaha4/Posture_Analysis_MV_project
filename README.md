# Posture Analysis MV (Machine Vision)

## What is this project?
Posture Analysis MV is an intelligent machine-vision-based application designed to monitor user posture and gaze in real-time using a standard webcam. It features a React-based frontend and a Python Flask backend powered by OpenCV.

## What does the project do?
This system acts as a real-time monitor for structured sessions such as online exams or prolonged work periods. It tracks the user's face position, head angle, eye direction, and overall movement in relation to a captured "baseline" profile. 
Based on these metrics, the system calculates a dynamic "Trust Score" or "Posture Score", penalizing the user for excessive movement, bad posture, or continuously looking away from the screen.

Key Features:
- **Baseline Calibration:** Captures an ideal reference snapshot of the user's face and alignment before a session begins.
- **Real-Time Video Feed:** Provides visual feedback via a bounding box and alignment overlays directly on the camera feed.
- **Exam & Question Modes:** Can track timed sessions (e.g., active questions), giving a summarized report of posture drift and gaze infractions once the timer ends.
- **Intelligent Edge/Corner Detection:** Uses OpenCV image processing to calculate movements and orientation accurately.

## Workflow
1. **Calibration:** Upon loading the app, the user aligns their face within a visual on-screen ellipse to establish a stable reference (baseline) for head angle and position.
2. **Session Start:** The user can initiate a monitoring session (like "Start Exam" or "Start Question").
3. **Active Monitoring:** The system constantly polls video frames, checking the current posture against the calibrated baseline.
4. **Correction Feedback:** The user's status updates dynamically (e.g., "Look at the screen", "Reduce movement").
5. **Session Wrap-Up:** Once the session ends, a comprehensive score (Excellent, Good, Average, Poor) and a breakdown of errors (time penalties, posture infractions) are presented to the user.

## Steps to Run

### 1. Backend Setup (Flask + OpenCV)
Open a terminal and navigate to the project root, then set up the server:
```bash
# Navigate to backend
cd backend

# Create & activate a virtual environment (if not using the existing 'myenv')
python3 -m venv myenv
source myenv/bin/activate  # On Windows use: myenv\Scripts\activate

# Install the dependencies
pip install -r requirements.txt

# Run the Flask backend
python app.py
```
*The backend will run on `http://127.0.0.1:8000`.*

### 2. Frontend Setup (React + Vite)
Open a separate, second terminal window and do the following:
```bash
# Navigate to frontend
cd frontend

# Install Node modules
npm install

# Start the Vite development server
npm run dev
```
*The frontend development server will usually spin up on `http://localhost:5173`. Open this URL in your browser to interact with the application.*
