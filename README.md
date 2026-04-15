Posture Analysis Studio is a real-time, machine-vision-based application designed to monitor user posture and gaze during structured sessions like online exams or prolonged work periods. Using a standard webcam, it continuously tracks the user's facial position, head angle, eye direction, and overall movement relative to a calibrated "baseline" profile. Based on this data, the system calculates a dynamic score that detects and penalizes excessive motion, slouching, or frequently looking away from the screen. With features like real-time visual feedback via bounding boxes, intelligent edge detection powered by OpenCV, and timed session reporting, it provides an effective way to improve ergonomic habits and ensure focus.

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
