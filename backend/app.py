import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import sys
import time
import cv2
import numpy as np
import atexit
import threading
from flask import Flask, Response, jsonify
from flask_cors import CORS

from backend.model.camera_rl_utils import (
    detect_face_info,
    build_state,
    is_bad_state,
    choose_action,
    reward_for_transition,
    is_face_in_circle,
    Q,
    actions,
    alpha,
    gamma,
    min_epsilon,
    epsilon_decay,
    load_q_table,
    save_q_table
)

app = Flask(__name__)
CORS(app)

# ---------------------------
# GLOBAL VARIABLES
# ---------------------------
cap = None
baseline = None
epsilon = 0.25
camera_thread = None
camera_active = False

current_frame = None

current_rl_data = {
    "state": None,
    "action": None,
    "reward": 0,
    "is_bad": False,
    "identified_by": "Initializing..."
}

data_lock = threading.Lock()

# ---------------------------
# PROCTOR EXAM TRACKING
# ---------------------------
exam_active = False
posture_stats = {"good": 0, "bad": 0}

# ---------------------------
# CAMERA THREAD
# ---------------------------
def camera_background_task():
    global cap, baseline, epsilon
    global current_frame, current_rl_data
    global exam_active, posture_stats, camera_active

    if cap is None:
        if sys.platform.startswith('darwin'):
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        elif sys.platform.startswith('win'):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam. Check camera permissions and device index.')

    stable_count = 0
    REQUIRED_STABLE_FRAMES = 15
    missing_face_count = 0
    MAX_MISSING_FRAMES = 60

    last_state = None
    frame_counter = 0

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 4
        axes = (int(radius * 0.8), int(radius * 1.2))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        info = detect_face_info(gray)

        rl_state = None
        action = None
        reward = 0
        ai_is_bad_posture = False
        identified_by = "Waiting for alignment..."

        # Draw default oval
        cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 255, 255), 2)

        # ---------------------------
        # NO FACE
        # ---------------------------
        if info is None:
            missing_face_count += 1

            if missing_face_count > MAX_MISSING_FRAMES and baseline is not None:
                baseline = None
                stable_count = 0
                print("🔄 User left → baseline reset")

            cv2.putText(frame, "No face detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            identified_by = "No Face"

        else:
            missing_face_count = 0

            is_inside, ratio = is_face_in_circle(info, center, radius)

            # ---------------------------
            # AUTO CALIBRATION
            # ---------------------------
            if baseline is None:
                identified_by = "Auto-Calibrating..."

                if ratio > 0.65:
                    stable_count += 1

                    cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 255, 0), 2)

                    if stable_count >= REQUIRED_STABLE_FRAMES:
                        baseline = info.copy()
                        print("✅ Calibration Done")

                else:
                    stable_count = max(0, stable_count - 1)
                    cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 165, 255), 2)

            # ---------------------------
            # RL TRACKING
            # ---------------------------
            else:
                circle_color = (0, 255, 0) if is_inside else (0, 165, 255)
                cv2.ellipse(frame, center, axes, 0, 0, 360, circle_color, 2)

                rl_state = build_state(info, baseline)

                if rl_state:
                    # Detect posture
                    ai_is_bad_posture = is_bad_state(rl_state)

                    # Identify source
                    if rl_state in Q and np.any(Q[rl_state]):
                        identified_by = "Q-Table"
                    else:
                        identified_by = "Heuristic"

                    # RL action
                    action_index = choose_action(rl_state, epsilon)
                    action = actions[action_index]

                    # ---------------------------
                    # ✅ FIXED RL TRANSITION
                    # ---------------------------
                    if last_state is not None:
                        reward = reward_for_transition(last_state, rl_state)

                        next_max = max(Q[rl_state]) if rl_state in Q else 0

                        Q[last_state][action_index] += alpha * (
                            reward + gamma * next_max - Q[last_state][action_index]
                        )

                    last_state = rl_state

                    epsilon = max(min_epsilon, epsilon * epsilon_decay)

                    # ---------------------------
                    # PROCTOR TRACKING
                    # ---------------------------
                    if exam_active:
                        if ai_is_bad_posture:
                            posture_stats["bad"] += 1
                        else:
                            posture_stats["good"] += 1

                # UI
                cv2.putText(frame, f"State: {rl_state}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.putText(frame, f"Action: {action}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if ai_is_bad_posture:
                    cv2.putText(frame, "FIX POSTURE!", (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Save Q periodically
        frame_counter += 1
        if frame_counter >= 300:
            save_q_table()
            frame_counter = 0

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)

        # Share safely
        with data_lock:
            current_frame = buffer.tobytes()
            current_rl_data = {
                "state": rl_state,
                "action": action,
                "reward": reward,
                "is_bad": bool(ai_is_bad_posture),
                "identified_by": identified_by
            }


# ---------------------------
# INIT
# ---------------------------
print("🧠 Loading Q-table...")
load_q_table()

# Camera thread starts on demand

# ---------------------------
# API
# ---------------------------
@app.route("/")
def home():
    return "✅ Backend Running"


@app.route("/state")
def get_state():
    with data_lock:
        return jsonify(current_rl_data)


@app.route("/start_exam")
def start_exam():
    global exam_active, posture_stats, camera_thread, camera_active
    exam_active = True
    posture_stats = {"good": 0, "bad": 0}
    
    if not camera_active:
        camera_active = True
        camera_thread = threading.Thread(target=camera_background_task, daemon=True)
        camera_thread.start()
        print("📹 Camera started")
    
    print("🎯 Exam started")
    return jsonify({"status": "started"})


@app.route("/end_exam")
def end_exam():
    global exam_active, posture_stats

    exam_active = False

    total = posture_stats["good"] + posture_stats["bad"]
    score = int((posture_stats["good"] / total) * 100) if total > 0 else 0

    if score > 80:
        label = "Excellent"
    elif score > 60:
        label = "Good"
    elif score > 40:
        label = "Average"
    else:
        label = "Poor"

    print(f"📊 Final Score: {score}%")

    return jsonify({
        "score": score,
        "label": label,
        "good_frames": posture_stats["good"],
        "bad_frames": posture_stats["bad"]
    })


@app.route("/stop_session")
def stop_session():
    global camera_active, cap, camera_thread
    camera_active = False
    
    if cap:
        cap.release()
        cap = None
        print("📷 Camera released")
    
    save_q_table()
    print("💾 Q-table saved")
    
    return jsonify({"status": "stopped"})


@app.route("/calibrate")
def calibrate():
    global baseline
    baseline = None  # Reset baseline to trigger calibration
    return jsonify({"status": "calibration_reset"})


@app.route("/video_feed")
def video_feed():
    global camera_active, camera_thread
    if not camera_active:
        camera_active = True
        camera_thread = threading.Thread(target=camera_background_task, daemon=True)
        camera_thread.start()
        print("📹 Camera started for video feed")
    
    def generate():
        while camera_active:
            with data_lock:
                frame = current_frame

            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       frame + b'\r\n')

            time.sleep(0.03)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------------------
# CLEANUP
# ---------------------------
@atexit.register
def cleanup():
    global cap
    if cap:
        cap.release()

    save_q_table()
    print("💾 Saved Q-table & exiting")


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(port=8000, debug=True, use_reloader=False)