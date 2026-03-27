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

# ---------------------------
# IMPORTS
# ---------------------------
try:
    from backend.model.camera_rl_utils import *
except:
    from model.camera_rl_utils import *

app = Flask(__name__)
CORS(app)

# ---------------------------
# GLOBAL STATE
# ---------------------------
cap = None
camera_thread = None
camera_active = False
camera_lock = threading.Lock()
camera_start_event = threading.Event()

current_frame = None
camera_error = None
latest_face_info = None
calibration_snapshot = None
calibration_frozen = False
calibration_frozen_until = 0.0
CALIBRATION_FREEZE_SECONDS = 1.5

baseline = None
epsilon = 0.25

exam_active = False
question_active = False

posture_stats = {"good": 0, "bad": 0}
question_stats = {"good": 0, "bad": 0}
question_last_state = None
question_history = []

exam_start_time = None
question_start_time = None

data_lock = threading.Lock()

current_rl_data = {
    "mode": "idle",
    "suggestion": "Waiting...",
    "calibration_ready": False,
    "calibration_frozen": False,
    "calibration_freeze_remaining": 0.0,
    "calibration_snapshot": None,
    "face_inside_ratio": 0.0,
    "face_width": 0,
    "face_height": 0,
    "head_angle": 0.0,
    "eye_dir": 0.0,
    "eye_ratio": 0.0,
    "eye_distance": 0.0,
    "trust_score": 0,
    "is_cheating": False,
    "reward": 0,
    "action": None,
    "state": None,
}


def time_score_for_elapsed(elapsed_seconds):
    if elapsed_seconds < 10:
        return 0
    if elapsed_seconds < 20:
        return 20
    if elapsed_seconds < 30:
        return 35
    if elapsed_seconds < 45:
        return 50
    if elapsed_seconds < 60:
        return 65
    if elapsed_seconds < 90:
        return 80
    if elapsed_seconds < 120:
        return 90
    return 100


def label_for_score(score):
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Average"
    return "Poor"

# ---------------------------
# CAMERA HANDLING
# ---------------------------
def open_camera():
    attempts = [
        (0, cv2.CAP_DSHOW),
        (0, cv2.CAP_MSMF),
        (0, None),
        (1, cv2.CAP_DSHOW),
        (1, cv2.CAP_MSMF),
        (1, None),
    ] if sys.platform.startswith("win") else [(0, None), (1, None)]

    for index, backend in attempts:
        cam = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
        if cam.isOpened():
            return cam
        cam.release()
    return None

def camera_loop():
    global cap, current_frame, current_rl_data, camera_error, latest_face_info
    global calibration_frozen, calibration_frozen_until

    cap = open_camera()
    if cap is None:
        camera_error = "Cannot open webcam. Close other camera apps or check camera permissions."
        with data_lock:
            current_rl_data = {
                **current_rl_data,
                "mode": "idle",
                "suggestion": camera_error,
            }
        return

    camera_error = None

    while camera_active:
        if calibration_frozen and time.time() < calibration_frozen_until:
            with data_lock:
                current_rl_data = {
                    **current_rl_data,
                    "mode": "calibration_freeze",
                    "calibration_frozen": True,
                    "calibration_freeze_remaining": max(0.0, calibration_frozen_until - time.time()),
                    "suggestion": "Locking reference...",
                    "calibration_snapshot": calibration_snapshot,
                }
            time.sleep(0.05)
            continue

        if calibration_frozen and time.time() >= calibration_frozen_until:
            calibration_frozen = False
            calibration_frozen_until = 0.0

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        info = detect_face_info(gray)
        latest_face_info = info
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 4
        axes = (int(radius * 1.0), int(radius * 1.28))

        mode = "calibrating" if baseline is None else (
            "exam_question" if question_active else (
                "exam" if exam_active else "posture"
            )
        )

        suggestion = "Align face in the frame."
        rl_state = None
        action = None
        trust_score = 0
        is_cheating = False
        face_w = int(info.get("face_w", 0)) if info else 0
        face_h = int(info.get("face_h", 0)) if info else 0
        head_angle = float(info.get("head_angle", 0.0)) if info else 0.0
        eye_dir = float(info.get("eye_dir", 0.0)) if info else 0.0
        eye_ratio = float(info.get("eye_ratio", 0.0)) if info and info.get("eye_ratio") is not None else 0.0
        eye_distance = float(info.get("eye_dist", 0.0)) if info else 0.0
        face_inside_ratio = 0.0
        calibration_ready = False

        if baseline is None:
            cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 255, 255), 2)

        if info and baseline is None:
            _, face_inside_ratio = is_face_in_circle(info, center, radius, ellipse_axes=axes)
            calibration_ready = face_inside_ratio >= 0.5
            suggestion = "Ready to capture reference." if calibration_ready else "Move your face inside the oval."
            oval_color = (0, 255, 0) if calibration_ready else (0, 165, 255)
            cv2.ellipse(frame, center, axes, 0, 0, 360, oval_color, 3)
        elif info and baseline:
            rl_state = build_state(info, baseline)

            if rl_state:
                action_idx = choose_action(rl_state, epsilon)
                action = actions[action_idx]
                is_cheating = rl_state[2] != "looking"
                posture_is_good = not is_bad_state(rl_state)
                suggestion = "Good posture" if posture_is_good else "Adjust posture and look at the screen."
                if exam_active:
                    if posture_is_good:
                        posture_stats["good"] += 1
                    else:
                        posture_stats["bad"] += 1
                if question_active:
                    if posture_is_good:
                        question_stats["good"] = question_stats.get("good", 0) + 1
                    else:
                        question_stats["bad"] = question_stats.get("bad", 0) + 1
                total = posture_stats["good"] + posture_stats["bad"]
                trust_score = int((posture_stats["good"] / total) * 100) if total > 0 else 0
        elif baseline is not None:
            suggestion = "Face not detected."

        _, buffer = cv2.imencode('.jpg', frame)

        with data_lock:
            current_frame = buffer.tobytes()
            current_rl_data = {
                **current_rl_data,
                "state": rl_state,
                "action": action,
                "mode": mode,
                "suggestion": suggestion,
                "calibration_ready": calibration_ready,
                "calibration_frozen": calibration_frozen,
                "calibration_freeze_remaining": max(0.0, calibration_frozen_until - time.time()) if calibration_frozen else 0.0,
                "face_inside_ratio": face_inside_ratio,
                "face_width": face_w,
                "face_height": face_h,
                "head_angle": head_angle,
                "eye_dir": eye_dir,
                "eye_ratio": eye_ratio,
                "eye_distance": eye_distance,
                "calibration_snapshot": calibration_snapshot,
                "trust_score": trust_score,
                "is_cheating": is_cheating,
            }

    if cap:
        cap.release()
        cap = None

def ensure_camera():
    global camera_thread, camera_active

    if camera_thread and camera_thread.is_alive():
        return True

    camera_active = True
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    time.sleep(0.2)
    return camera_error is None

# ---------------------------
# API ROUTES
# ---------------------------
@app.route("/")
def home():
    return "Backend Running"

@app.route("/state")
def state():
    with data_lock:
        return jsonify(current_rl_data)

@app.route("/calibrate")
def calibrate():
    global baseline, exam_active, question_active, exam_start_time, question_start_time, calibration_snapshot
    global calibration_frozen, calibration_frozen_until
    baseline = None
    calibration_snapshot = None
    calibration_frozen = False
    calibration_frozen_until = 0.0
    exam_active = False
    question_active = False
    exam_start_time = None
    question_start_time = None
    return jsonify({"status": "reset"})

@app.route("/capture_reference")
def capture():
    global baseline, calibration_snapshot, calibration_frozen, calibration_frozen_until

    if latest_face_info is None:
        return jsonify({"error": "no face detected"}), 400

    baseline = {
        "face_x": float(latest_face_info.get("face_x", 0.0)),
        "face_y": float(latest_face_info.get("face_y", 0.0)),
        "face_w": int(latest_face_info.get("face_w", 0)),
        "face_h": int(latest_face_info.get("face_h", 0)),
        "head_angle": float(latest_face_info.get("head_angle", 0.0)),
        "eye_dir": float(latest_face_info.get("eye_dir", 0.0)),
        "eye_dist": float(latest_face_info.get("eye_dist", 0.0)),
        "eye_ratio": float(latest_face_info.get("eye_ratio", 0.0)) if latest_face_info.get("eye_ratio") is not None else None,
        "eyes_detected": bool(latest_face_info.get("eyes_detected", False)),
    }

    calibration_snapshot = {
        "face_w": baseline["face_w"],
        "face_h": baseline["face_h"],
        "head_angle": baseline["head_angle"],
        "eye_dir": baseline["eye_dir"],
        "eye_ratio": baseline["eye_ratio"] if baseline["eye_ratio"] is not None else 0.0,
        "eye_dist": baseline["eye_dist"],
    }

    calibration_frozen = True
    calibration_frozen_until = time.time() + CALIBRATION_FREEZE_SECONDS

    return jsonify({"status": "captured", "baseline": baseline, "calibration": calibration_snapshot})

@app.route("/start_exam")
def start_exam():
    global exam_active, posture_stats, exam_start_time, question_history

    if baseline is None:
        return jsonify({"error": "calibrate first"}), 400

    if not ensure_camera():
        return jsonify({"error": camera_error or "camera unavailable"}), 500

    exam_active = True
    posture_stats = {"good": 0, "bad": 0}
    question_history = []
    exam_start_time = time.time()

    return jsonify({"status": "started"})

@app.route("/begin_answer")
def begin_answer():
    global question_active, question_stats, question_start_time, exam_active

    if baseline is None:
        return jsonify({"error": "calibrate first"}), 400

    if not ensure_camera():
        return jsonify({"error": camera_error or "camera unavailable"}), 500

    exam_active = True
    question_active = True
    question_start_time = time.time()

    question_stats = {
        "good": 0,
        "bad": 0
    }

    return jsonify({"status": "question_started"})

@app.route("/start_question")
def start_question():
    return begin_answer()

@app.route("/end_question")
def end_question():
    global question_active, question_start_time, question_history

    if not question_active:
        return jsonify({"error": "no active question"}), 400

    question_active = False
    elapsed_time = time.time() - question_start_time if question_start_time else 0.0
    question_start_time = None

    good_frames = question_stats.get("good", 0)
    bad_frames = question_stats.get("bad", 0)
    total_frames = good_frames + bad_frames
    posture_score = round((good_frames / total_frames) * 100, 1) if total_frames > 0 else 0.0
    time_score = float(time_score_for_elapsed(elapsed_time))
    score = round((posture_score * 0.7) + (time_score * 0.3), 1)
    label = label_for_score(score)
    time_penalty = round(max(0.0, 100.0 - time_score), 1)
    errors = []

    if time_score < 50:
        errors.append({
            "key": "time_penalty",
            "description": f"Very short answer duration ({elapsed_time:.1f}s).",
            "percent_frames": round(100.0 - time_score, 1),
        })
    elif time_score < 80:
        errors.append({
            "key": "time_penalty",
            "description": f"Answer duration could be longer ({elapsed_time:.1f}s).",
            "percent_frames": round(100.0 - time_score, 1),
        })
    else:
        errors.append({
            "key": "time_reward",
            "description": f"Good answer duration ({elapsed_time:.1f}s).",
            "percent_frames": time_score,
        })

    question_history.append(score)

    return jsonify({
        "status": "ended",
        "score": score,
        "label": label,
        "elapsed_time": round(elapsed_time, 1),
        "base_score": posture_score,
        "time_score": time_score,
        "time_penalty": time_penalty,
        "good_frames": good_frames,
        "bad_frames": bad_frames,
        "errors": errors,
    })

@app.route("/end_exam")
def end_exam():
    global exam_active, question_active, question_start_time, question_history

    exam_active = False
    question_active = False
    question_start_time = None

    if question_history:
        score = round(sum(question_history) / len(question_history), 1)
    else:
        total = posture_stats["good"] + posture_stats["bad"]
        posture_score = round((posture_stats["good"] / total) * 100, 1) if total > 0 else 0.0
        elapsed_time = time.time() - exam_start_time if exam_start_time else 0.0
        time_score = float(time_score_for_elapsed(elapsed_time / 3 if elapsed_time > 0 else 0.0))
        score = round((posture_score * 0.7) + (time_score * 0.3), 1)

    return jsonify({
        "status": "exam_ended",
        "score": score,
        "label": label_for_score(score),
        "base_score": round(score, 1),
        "time_penalty": 0,
    })

@app.route("/stop_session")
def stop():
    global camera_active, cap, camera_thread, exam_active, question_active

    camera_active = False
    exam_active = False
    question_active = False
    if cap:
        cap.release()
        cap = None
    camera_thread = None
    return jsonify({"status": "stopped"})

@app.route("/video_feed")
def video():
    ensure_camera()

    def gen():
        while camera_active:
            with data_lock:
                frame = current_frame

            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       frame + b'\r\n')

            time.sleep(0.03)

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------------
# CLEANUP
# ---------------------------
@atexit.register
def cleanup():
    global cap
    if cap:
        cap.release()
    save_q_table()

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    print("Routes:")
    for r in app.url_map.iter_rules():
        print(r)

    app.run(port=8000, debug=True, use_reloader=False)
