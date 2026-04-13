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
    from backend.model.camera_utils import *
except:
    from model.camera_utils import *

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
QUESTION_ERROR_TOLERANCE_PERCENT = 10.0
STATE_POLL_SECONDS = 0.5
QUESTION_DURATION_SECONDS = 5

baseline = None
prev_face_center = None
gaze_away_counter = 0
stable_state = None
state_candidate = None
state_candidate_count = 0

exam_active = False
question_active = False

posture_stats = {"good": 0, "bad": 0}
question_stats = {"good": 0, "bad": 0}
question_last_state = None
question_history = []

exam_start_time = None
question_start_time = None

data_lock = threading.Lock()

current_state_data = {
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
    "normalized_face_x": 0.0,
    "normalized_face_y": 0.0,
    "normalized_head_angle": 0.0,
    "normalized_eye_dir": 0.0,
    "movement": 0.0,
    "movement_value": 0.0,
    "movement_level": "low",
    "attention_duration": 0.0,
    "gaze_away_counter": 0,
    "processing_flags": {
        "edge_detected": False,
        "threshold_applied": False,
        "corners_detected": False,
    },
    "error_tolerance_percent": QUESTION_ERROR_TOLERANCE_PERCENT,
    "question_duration_seconds": QUESTION_DURATION_SECONDS,
    "position": None,
    "head": None,
    "gaze": None,
    "identified_by": "Machine Vision Rules",
    "trust_score": 100,
    "reward": 0,
    "pipeline_status": {
        "preprocessing": True,
        "segmentation": True,
        "feature_extraction": True,
        "normalization": True,
        "classification": True,
        "temporal_smoothing": True,
        "motion_analysis": True,
        "decision": True,
    },
    "vision_processing": {
        "edge_detection": True,
        "thresholding": True,
        "corner_detection": True,
    },
    "is_cheating": False,
    "action": None,
    "state": None,
}


def time_score_for_elapsed(elapsed_seconds, target_seconds):
    safe_target = max(float(target_seconds), 1.0)
    usage_percent = min((float(elapsed_seconds) / safe_target) * 100.0, 100.0)
    return round(usage_percent, 1)


def label_for_score(score):
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Average"
    return "Poor"


def draw_dashed_ellipse(frame, center, axes, color, thickness=1, dash=10, gap=8):
    angle = 0
    while angle < 360:
      dash_end = min(angle + dash, 360)
      cv2.ellipse(frame, center, axes, 0, angle, dash_end, color, thickness, cv2.LINE_AA)
      angle += dash + gap


def smooth_state(raw_state, required_frames=3):
    global stable_state, state_candidate, state_candidate_count

    if raw_state is None:
        state_candidate = None
        state_candidate_count = 0
        return stable_state

    if stable_state is None:
        stable_state = raw_state
        state_candidate = None
        state_candidate_count = 0
        return raw_state

    if raw_state == stable_state:
        state_candidate = None
        state_candidate_count = 0
        return stable_state

    if raw_state == state_candidate:
        state_candidate_count += 1
    else:
        state_candidate = raw_state
        state_candidate_count = 1

    if state_candidate_count >= required_frames:
        stable_state = raw_state
        state_candidate = None
        state_candidate_count = 0

    return stable_state

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
    global cap, current_frame, current_state_data, camera_error, latest_face_info
    global calibration_frozen, calibration_frozen_until, prev_face_center, gaze_away_counter
    global stable_state, state_candidate, state_candidate_count

    cap = open_camera()
    if cap is None:
        camera_error = "Cannot open webcam. Close other camera apps or check camera permissions."
        with data_lock:
            current_state_data = {
                **current_state_data,
                "mode": "idle",
                "suggestion": camera_error,
            }
        return

    camera_error = None
    prev_face_center = None
    gaze_away_counter = 0
    stable_state = None
    state_candidate = None
    state_candidate_count = 0

    while camera_active:
        if calibration_frozen and time.time() < calibration_frozen_until:
            with data_lock:
                current_state_data = {
                    **current_state_data,
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
        # Light smoothing improves cascade stability without adding heavy cost.
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.equalizeHist(gray)
        edges = cv2.Canny(gray, 100, 200)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
        processing_flags = {
            "edge_detected": edges is not None,
            "threshold_applied": thresh is not None,
            "corners_detected": corners is not None,
        }
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
        state = None
        action = None
        trust_score = 100
        reward = 0
        is_cheating = False
        movement = 0.0
        movement_label = "low"
        face_w = int(info.get("face_w", 0)) if info else 0
        face_h = int(info.get("face_h", 0)) if info else 0
        head_angle = float(info.get("head_angle", 0.0)) if info else 0.0
        eye_dir = float(info.get("eye_dir", 0.0)) if info else 0.0
        eye_ratio = float(info.get("eye_ratio", 0.0)) if info and info.get("eye_ratio") is not None else 0.0
        eye_distance = float(info.get("eye_dist", 0.0)) if info else 0.0
        normalized_face_x = 0.0
        normalized_face_y = 0.0
        normalized_head_angle = 0.0
        normalized_eye_dir = 0.0
        face_inside_ratio = 0.0
        calibration_ready = False

        if baseline is None:
            draw_dashed_ellipse(frame, center, axes, (0, 165, 255), thickness=1)

        if info and baseline is None:
            _, face_inside_ratio = is_face_in_circle(info, center, radius, ellipse_axes=axes)
            calibration_ready = face_inside_ratio >= 0.5
            suggestion = "Ready to capture reference." if calibration_ready else "Move your face inside the oval."
            oval_color = (0, 255, 0) if calibration_ready else (0, 165, 255)
            draw_dashed_ellipse(frame, center, axes, oval_color, thickness=1)
            prev_face_center = (float(info["face_x"]), float(info["face_y"]))
            gaze_away_counter = 0
        elif info and baseline:
            state = smooth_state(build_state(info, baseline))
            baseline_face_w = max(float(baseline.get("face_w", 1.0)), 1.0)
            baseline_face_h = max(float(baseline.get("face_h", 1.0)), 1.0)
            normalized_face_x = (float(info.get("face_x", 0.0)) - float(baseline.get("face_x", 0.0))) / baseline_face_w
            normalized_face_y = (float(info.get("face_y", 0.0)) - float(baseline.get("face_y", 0.0))) / baseline_face_h
            normalized_head_angle = float(info.get("head_angle", 0.0)) - float(baseline.get("head_angle", 0.0))
            normalized_eye_dir = float(info.get("eye_dir", 0.0)) - float(baseline.get("eye_dir", 0.0))

            if state:
                current_face_center = (float(info["face_x"]), float(info["face_y"]))
                if prev_face_center is not None:
                    dx = current_face_center[0] - prev_face_center[0]
                    dy = current_face_center[1] - prev_face_center[1]
                    movement = float(np.sqrt((dx * dx) + (dy * dy)))
                prev_face_center = current_face_center
                movement_label = movement_level(movement)

                if state[2] == "away":
                    gaze_away_counter += 1
                else:
                    gaze_away_counter = 0

                action = decide_action(state, movement, gaze_away_counter)
                is_cheating = state[2] != "looking"
                posture_is_good = not is_bad_state(state)
                if action == "look_screen":
                    suggestion = "Look at the screen."
                elif action == "reduce_movement":
                    suggestion = "Reduce movement."
                elif action == "adjust_posture":
                    suggestion = "Adjust posture."
                else:
                    suggestion = "Good posture"
                if exam_active and question_active:
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
                question_total = question_stats.get("good", 0) + question_stats.get("bad", 0)
                if question_total > 0:
                    trust_score = int((question_stats.get("good", 0) / question_total) * 100)
                else:
                    trust_score = int((posture_stats["good"] / total) * 100) if total > 0 else 100
                if action == "no_action":
                    reward = 1
                elif action == "reduce_movement":
                    reward = -1
                elif action in ("adjust_posture", "look_screen"):
                    reward = -0.6
        elif baseline is not None:
            suggestion = "Face not detected."
            action = "look_screen"
            reward = -1
            is_cheating = True
            prev_face_center = None
            gaze_away_counter = 0
            if exam_active and question_active:
                posture_stats["bad"] += 1
            if question_active:
                question_stats["bad"] = question_stats.get("bad", 0) + 1
            total = posture_stats["good"] + posture_stats["bad"]
            question_total = question_stats.get("good", 0) + question_stats.get("bad", 0)
            if question_total > 0:
                trust_score = int((question_stats.get("good", 0) / question_total) * 100)
            else:
                trust_score = int((posture_stats["good"] / total) * 100) if total > 0 else 100

        debug_overlay = frame.copy()
        debug_overlay[edges > 0] = (255, 255, 255)
        if corners is not None and corners.max() > 0:
            debug_overlay[corners > 0.02 * corners.max()] = (255, 0, 255)
        frame = cv2.addWeighted(debug_overlay, 0.12, frame, 0.88, 0)

        if info:
            fx = int(info["face_x"] - info["face_w"] / 2)
            fy = int(info["face_y"] - info["face_h"] / 2)
            fw = int(info["face_w"])
            fh = int(info["face_h"])
            box_color = (0, 255, 0) if action in (None, "no_action") else (0, 165, 255)
            if action in ("look_screen", "reduce_movement"):
                box_color = (0, 0, 255)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 1)
            cv2.circle(frame, (int(info["face_x"]), int(info["face_y"])), 4, box_color, -1)

        _, buffer = cv2.imencode('.jpg', frame)
        position = state[0] if state else None
        head = state[1] if state else None
        gaze = state[2] if state else None
        attention_duration = round(gaze_away_counter * STATE_POLL_SECONDS, 2)
        pipeline_status = {
            "preprocessing": True,
            "segmentation": True,
            "feature_extraction": True,
            "normalization": True,
            "classification": True,
            "temporal_smoothing": True,
            "motion_analysis": True,
            "decision": True,
        }
        vision_processing = {
            "edge_detection": processing_flags["edge_detected"],
            "thresholding": processing_flags["threshold_applied"],
            "corner_detection": processing_flags["corners_detected"],
        }

        with data_lock:
            current_frame = buffer.tobytes()
            current_state_data = {
                **current_state_data,
                "state": state,
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
                "normalized_face_x": normalized_face_x,
                "normalized_face_y": normalized_face_y,
                "normalized_head_angle": normalized_head_angle,
                "normalized_eye_dir": normalized_eye_dir,
                "movement": movement,
                "movement_value": movement,
                "movement_level": movement_label,
                "attention_duration": attention_duration,
                "gaze_away_counter": gaze_away_counter,
                "processing_flags": processing_flags,
                "pipeline_status": pipeline_status,
                "vision_processing": vision_processing,
                "error_tolerance_percent": QUESTION_ERROR_TOLERANCE_PERCENT,
                "position": position,
                "head": head,
                "gaze": gaze,
                "calibration_snapshot": calibration_snapshot,
                "identified_by": "Machine Vision Rules",
                "trust_score": trust_score,
                "reward": reward,
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
        return jsonify(current_state_data)

@app.route("/calibrate")
def calibrate():
    global baseline, exam_active, question_active, exam_start_time, question_start_time, calibration_snapshot
    global calibration_frozen, calibration_frozen_until, prev_face_center, gaze_away_counter
    global stable_state, state_candidate, state_candidate_count
    baseline = None
    prev_face_center = None
    gaze_away_counter = 0
    stable_state = None
    state_candidate = None
    state_candidate_count = 0
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
    global stable_state, state_candidate, state_candidate_count

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
    stable_state = None
    state_candidate = None
    state_candidate_count = 0

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
        "bad": 0,
        "started_at": question_start_time,
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

    ended_at = time.time()
    question_active = False
    elapsed_time = ended_at - question_start_time if question_start_time else 0.0
    question_start_time = None

    good_frames = question_stats.get("good", 0)
    bad_frames = question_stats.get("bad", 0)
    total_frames = good_frames + bad_frames
    raw_error_percent = round((bad_frames / total_frames) * 100, 1) if total_frames > 0 else 100.0
    penalty_error_percent = round(max(0.0, raw_error_percent - QUESTION_ERROR_TOLERANCE_PERCENT), 1)
    posture_score = round(max(0.0, 100.0 - penalty_error_percent), 1)
    time_score = float(time_score_for_elapsed(elapsed_time, QUESTION_DURATION_SECONDS))
    time_penalty = round(max(0.0, 100.0 - time_score), 1)
    score = round(max(0.0, (posture_score * 0.25) + (time_score * 0.75)), 1)
    label = label_for_score(score)
    errors = []
    time_usage_percent = round(min((elapsed_time / max(float(QUESTION_DURATION_SECONDS), 1.0)) * 100.0, 100.0), 1)

    if penalty_error_percent > 0:
        errors.append({
            "key": "posture_penalty",
            "description": f"Posture/gaze drift exceeded {QUESTION_ERROR_TOLERANCE_PERCENT:.0f}% tolerance.",
            "percent_frames": penalty_error_percent,
        })
    else:
        errors.append({
            "key": "posture_tolerance",
            "description": f"Posture/gaze drift stayed within {QUESTION_ERROR_TOLERANCE_PERCENT:.0f}% tolerance.",
            "percent_frames": raw_error_percent,
        })

    if time_usage_percent < 50:
        errors.append({
            "key": "time_penalty",
            "description": f"Only {time_usage_percent:.1f}% of the allotted time was used ({elapsed_time:.1f}s of {QUESTION_DURATION_SECONDS}s).",
            "percent_frames": round(100.0 - time_score, 1),
        })
    elif time_usage_percent < 80:
        errors.append({
            "key": "time_penalty",
            "description": f"Answer used {time_usage_percent:.1f}% of the allotted time ({elapsed_time:.1f}s of {QUESTION_DURATION_SECONDS}s).",
            "percent_frames": round(100.0 - time_score, 1),
        })
    else:
        errors.append({
            "key": "time_credit",
            "description": f"Good time usage: {time_usage_percent:.1f}% of the allotted time ({elapsed_time:.1f}s of {QUESTION_DURATION_SECONDS}s).",
            "percent_frames": time_score,
        })

    question_history.append({
        "score": score,
        "elapsed_time": round(elapsed_time, 1),
        "good_frames": good_frames,
        "bad_frames": bad_frames,
        "total_frames": total_frames,
        "raw_error_percent": raw_error_percent,
        "penalty_error_percent": penalty_error_percent,
        "time_score": time_score,
        "time_penalty": time_penalty,
        "time_usage_percent": time_usage_percent,
    })

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
        "total_frames": total_frames,
        "error_tolerance_percent": QUESTION_ERROR_TOLERANCE_PERCENT,
        "raw_error_percent": raw_error_percent,
        "penalty_error_percent": penalty_error_percent,
        "time_usage_percent": time_usage_percent,
        "question_duration_seconds": QUESTION_DURATION_SECONDS,
        "errors": errors,
    })

@app.route("/end_exam")
def end_exam():
    global exam_active, question_active, question_start_time, question_history

    exam_active = False
    question_active = False
    question_start_time = None

    if question_history:
        score = round(sum(q["score"] for q in question_history) / len(question_history), 1)
    else:
        score = 0.0

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

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    print("Routes:")
    for r in app.url_map.iter_rules():
        print(r)

    app.run(port=8000, debug=True, use_reloader=False)
