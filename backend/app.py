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
camera_error = None
camera_lock = threading.Lock()
camera_start_event = threading.Event()

current_frame = None

# Time tracking for scoring penalties
exam_start_time = None
question_start_time = None

current_rl_data = {
    "state": None,
    "action": None,
    "reward": 0,
    "is_bad": False,
    "is_cheating": False,
    "trust_score": 0,
    "identified_by": "Initializing...",
    "mode": "idle",
    "calibration_ready": False,
    "face_inside_ratio": 0.0,
    "face_width": 0,
    "face_height": 0,
    "head_angle": 0.0,
    "eye_dir": 0.0,
    "eye_ratio": 0.0,
    "eye_distance": 0.0,
    "calibration_snapshot": None,
    "calibration_frozen": False,
    "calibration_freeze_remaining": 0.0,
    "suggestion": "Waiting to start..."
}

data_lock = threading.Lock()

# ---------------------------
# PROCTOR EXAM TRACKING
# ---------------------------
exam_active = False
question_active = False
posture_stats = {"good": 0, "bad": 0}

# Per-question accumulators (reset on /start_question)
question_stats = {
    "good": 0,
    "bad": 0,
    "position_error": 0,
    "head_error": 0,
    "gaze_error": 0,
    "reward_total": 0.0,
    "reward_transitions": 0
}

question_last_state = None

# ---------------------------
# CALIBRATION STATE
# ---------------------------
latest_face_info = None
latest_face_inside_ratio = 0.0
calibration_snapshot = None
CALIBRATION_CAPTURE_MIN_RATIO = 0.5
calibration_frozen = False
calibration_frozen_until = 0.0
CALIBRATION_FREEZE_SECONDS = 1.5
POST_CAPTURE_GRACE_SECONDS = 3.0
post_capture_grace_until = 0.0

# ---------------------------
# CAMERA THREAD
# ---------------------------
def camera_background_task():
    global cap, baseline, epsilon
    global current_frame, current_rl_data
    global exam_active, question_active, posture_stats
    global question_stats, question_last_state
    global latest_face_info, latest_face_inside_ratio, calibration_snapshot
    global calibration_frozen, calibration_frozen_until, post_capture_grace_until
    global camera_active, camera_thread, camera_error

    if cap is None:
        try:
            if sys.platform.startswith('darwin'):
                cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            elif sys.platform.startswith('win'):
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                raise RuntimeError('Cannot open webcam. Check camera permissions and device index.')
        except Exception as e:
            camera_error = str(e)
            camera_active = False
            camera_thread = None
            camera_start_event.set()
            print(f"Error opening camera: {e}")
            with data_lock:
                current_rl_data = {
                    **current_rl_data,
                    "mode": "idle",
                    "suggestion": camera_error,
                }
            cap = None
            return

    camera_error = None
    camera_start_event.set()

    missing_face_count = 0
    MAX_MISSING_FRAMES = 60

    last_state = None
    frame_counter = 0

    while camera_active:
        # Freeze the camera frames during manual calibration capture.
        if calibration_frozen:
            if time.time() < calibration_frozen_until:
                with data_lock:
                    current_rl_data = {
                        **current_rl_data,
                        "mode": "calibration_freeze",
                        "calibration_frozen": True,
                        "calibration_freeze_remaining": max(0.0, calibration_frozen_until - time.time()),
                        "calibration_ready": False,
                        "suggestion": "Locking reference frame..."
                    }
                time.sleep(0.05)
                continue
            calibration_frozen = False
            calibration_frozen_until = 0.0

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 4
        axes = (int(radius * 1.0), int(radius * 1.28))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        info = detect_face_info(gray)

        rl_state = None
        action = None
        reward = 0
        ai_is_bad_posture = False
        is_cheating = False
        trust_score = 0
        identified_by = "Waiting for alignment..."
        mode = "calibrating" if baseline is None else "posture"
        calibration_ready = False
        face_inside_ratio = 0.0
        face_width = 0
        face_height = 0
        head_angle = 0.0
        eye_dir = 0.0
        eye_ratio = 0.0
        eye_distance = 0.0
        suggestion = "Align your face inside the oval."

        # Draw default oval only during manual calibration mode.
        if baseline is None:
            cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 255, 255), 2)

        # ---------------------------
        # NO FACE
        # ---------------------------
        if info is None:
            missing_face_count += 1

            if baseline is None:
                identified_by = "No Face"
            else:
                identified_by = "Face temporarily lost"
                suggestion = "Face lost briefly. Return to the frame."

        else:
            missing_face_count = 0

            is_inside, ratio = is_face_in_circle(info, center, radius, ellipse_axes=axes)

            # ---------------------------
            # MANUAL CALIBRATION
            # ---------------------------
            if baseline is None:
                mode = "calibrating"
                face_inside_ratio = float(ratio)
                calibration_ready = (face_inside_ratio >= CALIBRATION_CAPTURE_MIN_RATIO)

                # Cache the candidate info for `/capture_reference`.
                latest_face_info = info.copy()
                latest_face_inside_ratio = face_inside_ratio
                face_width = int(info.get("face_w", 0))
                face_height = int(info.get("face_h", 0))
                head_angle = float(info.get("head_angle", 0.0))
                eye_dir = float(info.get("eye_dir", 0.0))
                eye_ratio = float(info.get("eye_ratio", 0.0)) if info.get("eye_ratio") is not None else 0.0
                eye_distance = float(info.get("eye_dist", 0.0))
                if calibration_ready:
                    identified_by = "Capture Ready (green)"
                    suggestion = "Ready. Click 'Capture Reference'."
                    cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 255, 0), 2)
                else:
                    identified_by = "Move face into oval"
                    suggestion = "Move your face into the green zone (>60%)."
                    cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 165, 255), 2)

            # ---------------------------
            # RL TRACKING / EXAM MODE
            # ---------------------------
            else:
                mode = "exam_question" if question_active else ("exam" if exam_active else "posture")

                rl_state = build_state(info, baseline)

                if rl_state:
                    position, head, gaze = rl_state
                    in_post_capture_grace = time.time() < post_capture_grace_until

                    # Detect posture
                    ai_is_bad_posture = False if in_post_capture_grace else is_bad_state(rl_state)

                    # Frontend proctor UI expects `is_cheating` to reflect eye-contact/gaze loss.
                    # `build_state` returns (position, head, gaze) where gaze is "looking" or "away".
                    is_cheating = False if in_post_capture_grace else (gaze != "looking")

                    # Human-readable suggestion (used in the UI).
                    if in_post_capture_grace:
                        suggestion = "Reference captured. Hold steady."
                    elif position != "centered":
                        suggestion = "Center your face in the frame."
                    elif head != "straight":
                        suggestion = "Keep your head straight."
                    elif gaze != "looking":
                        suggestion = "Look at the camera/screen."
                    else:
                        suggestion = "Good posture. Stay steady."

                    # Identify source
                    if in_post_capture_grace:
                        identified_by = "Calibration Locked"
                    elif rl_state in Q and np.any(Q[rl_state]):
                        identified_by = "Q-Table"
                    else:
                        identified_by = "Heuristic"

                    # RL action
                    action_index = choose_action(rl_state, epsilon)
                    action = actions[action_index]

                    # RL update (Q-learning)
                    if last_state is not None:
                        reward = reward_for_transition(last_state, rl_state)
                        next_max = max(Q[rl_state]) if rl_state in Q else 0
                        Q[last_state][action_index] += alpha * (
                            reward + gamma * next_max - Q[last_state][action_index]
                        )

                    last_state = rl_state

                    epsilon = max(min_epsilon, epsilon * epsilon_decay)

                    # ---------------------------
                    # QUESTION SCORING
                    # ---------------------------
                    if question_active and not in_post_capture_grace:
                        if ai_is_bad_posture:
                            question_stats["bad"] += 1
                        else:
                            question_stats["good"] += 1

                        if position != "centered":
                            question_stats["position_error"] += 1
                        if head != "straight":
                            question_stats["head_error"] += 1
                        if gaze != "looking":
                            question_stats["gaze_error"] += 1

                        if question_last_state is not None:
                            q_reward = reward_for_transition(question_last_state, rl_state)
                            question_stats["reward_total"] += float(q_reward)
                            question_stats["reward_transitions"] += 1

                        question_last_state = rl_state

                    # ---------------------------
                    # OVERALL EXAM TRUST (accumulate only while answering)
                    # ---------------------------
                    if question_active and not in_post_capture_grace:
                        if ai_is_bad_posture:
                            posture_stats["bad"] += 1
                        else:
                            posture_stats["good"] += 1

        # Derived proctor metrics for the frontend (do not change RL/Q logic).
        if exam_active:
            total = posture_stats["good"] + posture_stats["bad"]
            trust_score = int((posture_stats["good"] / total) * 100) if total > 0 else 0
        else:
            trust_score = 0

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
                "is_cheating": bool(is_cheating),
                "trust_score": trust_score,
                "identified_by": identified_by,
                "mode": mode,
                "calibration_ready": calibration_ready,
                "face_inside_ratio": face_inside_ratio,
                "face_width": face_width,
                "face_height": face_height,
                "head_angle": head_angle,
                "eye_dir": eye_dir,
                "eye_ratio": eye_ratio,
                "eye_distance": eye_distance,
                "calibration_snapshot": calibration_snapshot,
                "calibration_frozen": calibration_frozen,
                "calibration_freeze_remaining": max(0.0, calibration_frozen_until - time.time()) if calibration_frozen else 0.0,
                "suggestion": suggestion,
            }

    with camera_lock:
        camera_thread = None


def ensure_camera_running():
    global camera_active, camera_thread, camera_error, current_frame

    with camera_lock:
        if camera_thread is not None and camera_thread.is_alive():
            return True, None

        camera_error = None
        current_frame = None
        camera_start_event.clear()
        camera_active = True
        camera_thread = threading.Thread(target=camera_background_task, daemon=True)
        camera_thread.start()

    started = camera_start_event.wait(timeout=2.0)
    if started and not camera_error:
        return True, None

    with camera_lock:
        if camera_thread is not None and not camera_thread.is_alive():
            camera_thread = None
        camera_active = False

    return False, camera_error or "Camera did not start in time."


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
    global exam_active, question_active, posture_stats
    global question_stats, question_last_state
    global calibration_frozen, post_capture_grace_until, exam_start_time

    if baseline is None:
        return jsonify({"status": "error", "message": "Please complete calibration first."}), 400

    exam_active = True
    question_active = False
    posture_stats = {"good": 0, "bad": 0}
    exam_start_time = time.time()

    question_stats = {
        "good": 0,
        "bad": 0,
        "position_error": 0,
        "head_error": 0,
        "gaze_error": 0,
        "reward_total": 0.0,
        "reward_transitions": 0,
    }
    question_last_state = None

    calibration_frozen = False
    post_capture_grace_until = 0.0

    camera_ready, error_message = ensure_camera_running()
    if not camera_ready:
        exam_active = False
        return jsonify({"status": "error", "message": error_message}), 500
        print("📹 Camera started")

    print("🎯 Exam started")
    return jsonify({"status": "started"})


@app.route("/redo_question")
def redo_question():
    global exam_active, question_active, question_stats, question_last_state, question_start_time

    if not exam_active:
        return jsonify({"status": "error", "message": "No active exam."}), 400

    # Reset question state but keep exam active
    question_active = False
    question_start_time = None

    question_stats = {
        "good": 0,
        "bad": 0,
        "position_error": 0,
        "head_error": 0,
        "gaze_error": 0,
        "reward_total": 0.0,
        "reward_transitions": 0,
    }
    question_last_state = None

    print("🔄 Question reset for redo")

    return jsonify({"status": "question_reset"})


@app.route("/start_question")
def start_question():
    global exam_active, question_active, question_stats, question_last_state, question_start_time

    if baseline is None:
        return jsonify({"status": "error", "message": "Please complete calibration first."}), 400

    if not exam_active:
        return jsonify({"status": "error", "message": "Start the exam session first."}), 400

    camera_ready, error_message = ensure_camera_running()
    if not camera_ready:
        return jsonify({"status": "error", "message": error_message}), 500

    question_active = True
    question_start_time = time.time()
    question_last_state = None
    question_stats = {
        "good": 0,
        "bad": 0,
        "position_error": 0,
        "head_error": 0,
        "gaze_error": 0,
        "reward_total": 0.0,
        "reward_transitions": 0,
    }

    return jsonify({"status": "question_started"})


@app.route("/end_question")
def end_question():
    global question_active, question_stats, question_last_state, question_start_time

    if not question_active:
        return jsonify({"status": "error", "message": "No active question."}), 400

    question_active = False
    question_last_state = None

    # Calculate time elapsed
    elapsed_time = time.time() - question_start_time if question_start_time else 180
    time_penalty = 0

    # Time-based penalties: if answered too quickly (< 30 seconds), penalty
    if elapsed_time < 30:
        time_penalty = 20  # 20% penalty for rushing
    elif elapsed_time < 60:
        time_penalty = 10  # 10% penalty for somewhat rushed
    elif elapsed_time > 150:  # Took too long (> 2.5 minutes)
        time_penalty = 5   # 5% penalty for taking too long

    good_frames = question_stats["good"]
    bad_frames = question_stats["bad"]
    total_frames = good_frames + bad_frames
    base_score = int((good_frames / total_frames) * 100) if total_frames > 0 else 0

    # Apply time penalty
    score = max(0, base_score - time_penalty)

    # Error breakdown (frame-based)
    def err_pct(count):
        return round((count / total_frames) * 100, 1) if total_frames > 0 else 0.0

    errors = []
    if question_stats["position_error"] > 0:
        errors.append({
            "key": "position",
            "description": "Face not centered (off-center posture).",
            "count_frames": question_stats["position_error"],
            "percent_frames": err_pct(question_stats["position_error"]),
        })
    if question_stats["head_error"] > 0:
        errors.append({
            "key": "head",
            "description": "Head not straight (tilted posture).",
            "count_frames": question_stats["head_error"],
            "percent_frames": err_pct(question_stats["head_error"]),
        })
    if question_stats["gaze_error"] > 0:
        errors.append({
            "key": "gaze",
            "description": "Gaze away from camera (eye contact lost).",
            "count_frames": question_stats["gaze_error"],
            "percent_frames": err_pct(question_stats["gaze_error"]),
        })

    # Add time penalty to errors if applicable
    if time_penalty > 0:
        time_desc = f"Time penalty: answered in {elapsed_time:.1f}s"
        if elapsed_time < 30:
            time_desc += " (too rushed)"
        elif elapsed_time > 150:
            time_desc += " (took too long)"
        errors.append({
            "key": "time_penalty",
            "description": time_desc,
            "count_frames": 0,
            "percent_frames": time_penalty,
        })

    # Provide at least one error line for UI consistency.
    if not errors:
        errors = [{
            "key": "none",
            "description": "No posture/gaze errors detected during this interval.",
            "count_frames": 0,
            "percent_frames": 0.0,
        }]

    label = "Excellent" if score > 80 else ("Good" if score > 60 else ("Average" if score > 40 else "Poor"))

    return jsonify({
        "status": "question_ended",
        "score": score,
        "base_score": base_score,
        "time_penalty": time_penalty,
        "elapsed_time": round(elapsed_time, 1),
        "label": label,
        "good_frames": good_frames,
        "bad_frames": bad_frames,
        "reward_total": question_stats["reward_total"],
        "reward_transitions": question_stats["reward_transitions"],
        "errors": errors
    })


@app.route("/end_exam")
def end_exam():
    global exam_active, question_active, exam_start_time, question_start_time, question_last_state

    if not exam_active:
        return jsonify({"status": "error", "message": "No active exam."}), 400

    question_active = False
    question_start_time = None
    question_last_state = None

    total = posture_stats["good"] + posture_stats["bad"]
    base_score = int((posture_stats["good"] / total) * 100) if total > 0 else 0

    elapsed_time = time.time() - exam_start_time if exam_start_time else 0
    time_penalty = 0
    if elapsed_time > 9 * 60:
        time_penalty = 10
    elif elapsed_time > 7 * 60:
        time_penalty = 5

    score = max(0, base_score - time_penalty)
    label = "Excellent" if score > 80 else ("Good" if score > 60 else ("Average" if score > 40 else "Poor"))

    exam_active = False
    exam_start_time = None

    return jsonify({
        "status": "exam_ended",
        "score": score,
        "base_score": base_score,
        "time_penalty": time_penalty,
        "elapsed_time": round(elapsed_time, 1),
        "label": label,
        "good_frames": posture_stats["good"],
        "bad_frames": posture_stats["bad"],
    })


@app.route("/restart_exam")
def restart_exam():
    global exam_active, question_active, posture_stats, question_stats, question_last_state
    global exam_start_time, question_start_time

    # Reset all exam state
    exam_active = False
    question_active = False
    posture_stats = {"good": 0, "bad": 0}
    exam_start_time = None
    question_start_time = None

    question_stats = {
        "good": 0,
        "bad": 0,
        "position_error": 0,
        "head_error": 0,
        "gaze_error": 0,
        "reward_total": 0.0,
        "reward_transitions": 0,
    }
    question_last_state = None

    print("🔄 Exam session restarted")

    return jsonify({"status": "exam_restarted"})


@app.route("/stop_session")
def stop_session():
    global camera_active, cap, camera_thread, camera_error, current_frame
    global exam_active, question_active
    camera_active = False
    exam_active = False
    question_active = False
    camera_error = None
    current_frame = None
    
    if cap:
        cap.release()
        cap = None

    with camera_lock:
        camera_thread = None
        print("📷 Camera released")
    
    save_q_table()
    print("💾 Q-table saved")
    
    return jsonify({"status": "stopped"})


@app.route("/calibrate")
def calibrate():
    global baseline, latest_face_info, latest_face_inside_ratio, calibration_snapshot
    global calibration_frozen, calibration_frozen_until, post_capture_grace_until
    global exam_active, question_active, posture_stats
    global question_stats, question_last_state

    baseline = None
    latest_face_info = None
    latest_face_inside_ratio = 0.0
    calibration_snapshot = None

    calibration_frozen = False
    calibration_frozen_until = 0.0
    post_capture_grace_until = 0.0

    # Calibration and exam should not overlap.
    exam_active = False
    question_active = False
    posture_stats = {"good": 0, "bad": 0}
    question_stats = {
        "good": 0,
        "bad": 0,
        "position_error": 0,
        "head_error": 0,
        "gaze_error": 0,
        "reward_total": 0.0,
        "reward_transitions": 0,
    }
    question_last_state = None

    return jsonify({"status": "calibration_reset"})


@app.route("/capture_reference")
def capture_reference():
    global baseline, latest_face_info, latest_face_inside_ratio, calibration_snapshot
    global calibration_frozen, calibration_frozen_until, post_capture_grace_until
    global exam_active, question_active

    if baseline is not None:
        return jsonify({"status": "error", "message": "Calibration already captured."}), 400

    if latest_face_info is None:
        return jsonify({"status": "error", "message": "No face candidate available. Move into the circle."}), 400

    if latest_face_inside_ratio < CALIBRATION_CAPTURE_MIN_RATIO:
        return jsonify({
            "status": "error",
            "message": f"Move face into green zone (> {int(CALIBRATION_CAPTURE_MIN_RATIO*100)}%).",
            "face_inside_ratio": latest_face_inside_ratio
        }), 400

    baseline = latest_face_info.copy()
    # Freeze camera briefly so the UI sees a stable calibration moment.
    calibration_frozen = True
    calibration_frozen_until = time.time() + CALIBRATION_FREEZE_SECONDS
    post_capture_grace_until = calibration_frozen_until + POST_CAPTURE_GRACE_SECONDS

    # If someone tries capturing during an exam, stop scoring.
    question_active = False
    exam_active = False

    calibration_snapshot = {
        "face_x": float(baseline.get("face_x")),
        "face_y": float(baseline.get("face_y")),
        "face_w": int(baseline.get("face_w")),
        "face_h": int(baseline.get("face_h")),
        "head_angle": float(baseline.get("head_angle")),
        "eye_dir": float(baseline.get("eye_dir")),
        "eye_ratio": float(baseline.get("eye_ratio")) if baseline.get("eye_ratio") is not None else None,
        "eye_dist": float(baseline.get("eye_dist")),
    }

    print("✅ Calibration captured:", calibration_snapshot)
    return jsonify({"status": "captured", "calibration": calibration_snapshot})


@app.route("/video_feed")
def video_feed():
    camera_ready, error_message = ensure_camera_running()
    if not camera_ready:
        return jsonify({"status": "error", "message": error_message}), 500
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
