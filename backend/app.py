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
    "is_cheating": False,
    "trust_score": 0,
    "identified_by": "Initializing...",
    "mode": "idle",
    "calibration_ready": False,
    "face_inside_ratio": 0.0,
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
CALIBRATION_CAPTURE_MIN_RATIO = 0.8
calibration_frozen = False
calibration_frozen_until = 0.0
CALIBRATION_FREEZE_SECONDS = 1.5

# ---------------------------
# CAMERA THREAD
# ---------------------------
def camera_background_task():
    global cap, baseline, epsilon
    global current_frame, current_rl_data
    global exam_active, question_active, posture_stats
    global question_stats, question_last_state
    global latest_face_info, latest_face_inside_ratio
    global calibration_frozen, calibration_frozen_until

    if cap is None:
        if sys.platform.startswith('darwin'):
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        elif sys.platform.startswith('win'):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam. Check camera permissions and device index.')

    missing_face_count = 0
    MAX_MISSING_FRAMES = 60

    last_state = None
    frame_counter = 0

    while camera_active:
        # Freeze the camera frames during manual calibration capture.
        if calibration_frozen:
            if time.time() < calibration_frozen_until:
                time.sleep(0.05)
                continue
            calibration_frozen = False

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
        is_cheating = False
        trust_score = 0
        identified_by = "Waiting for alignment..."
        mode = "calibrating" if baseline is None else "posture"
        calibration_ready = False
        face_inside_ratio = 0.0
        suggestion = "Align your face inside the circle."

        # Draw default oval only during manual calibration mode.
        if baseline is None:
            cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 255, 255), 2)

        # ---------------------------
        # NO FACE
        # ---------------------------
        if info is None:
            missing_face_count += 1

            if missing_face_count > MAX_MISSING_FRAMES and baseline is not None:
                baseline = None
                print("🔄 User left → baseline reset")

            cv2.putText(frame, "No face detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            identified_by = "No Face"

        else:
            missing_face_count = 0

            is_inside, ratio = is_face_in_circle(info, center, radius)

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

                if calibration_ready:
                    identified_by = "Capture Ready (green)"
                    suggestion = "Ready. Click 'Capture Reference'."
                    cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 255, 0), 2)
                else:
                    identified_by = "Move face into circle"
                    suggestion = "Move your face to the green zone (>80%)."
                    cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 165, 255), 2)

                cv2.putText(
                    frame,
                    f"Inside: {face_inside_ratio*100:.0f}%",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            # ---------------------------
            # RL TRACKING / EXAM MODE
            # ---------------------------
            else:
                mode = "exam_question" if question_active else ("exam" if exam_active else "posture")

                rl_state = build_state(info, baseline)

                if rl_state:
                    # Detect posture
                    ai_is_bad_posture = is_bad_state(rl_state)

                    # Frontend proctor UI expects `is_cheating` to reflect eye-contact/gaze loss.
                    # `build_state` returns (position, head, gaze) where gaze is "looking" or "away".
                    gaze = rl_state[2]
                    is_cheating = gaze != "looking"

                    position, head, gaze = rl_state

                    # Human-readable suggestion (used in the UI).
                    if position != "centered":
                        suggestion = "Center your face in the frame."
                    elif head != "straight":
                        suggestion = "Keep your head straight."
                    elif gaze != "looking":
                        suggestion = "Look at the camera/screen."
                    else:
                        suggestion = "Good posture. Stay steady."

                    # Identify source
                    if rl_state in Q and np.any(Q[rl_state]):
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
                    if question_active:
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
                    if question_active:
                        if ai_is_bad_posture:
                            posture_stats["bad"] += 1
                        else:
                            posture_stats["good"] += 1

                # UI overlays for training/exam
                cv2.putText(frame, f"State: {rl_state}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.putText(frame, f"Action: {action}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.putText(frame, f"{suggestion}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

                if ai_is_bad_posture:
                    cv2.putText(frame, "FIX POSTURE!", (10, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
                "suggestion": suggestion,
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
    global exam_active, question_active, posture_stats, camera_thread, camera_active
    global question_stats, question_last_state
    global calibration_frozen

    if baseline is None:
        return jsonify({"status": "error", "message": "Please complete calibration first."}), 400

    exam_active = True
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

    calibration_frozen = False

    if not camera_active:
        camera_active = True
        camera_thread = threading.Thread(target=camera_background_task, daemon=True)
        camera_thread.start()
        print("📹 Camera started")

    print("🎯 Exam started")
    return jsonify({"status": "started"})


@app.route("/start_question")
def start_question():
    global question_active, exam_active
    global question_stats, question_last_state
    global camera_thread, camera_active

    if baseline is None:
        return jsonify({"status": "error", "message": "Calibration missing. Capture reference first."}), 400

    exam_active = True
    question_active = True
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

    if not camera_active:
        camera_active = True
        camera_thread = threading.Thread(target=camera_background_task, daemon=True)
        camera_thread.start()
        print("📹 Camera started")

    return jsonify({"status": "question_started"})


@app.route("/end_question")
def end_question():
    global question_active, question_stats, question_last_state

    if not question_active:
        return jsonify({"status": "error", "message": "No active question."}), 400

    question_active = False
    question_last_state = None

    good_frames = question_stats["good"]
    bad_frames = question_stats["bad"]
    total_frames = good_frames + bad_frames
    score = int((good_frames / total_frames) * 100) if total_frames > 0 else 0

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
        "label": label,
        "good_frames": good_frames,
        "bad_frames": bad_frames,
        "reward_total": question_stats["reward_total"],
        "reward_transitions": question_stats["reward_transitions"],
        "errors": errors
    })


@app.route("/end_exam")
def end_exam():
    global exam_active, question_active, posture_stats, camera_active

    exam_active = False
    question_active = False

    total = posture_stats["good"] + posture_stats["bad"]
    score = int((posture_stats["good"] / total) * 100) if total > 0 else 0

    label = "Excellent" if score > 80 else ("Good" if score > 60 else ("Average" if score > 40 else "Poor"))

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
    global exam_active, question_active
    camera_active = False
    exam_active = False
    question_active = False
    
    if cap:
        cap.release()
        cap = None
        print("📷 Camera released")
    
    save_q_table()
    print("💾 Q-table saved")
    
    return jsonify({"status": "stopped"})


@app.route("/calibrate")
def calibrate():
    global baseline, latest_face_info, latest_face_inside_ratio
    global calibration_frozen, calibration_frozen_until
    global exam_active, question_active, posture_stats
    global question_stats, question_last_state

    baseline = None
    latest_face_info = None
    latest_face_inside_ratio = 0.0

    calibration_frozen = False
    calibration_frozen_until = 0.0

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
    global baseline, latest_face_info, latest_face_inside_ratio
    global calibration_frozen, calibration_frozen_until
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

    # If someone tries capturing during an exam, stop scoring.
    question_active = False
    exam_active = False

    calibration = {
        "face_x": baseline.get("face_x"),
        "face_y": baseline.get("face_y"),
        "face_w": baseline.get("face_w"),
        "face_h": baseline.get("face_h"),
        "head_angle": baseline.get("head_angle"),
        "eye_dir": baseline.get("eye_dir"),
        "eye_ratio": baseline.get("eye_ratio"),
        "eye_dist": baseline.get("eye_dist"),
    }

    print("✅ Calibration captured:", calibration)
    return jsonify({"status": "captured", "calibration": calibration})


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