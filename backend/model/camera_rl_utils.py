import cv2
import numpy as np
import time
import os
import pickle
from collections import defaultdict

# Paths
Q_TABLE_PATH = os.path.join(os.path.dirname(__file__), "q_table.pkl")

# RL setup
actions = ["no_action", "adjust_posture", "look_screen", "reduce_movement"]
Q = defaultdict(lambda: np.zeros(len(actions), dtype=np.float32))

alpha = 0.1
gamma = 0.9
epsilon = 0.25
min_epsilon = 0.01
epsilon_decay = 0.9995

# popup cooldown (removed tkinter usage)
last_popup_time = 0
popup_cooldown = 3.0

# Thresholds
position_threshold = 0.15
head_angle_threshold = 10
eye_dir_threshold = 0.2
ratio_diff_threshold = 0.14

# Haar cascades (TUNED)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# 🔥 NEW: memory to reduce flicker
last_face_info = None


# ---------------------------
# FACE DETECTION (STABLE)
# ---------------------------
def detect_face_info(gray_frame):
    global last_face_info

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(120, 120)
    )

    # If no face detected → reuse last
    if len(faces) == 0:
        return last_face_info

    x, y, w, h = faces[0]
    face_cx = x + w / 2
    face_cy = y + h / 2

    roi_gray = gray_frame[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(25, 25)
    )

    angle = 0.0
    eye_dir = 0.0
    eye_dist = 0.0
    eye_ratio = None
    eyes_detected = False

    if len(eyes) >= 2:
        eyes_detected = True
        eyes = sorted(eyes, key=lambda e: -e[2])[:2]

        (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes

        x1, y1 = x + ex1 + ew1 / 2, y + ey1 + eh1 / 2
        x2, y2 = x + ex2 + ew2 / 2, y + ey2 + eh2 / 2

        dx = x2 - x1
        dy = y2 - y1

        eye_dist = np.sqrt(dx * dx + dy * dy)
        eye_ratio = eye_dist / float(w)

        if abs(dx) > 1e-3:
            angle = np.degrees(np.arctan2(dy, dx))

        avg_eye_x = (x1 + x2) / 2
        eye_dir = (avg_eye_x - face_cx) / w

    elif len(eyes) == 1:
        ex, ey, ew, eh = eyes[0]
        eyes_detected = True

        eye_center_x = x + ex + ew / 2
        eye_dir = (eye_center_x - face_cx) / w

        eye_ratio = 0.0
        eye_dist = 0.0

    else:
        eyes_detected = False
        eye_ratio = None

    info = {
        "face_x": face_cx,
        "face_y": face_cy,
        "head_angle": angle,
        "eye_dir": eye_dir,
        "face_w": w,
        "face_h": h,
        "eye_dist": eye_dist,
        "eye_ratio": eye_ratio,
        "eyes_detected": eyes_detected,
    }

    # Save last valid
    last_face_info = info

    return info


# ---------------------------
# STATE BUILDING
# ---------------------------
def build_state(info, baseline):
    if info is None or baseline is None:
        return None

    dx = info["face_x"] - baseline["face_x"]
    dy = info["face_y"] - baseline["face_y"]
    angle_diff = info["head_angle"] - baseline["head_angle"]

    w = info.get("face_w", 1)

    if abs(dx) < position_threshold * w and abs(dy) < position_threshold * w:
        position = "centered"
    elif dx < 0:
        position = "left"
    else:
        position = "right"

    head = "straight" if abs(angle_diff) < head_angle_threshold else "tilted"

    if info.get("eyes_detected") and baseline.get("eye_ratio") is not None:
        eye_diff = abs(info["eye_dir"] - baseline["eye_dir"])
        ratio_diff = abs(info["eye_ratio"] - baseline["eye_ratio"])
        gaze = "looking" if eye_diff < eye_dir_threshold and ratio_diff < ratio_diff_threshold else "away"
    else:
        gaze = "looking" if position == "centered" and head == "straight" else "away"

    return (position, head, gaze)


# ---------------------------
# RL HELPERS
# ---------------------------
def is_bad_state(state):
    if state is None:
        return True
    p, h, g = state
    return p != "centered" or h != "straight" or g != "looking"


def badness(state):
    if state is None:
        return 3
    p, h, g = state
    return (p != "centered") + (h != "straight") + (0 if g == "looking" else 0.5)


def reward_for_transition(state, next_state):
    if next_state is None:
        return -2

    current_bad = badness(state)
    next_bad = badness(next_state)

    if next_bad == 0:
        return 2
    if next_bad < current_bad:
        return 3
    if next_bad <= 1:
        return 1
    if next_bad > current_bad:
        return -2

    return -1


def choose_action(state, epsilon_val):
    if state is None:
        return 0
    if np.random.rand() < epsilon_val:
        return np.random.randint(len(actions))
    return int(np.argmax(Q[state]))


# ---------------------------
# SAVE / LOAD
# ---------------------------
def save_q_table():
    with open(Q_TABLE_PATH, "wb") as f:
        pickle.dump(dict(Q), f)


def load_q_table():
    if os.path.exists(Q_TABLE_PATH):
        with open(Q_TABLE_PATH, "rb") as f:
            data = pickle.load(f)
            for k, v in data.items():
                Q[k] = v


# ---------------------------
# FACE INSIDE CIRCLE
# ---------------------------
def is_face_in_circle(face_info, frame_center, circle_radius):
    if face_info is None:
        return False, 0.0

    dx = face_info["face_x"] - frame_center[0]
    dy = face_info["face_y"] - frame_center[1]

    dist = np.sqrt(dx * dx + dy * dy)

    face_radius = np.sqrt(
        (face_info["face_w"] / 2) ** 2 +
        (face_info["face_h"] / 2) ** 2
    )

    ratio_inside = (circle_radius - dist) / face_radius if face_radius > 0 else 0
    ratio_inside = max(0, min(1, ratio_inside))

    is_inside = ratio_inside > 0.55  # 🔥 relaxed condition

    return is_inside, ratio_inside