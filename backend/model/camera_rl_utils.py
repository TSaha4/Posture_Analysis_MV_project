import cv2
import numpy as np
import time
import os
import pickle
import tkinter as tk
from tkinter import messagebox
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

# popup cooldown
time.sleep(0)
last_popup_time = 0
popup_cooldown = 3.0

# Global thresholds for state detection (shown during calibration)
position_threshold = 0.15  # % of face width
head_angle_threshold = 10  # degrees
eye_dir_threshold = 0.2
ratio_diff_threshold = 0.14

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def detect_face_info(gray_frame):
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face_cx = x + w / 2
    face_cy = y + h / 2
    roi_gray = gray_frame[y : y + h, x : x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

    angle = 0.0
    eye_dir = 0.0
    eye_dist = 0.0
    eye_ratio = None
    eyes_detected = False

    if len(eyes) >= 2:
        eyes_detected = True
        eyes = sorted(eyes, key=lambda e: -e[2])[:2]
        eye_centers = []
        for (ex, ey, ew, eh) in eyes:
            eye_centers.append((x + ex + ew / 2, y + ey + eh / 2))

        (x1, y1), (x2, y2) = eye_centers
        dx = x2 - x1
        dy = y2 - y1
        eye_dist = np.sqrt(dx * dx + dy * dy)
        eye_ratio = eye_dist / float(w) if w > 0 else 0.0

        if abs(dx) > 1e-3:
            angle = np.degrees(np.arctan2(dy, dx))

        average_eye_cx = (x1 + x2) / 2
        eye_dir = (average_eye_cx - face_cx) / w
    elif len(eyes) == 1:
        # fallback: estimate direction from single eye
        ex, ey, ew, eh = eyes[0]
        eyes_detected = True
        eye_center_x = x + ex + ew / 2
        eye_dir = (eye_center_x - face_cx) / w
        eye_dist = 0.0
        eye_ratio = 0.0
    else:
        # no eyes
        eyes_detected = False
        eye_dir = 0.0
        eye_dist = 0.0
        eye_ratio = None

    return {
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

    if info.get("eyes_detected", False) and baseline.get("eye_ratio") is not None and info.get("eye_ratio") is not None:
        eye_diff = abs(info["eye_dir"] - baseline["eye_dir"])
        ratio_diff = abs(info["eye_ratio"] - baseline["eye_ratio"])
        gaze = "looking" if eye_diff < eye_dir_threshold and ratio_diff < ratio_diff_threshold else "away"
    else:
        # fallback: if posture is good, assume looking
        gaze = "looking" if position == "centered" and head == "straight" else "away"

    return (position, head, gaze)


def is_bad_state(state):
    if state is None:
        return True
    position, head, gaze = state
    return position != "centered" or head != "straight" or gaze != "looking"


def badness(state):
    if state is None:
        return 3
    p, h, g = state
    score = (0 if p == "centered" else 1) + (0 if h == "straight" else 1)
    score += 0 if g == "looking" else 0.5
    return score


def reward_for_transition(state, next_state):
    if next_state is None:
        return -2
    
    current_bad = badness(state)
    next_bad = badness(next_state)
    
    # Perfect state achieved
    if next_bad == 0:
        return 2
    
    # Improvement detected
    if next_bad < current_bad:
        return 3
    
    # Maintaining a good state (badness <= 1 means at most 1 thing wrong)
    if next_bad <= 1:
        return 1
    
    # Worsening state
    if next_bad > current_bad:
        return -2
    
    # Stuck in bad state but not worsening
    return -1


def choose_action(state, epsilon_val):
    if state is None:
        return 0
    if np.random.rand() < epsilon_val:
        return np.random.randint(len(actions))
    return int(np.argmax(Q[state]))


def show_popup(message):
    global last_popup_time
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Posture RL", message)
    last_popup_time = time.time()
    root.destroy()


def save_q_table():
    with open(Q_TABLE_PATH, "wb") as f:
        pickle.dump(dict(Q), f)


def load_q_table():
    if os.path.exists(Q_TABLE_PATH):
        with open(Q_TABLE_PATH, "rb") as f:
            data = pickle.load(f)
            for k, v in data.items():
                Q[k] = v


def is_face_in_circle(face_info, frame_center, circle_radius):
    if face_info is None:
        return False, 0.0
    
    face_x = face_info["face_x"]
    face_y = face_info["face_y"]
    face_w = face_info["face_w"]
    face_h = face_info["face_h"]
    
    # Distance from face center to circle center
    dx = face_x - frame_center[0]
    dy = face_y - frame_center[1]
    dist = np.sqrt(dx * dx + dy * dy)
    
    # Face radius (half diagonal of face)
    face_radius = np.sqrt((face_w / 2) ** 2 + (face_h / 2) ** 2)
    
    # Check if majority of face is inside circle (face_center + face_radius <= circle_radius)
    required_radius = dist + face_radius * 0.6  # Allow some tolerance
    ratio_inside = (circle_radius - dist) / face_radius if face_radius > 0 else 0
    ratio_inside = max(0, min(1, ratio_inside))
    
    is_inside = required_radius <= circle_radius
    return is_inside, ratio_inside