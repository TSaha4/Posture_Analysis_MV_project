import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox

actions = ["no_action", "adjust_posture", "look_screen", "reduce_movement"]

# popup cooldown
time.sleep(0)
last_popup_time = 0
popup_cooldown = 3.0

# Global thresholds for state detection (shown during calibration)
position_threshold = 0.25  # normalized face-width offset
head_angle_threshold = 16  # degrees
eye_dir_threshold = 0.35
ratio_diff_threshold = 0.3

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

    face_w = max(float(info.get("face_w", 1)), 1.0)
    face_h = max(float(info.get("face_h", face_w)), 1.0)
    dx = info["face_x"] - baseline["face_x"]
    normalized_x = dx / face_w
    angle_diff = info["head_angle"] - baseline["head_angle"]

    if abs(normalized_x) < position_threshold:
        position = "centered"
    elif normalized_x < 0:
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


def movement_level(movement, low_threshold=12.0, high_threshold=35.0):
    if movement >= high_threshold:
        return "high"
    if movement >= low_threshold:
        return "medium"
    return "low"


def decide_action(state, movement, gaze_away_counter, movement_threshold=35.0):
    if state is None:
        return "no_action"

    position, head, gaze = state

    if gaze == "away" and gaze_away_counter > 6:
        return "look_screen"

    if head == "tilted":
        return "adjust_posture"

    if position in ["left", "right"]:
        return "adjust_posture"

    # Motion analysis helps catch repeated shifting even when the current
    # discrete posture state still looks acceptable.
    if movement > movement_threshold:
        return "reduce_movement"

    return "no_action"


def show_popup(message):
    global last_popup_time
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Posture Monitor", message)
    last_popup_time = time.time()
    root.destroy()


def is_face_in_circle(face_info, frame_center, circle_radius, ellipse_axes=None):
    if face_info is None:
        return False, 0.0

    face_x = face_info["face_x"]
    face_y = face_info["face_y"]
    face_w = face_info["face_w"]
    face_h = face_info["face_h"]

    if ellipse_axes is not None:
        axis_x = max(float(ellipse_axes[0]), 1.0)
        axis_y = max(float(ellipse_axes[1]), 1.0)
        # Less strict sampling: use a smaller inset so "mostly inside" is achievable.
        inset_x = face_w * 0.1
        inset_y = face_h * 0.1
        rect_left = face_x - face_w / 2 + inset_x
        rect_right = face_x + face_w / 2 - inset_x
        rect_top = face_y - face_h / 2 + inset_y
        rect_bottom = face_y + face_h / 2 - inset_y

        sample_points = [
            (face_x, face_y),
            (face_x, rect_top),
            (face_x, rect_bottom),
            (rect_left, face_y),
            (rect_right, face_y),
            (rect_left, rect_top),
            (rect_left, rect_bottom),
            (rect_right, rect_top),
            (rect_right, rect_bottom),
        ]

        inside_count = 0
        for px, py in sample_points:
            nx = (px - frame_center[0]) / axis_x
            ny = (py - frame_center[1]) / axis_y
            if (nx * nx) + (ny * ny) <= 1.0:
                inside_count += 1

        ratio_inside = inside_count / float(len(sample_points))
        # Lower the cutoff slightly to avoid "stuck at ~50-60%" behavior.
        return ratio_inside >= 0.5, ratio_inside

    # Fallback for legacy circle-based callers.
    dx = face_x - frame_center[0]
    dy = face_y - frame_center[1]
    dist = np.sqrt(dx * dx + dy * dy)
    face_radius = np.sqrt((face_w / 2) ** 2 + (face_h / 2) ** 2)
    # Less strict margin: allow more of the face to extend outside the circle/oval.
    required_radius = dist + face_radius * 0.3
    ratio_inside = (circle_radius - dist) / face_radius if face_radius > 0 else 0
    ratio_inside = max(0, min(1, ratio_inside))

    is_inside = required_radius <= circle_radius
    return is_inside, ratio_inside
