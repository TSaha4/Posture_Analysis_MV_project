import cv2
import time
from backend.camera_rl_utils import detect_face_info, is_face_in_circle

def capture_initial_reference(cap):
    print("Align your face inside the circle...")

    stable_count = 0
    required_stable_frames = 8
    start_time = time.time()

    best_info = None
    best_ratio = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        info = detect_face_info(gray)

        if info is not None:
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            radius = min(w, h) // 4

            is_inside, ratio = is_face_in_circle(info, center, radius)

            # Track best frame
            if ratio > best_ratio:
                best_ratio = ratio
                best_info = info

            if ratio > 0.65:   # relaxed condition
                stable_count += 1
                print(f"✔ Good ({ratio:.1%}) | Stable: {stable_count}/{required_stable_frames}")
            else:
                stable_count = max(0, stable_count - 1)
                print(f"⚠ Adjust ({ratio:.1%})")

            if stable_count >= required_stable_frames:
                print("✅ Reference captured:", info)
                return info

        else:
            stable_count = max(0, stable_count - 1)
            print("❌ No face")

        # Timeout fallback (IMPORTANT)
        if time.time() - start_time > 8:
            print("⏱ Timeout — using best frame")
            return best_info