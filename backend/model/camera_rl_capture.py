import cv2
try:
    from .camera_rl_utils import detect_face_info, is_face_in_circle
except ImportError:  # pragma: no cover
    from backend.model.camera_rl_utils import detect_face_info, is_face_in_circle


def capture_initial_reference(cap):
    print("Position your face in the green circle and press 'c' to capture reference.")
    while True:
        ret, frame = cap.read()
        if not ret:
            return None

        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 4
        axes = (int(radius * 1.0), int(radius * 1.28))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        info = detect_face_info(gray)
        
        circle_color = (0, 255, 0)  # default green
        if info is not None:
            is_inside, ratio = is_face_in_circle(info, center, radius, ellipse_axes=axes)
            if is_inside:
                circle_color = (0, 255, 0)  # green if inside
            else:
                circle_color = (0, 165, 255)  # orange if partially inside

        cv2.ellipse(frame, center, axes, 0, 0, 360, circle_color, 2)
        cv2.putText(frame, "Align face inside oval and press 'c'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if info is not None:
            is_inside, ratio = is_face_in_circle(info, center, radius, ellipse_axes=axes)
            cv2.putText(frame, f"Face inside: {ratio:.1%}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("RL cam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if info is None:
                print("No face detected. Reposition and retry.")
                continue
            is_inside, ratio = is_face_in_circle(info, center, radius, ellipse_axes=axes)
            if not is_inside:
                print(f"Face not fully in circle (only {ratio:.1%} inside). Reposition and retry.")
                continue
            print("Reference captured: ", info)
            return info

        if key == ord("q"):
            return None
