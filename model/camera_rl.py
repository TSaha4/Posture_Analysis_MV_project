import cv2
import time
import os

from camera_rl_utils import (
    detect_face_info,
    build_state,
    is_bad_state,
    choose_action,
    show_popup,
    save_q_table,
    load_q_table,
    reward_for_transition,
    is_face_in_circle,
    Q,
    actions,
    alpha,
    gamma,
    min_epsilon,
    epsilon_decay,
    popup_cooldown,
    position_threshold,
    head_angle_threshold,
    eye_dir_threshold,
    ratio_diff_threshold,
)
from camera_rl_capture import capture_initial_reference


def main():
    epsilon = 0.25
    load_q_table()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    initial_info = capture_initial_reference(cap)
    if initial_info is None:
        print("Reference capture canceled")
        cap.release()
        cv2.destroyAllWindows()
        return

    baseline = initial_info.copy()
    calibration_frames = [initial_info]
    calibration_target = 80
    start_time = time.time()

    while len(calibration_frames) < calibration_target and time.time() - start_time < 12:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        info = detect_face_info(gray)
        if info is not None:
            calibration_frames.append(info)

        cv2.putText(frame, f"Calibrating {len(calibration_frames)}/{calibration_target}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show thresholds
        cv2.putText(frame, f"Pos thresh: {position_threshold:.2f} | Head: {head_angle_threshold}deg | Eye: {eye_dir_threshold:.2f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Ratio thresh: {ratio_diff_threshold:.2f}", 
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show running average of calibration ratios
        if len(calibration_frames) > 0:
            avg_angle = sum(f["head_angle"] for f in calibration_frames) / len(calibration_frames)
            avg_eye_dir = sum(f["eye_dir"] for f in calibration_frames) / len(calibration_frames)
            avg_ratio = sum((f.get("eye_ratio") or 0) for f in calibration_frames) / len(calibration_frames)
            
            cv2.putText(frame, f"Avg angle: {avg_angle:.2f}deg | Avg eye_dir: {avg_eye_dir:.3f} | Avg ratio: {avg_ratio:.3f}", 
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("RL cam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if len(calibration_frames) > 0:
        baseline = {
            "face_x": sum(f["face_x"] for f in calibration_frames) / len(calibration_frames),
            "face_y": sum(f["face_y"] for f in calibration_frames) / len(calibration_frames),
            "head_angle": sum(f["head_angle"] for f in calibration_frames) / len(calibration_frames),
            "eye_dir": sum(f["eye_dir"] for f in calibration_frames) / len(calibration_frames),
            "eye_ratio": sum((f.get("eye_ratio") or 0) for f in calibration_frames) / len(calibration_frames),
        }
        print("\n" + "="*70)
        print("CALIBRATION DONE - Baseline & Thresholds")
        print("="*70)
        print(f"Baseline Position:  face_x={baseline['face_x']:.1f}, face_y={baseline['face_y']:.1f}")
        print(f"Baseline Head Angle: {baseline['head_angle']:.2f}°")
        print(f"Baseline Eye Dir:    {baseline['eye_dir']:.3f}")
        print(f"Baseline Eye Ratio:  {baseline['eye_ratio']:.3f}")
        print(f"\nState Detection Thresholds:")
        print(f"  - Position threshold:    ±{position_threshold*100:.0f}% of face width")
        print(f"  - Head angle threshold:  ±{head_angle_threshold}°")
        print(f"  - Eye direction thresh:  {eye_dir_threshold:.2f}")
        print(f"  - Ratio diff threshold:  {ratio_diff_threshold:.2f}")
        print("="*70 + "\n")
    else:
        print("Calibration failed")
        cap.release()
        cv2.destroyAllWindows()
        return

    prev_state = None
    prev_action = None
    last_popup = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        info = detect_face_info(gray)
        state = build_state(info, baseline)

        if state is None:
            cv2.putText(frame, "No face found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "PAUSED - NO FACE DETECTED", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("RL cam", frame)

            if time.time() - last_popup > popup_cooldown:
                show_popup("No face detected. Please position your face in the circle.")
                last_popup = time.time()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        action_index = choose_action(state, epsilon)
        action = actions[action_index]
        next_state = state

        if is_bad_state(state) and (time.time() - last_popup > popup_cooldown):
            popup_text = {
                "adjust_posture": "Sit straight!",
                "look_screen": "Look at screen!",
                "reduce_movement": "Stay steady!",
            }.get(action, "Please fix your posture!")

            cv2.putText(frame, "PAUSED - FIX POSTURE", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("RL cam", frame)
            cv2.waitKey(1)

            show_popup(popup_text)
            last_popup = time.time()

            ret2, frame2 = cap.read()
            if ret2:
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                info2 = detect_face_info(gray2)
                next_state = build_state(info2, baseline) or state

        reward = reward_for_transition(state, next_state)

        next_max = max(Q[next_state]) if next_state in Q else 0
        Q[state][action_index] += alpha * (reward + gamma * next_max - Q[state][action_index])

        print(f"STATE: {state} | ACTION: {action} | NEXT_STATE: {next_state} | REWARD: {reward}")
        print("-------------------------")

        prev_state = state
        prev_action = action_index

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        cv2.putText(frame, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Action: {action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Reward: {reward}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("RL cam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    save_q_table()
    cap.release()
    cv2.destroyAllWindows()
    print("Q-table saved")


if __name__ == "__main__":
    main()
