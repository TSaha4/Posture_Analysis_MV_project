import cv2
import time

from backend.camera_rl_utils import (
    detect_face_info,
    build_state,
    is_bad_state,
    choose_action,
    save_q_table,
    load_q_table,
    reward_for_transition,
    Q,
    actions,
    alpha,
    gamma,
    min_epsilon,
    epsilon_decay,
)

from backend.camera_rl_capture import capture_initial_reference


def main():
    epsilon = 0.25
    load_q_table()

    # macOS-safe camera
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # ---------------------------
    # CALIBRATION
    # ---------------------------
    initial_info = capture_initial_reference(cap)

    if initial_info is None:
        print("Calibration failed")
        cap.release()
        return

    baseline = initial_info.copy()
    print("✅ Calibration done")

    # ---------------------------
    # MAIN LOOP
    # ---------------------------
    last_valid_state = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        info = detect_face_info(gray)

        state = build_state(info, baseline)

        # 🔥 FIX: reuse last valid state (no flicker)
        if state is None:
            state = last_valid_state

        if state is None:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("RL cam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        last_valid_state = state

        action_index = choose_action(state, epsilon)
        action = actions[action_index]

        next_state = state
        reward = reward_for_transition(state, next_state)

        next_max = max(Q[next_state]) if next_state in Q else 0
        Q[state][action_index] += alpha * (
            reward + gamma * next_max - Q[state][action_index]
        )

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # ---------------------------
        # UI (NO POPUPS)
        # ---------------------------
        cv2.putText(frame, f"State: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Action: {action}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Reward: {reward}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if is_bad_state(state):
            cv2.putText(frame, "FIX POSTURE!", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("RL cam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    save_q_table()
    cap.release()
    cv2.destroyAllWindows()
    print("Q-table saved")


if __name__ == "__main__":
    main()