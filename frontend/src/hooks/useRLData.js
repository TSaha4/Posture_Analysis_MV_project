import { useEffect, useState } from "react";
import axios from "axios";

const API = "http://127.0.0.1:8000";

const useRLData = (isActive) => {
  const [data, setData] = useState({
    connected: false,
    state: null,
    action: null,
    reward: 0,
    is_bad: false,
    is_cheating: false,
    trust_score: 100,
    identified_by: "Initializing...",
    mode: "idle",
    calibration_ready: false,
    face_inside_ratio: 0,
    face_width: 0,
    face_height: 0,
    head_angle: 0.0,
    eye_dir: 0.0,
    eye_ratio: 0.0,
    eye_distance: 0.0,
    calibration_snapshot: null,
    calibration_frozen: false,
    calibration_freeze_remaining: 0.0,
    suggestion: "Waiting..."
  });

  useEffect(() => {
    if (!isActive) return;

    const interval = setInterval(() => {
      axios
        .get(`${API}/state`, { timeout: 2000 })
        .then((res) => setData({ connected: true, ...res.data }))
        .catch((err) => {
          console.error("Error fetching RL data:", err);
          setData((prev) => ({ ...prev, connected: false }));
        });
    }, 500);

    return () => clearInterval(interval);
  }, [isActive]);

  return data;
};

export default useRLData;
