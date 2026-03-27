import { useEffect, useState } from "react";
import axios from "axios";

const API = "/api";

const useRLData = (isActive) => {
  const [data, setData] = useState({
    connected: false,
    state: null,
    action: null,
    reward: 0,
    is_bad: false,
    is_cheating: false,
    trust_score: 0,
    identified_by: "Initializing...",
    mode: "idle",
    calibration_ready: false,
    face_inside_ratio: 0,
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