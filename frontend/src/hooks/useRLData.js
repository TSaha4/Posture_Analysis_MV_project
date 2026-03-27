import { useEffect, useState } from "react";
import axios from "axios";

const useRLData = (isActive) => {
  const [data, setData] = useState({
    state: null,
    action: null,
    reward: 0,
    alert: ""
  });

  useEffect(() => {
    // If the session is not active, do not ping the backend
    if (!isActive) return;

    const interval = setInterval(() => {
      axios.get("http://localhost:8000/state")
        .then(res => setData(res.data))
        .catch(err => console.error("Error fetching RL data:", err));
    }, 500);

    return () => clearInterval(interval);
  }, [isActive]); // Re-run this effect if isActive changes

  return data;
};

export default useRLData;