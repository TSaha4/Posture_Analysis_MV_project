const StatsDashboard = ({ stats }) => {
  const state = Array.isArray(stats.state) ? stats.state : [];
  const position = stats.position || state[0] || "N/A";
  const head = stats.head || state[1] || "N/A";
  const gaze = stats.gaze || state[2] || "N/A";

  return (
    <div className="card">
      <h3>Machine Vision Stats</h3>
      <p>Position: {position === "centered" ? "center" : position}</p>
      <p>Head: {head}</p>
      <p>Gaze: {gaze}</p>
      <p>Suggestion: {stats.action || "None"}</p>
    </div>
  );
};

export default StatsDashboard;
