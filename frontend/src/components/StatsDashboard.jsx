const StatsDashboard = ({ stats }) => {
  return (
    <div className="card">
      <h3>Stats</h3>
      <p>Reward: {stats.reward}</p>
      <p>State: {JSON.stringify(stats.state)}</p>
      <p>Action: {stats.action}</p>
    </div>
  );
};

export default StatsDashboard;