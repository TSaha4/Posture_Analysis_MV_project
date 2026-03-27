const StatusPanel = ({ state }) => {
  if (!state) return <div>Loading state...</div>;

  return (
    <div className="card">
      <h3>State</h3>
      <p>{JSON.stringify(state)}</p>
    </div>
  );
};

export default StatusPanel;