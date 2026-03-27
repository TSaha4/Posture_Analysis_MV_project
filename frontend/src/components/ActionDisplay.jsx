const ActionDisplay = ({ action }) => {
  return (
    <div className="card">
      <h3>Action</h3>
      <p>{action || "None"}</p>
    </div>
  );
};

export default ActionDisplay;