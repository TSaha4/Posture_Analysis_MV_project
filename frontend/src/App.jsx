import React, { useState } from "react";
import axios from "axios";
import useRLData from "./hooks/useRLData";
import hrQuestions from "./data/hrQuestions.json";
import "./App.css";

function App() {
  const [view, setView] = useState("home");
  const [questionIndex, setQuestionIndex] = useState(0);
  
  const isActive = view !== "home";
  const data = useRLData(isActive);

  const handleExit = async () => {
    try {
      await axios.get("http://localhost:8000/stop_session");
    } catch (err) {
      console.error("Error stopping session:", err);
    }
    setView("home");
    setQuestionIndex(0);
  };

  if (view === "home") {
    return (
      <div className="welcome-screen">
        <div className="bg-glow blue"></div>
        <div className="bg-glow purple"></div>

        <div className="hero-section">
          <div className="badge-chip">AI-Powered Interview Coach</div>
          <h2 className="glitch-text">Professional Presence Studio</h2>
          <p className="hero-subtitle">Master your non-verbal cues and integrity before the high-stakes call.</p>

          <div className="mode-selector-grid">
            <div className="mode-card glass-morphism" onClick={() => setView("posture")}>
              <div className="icon-wrapper">🧘‍♂️</div>
              <h3>Body Language Prep</h3>
              <p>Calibrate your "Interview Stance." Train the AI to recognize your most confident, professional alignment.</p>
              <button className="start-btn posture-btn">Start Calibration</button>
            </div>

            <div className="mode-card glass-morphism proctor-card" onClick={() => setView("proctor")}>
              <div className="icon-wrapper">🛡️</div>
              <h3>Mock Interview Room</h3>
              <p>Answer behavioral questions while the AI monitors gaze integrity and professional poise.</p>
              <button className="start-btn proctor-btn">Enter Exam Room</button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <header className="header">
        <div className="header-title">
          <h1>{view === "proctor" ? "🛡️ Live Mock Interview" : "🔵 Body Language Calibration"}</h1>
          <span className="status-badge active">🟢 COACHING LIVE</span>
        </div>
        <button className="toggle-btn btn-danger" onClick={handleExit}>End Session</button>
      </header>

      <div className="main-content">
        {view === "proctor" ? (
          <ProctorPage data={data} idx={questionIndex} setIdx={setQuestionIndex} />
        ) : (
          <PosturePage data={data} />
        )}
      </div>
    </div>
  );
}

// --- PAGE: POSTURE TRAINING (With Action Card) ---
const PosturePage = ({ data }) => (
  <>
    {/* LEFT: VIDEO */}
    <div className="left-panel">
      <div className="video-frame-container full-height">
        <VideoFeed />
        <button className="calibrate-btn" onClick={async () => {
          try {
            await axios.get("http://localhost:8000/calibrate");
            alert("Calibration reset. Position yourself and wait for auto-calibration.");
          } catch (err) {
            console.error("Error resetting calibration:", err);
          }
        }}>Reset Calibration</button>
      </div>
    </div>

    {/* RIGHT: SIDEBAR (ONLY 3 COMPONENTS) */}
    <div className="dashboard-sidebar">

      {/* 1. STATUS */}
      <StatusPanel data={data} />

      {/* 2. ACTION */}
      <ActionDisplay action={data?.action} />

      {/* 3. STATS */}
      <StatsDashboard stats={data} />

    </div>
  </>
);

// --- PAGE: PROCTORED EXAM (Action Card Hidden) ---
const ProctorPage = ({ data, idx, setIdx }) => (
  <>
    <div className="left-panel proctor-layout">
      <div className="row-question">
        {data.is_cheating && <div className="cheating-banner-mini">⚠️ EYE CONTACT LOST: LOOK AT THE CAMERA</div>}
        <InterviewPrompter idx={idx} setIdx={setIdx} />
      </div>
      <div className="row-video">
        <div className="video-frame-container">
          <VideoFeed />
        </div>
      </div>
    </div>
    <div className="dashboard-sidebar">
      <div className="card trust-card-pro" style={{ borderBottom: data.trust_score < 70 ? '4px solid #f85149' : '4px solid #3fb950' }}>
        <h3 className="card-label">Integrity & Gaze Score</h3>
        <h1 className={`score-number ${data.trust_score < 70 ? 'critical' : 'stable'}`}>{data.trust_score}%</h1>
        <div className="trust-bar-bg">
          <div className="trust-bar-fill" style={{ width: `${data.trust_score}%`, backgroundColor: data.trust_score < 70 ? '#f85149' : '#3fb950' }}></div>
        </div>
      </div>
      <StatusPanel data={data} title="Posture Integrity" />
      <StatsDashboard stats={data} />
    </div>
  </>
);

// --- COMPONENTS ---

export const ActionDisplay = ({ action }) => (
  <div className="card sidebar-card action-card-highlight">
    <h3>AI Correction</h3>
    <h2 className="highlight-text">{action || "Analyzing..."}</h2>
    <p className="mini-desc">Real-time guidance from the RL model.</p>
  </div>
);

export const StatusPanel = ({ data, title }) => {
  const isGood = !data?.is_bad;
  return (
    <div className={`card sidebar-card ${isGood ? 'border-success' : 'border-danger'}`} style={{ flex: 1 }}>
      <h3>{title || "Status"}</h3>
      <h2 style={{ color: isGood ? '#3fb950' : '#f85149', margin: '10px 0' }}>
        {isGood ? "✅ Interview Ready" : "⚠️ Fix Presence"}
      </h2>
      <div className="advice-box">
        {isGood ? "Professional alignment detected. Maintain this level of poise." : "You appear to be slouching or off-center. Recenter for impact."}
      </div>
    </div>
  );
};

export const InterviewPrompter = ({ idx, setIdx }) => {
  const q = hrQuestions[idx];
  const progress = ((idx + 1) / hrQuestions.length) * 100;
  return (
    <div className="card exam-card">
      <div className="exam-header">
        <span className="exam-tag">BEHAVIORAL Q {idx + 1} / {hrQuestions.length}</span>
        <div className="exam-progress-bg"><div className="exam-progress-fill" style={{width: `${progress}%`}}></div></div>
      </div>
      <h2 className="exam-question-text">"{q.question}"</h2>
      <div className="exam-controls">
        <button onClick={() => setIdx(idx - 1)} disabled={idx === 0} className="exam-btn secondary">Prev</button>
        <button onClick={() => setIdx(idx + 1)} disabled={idx === hrQuestions.length - 1} className="exam-btn primary">Next</button>
      </div>
    </div>
  );
};

export const StatsDashboard = ({ stats }) => (
  <div className="card sidebar-card" style={{ flex: 1 }}>
    <h3>Poise Analytics</h3>
    <p>Reward: <span style={{ color: stats.reward >= 0 ? '#3fb950' : '#f85149' }}>{stats.reward}</span></p>
    <div className="vector-grid">
      {stats.state?.map((s, i) => <div key={i} className="state-box">{s}</div>)}
    </div>
  </div>
);

export const VideoFeed = () => (
  <img src="http://localhost:8000/video_feed" alt="Live Feed" className="video-element" />
);

export default App;

//App.jsx