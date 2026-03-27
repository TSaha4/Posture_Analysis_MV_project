import React, { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import useRLData from "./hooks/useRLData";
import hrQuestions from "./data/hrQuestions.json";
import "./App.css";

const API = "/api";
const QUESTION_SECONDS = 180;

function App() {
  const [screen, setScreen] = useState("home");
  const [toast, setToast] = useState("");
  const data = useRLData(screen !== "home");
  const backendOnline = Boolean(data?.connected);
  const streamSrc = useMemo(() => `${API}/video_feed?screen=${screen}`, [screen]);

  const stopAll = async () => {
    try {
      await axios.get(`${API}/end_exam`);
    } catch (err) {
      console.debug("end_exam skipped:", err?.message);
    }
    try {
      await axios.get(`${API}/stop_session`);
    } catch (err) {
      console.debug("stop_session skipped:", err?.message);
    }
  };

  const leaveSession = async () => {
    await stopAll();
    setScreen("home");
  };

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 3000);
    return () => clearTimeout(t);
  }, [toast]);

  useEffect(() => {
    if (screen !== "calibration") return;
    // Always start calibration in a clean state so capture can become active.
    axios
      .get(`${API}/calibrate`)
      .catch((err) => console.debug("calibrate on enter failed:", err?.message));
  }, [screen]);

  if (screen === "home") {
    return (
      <div className="welcome-screen">
        <div className="bg-glow blue" />
        <div className="bg-glow purple" />
        <div className="hero-section">
          <div className="badge-chip">RL Interview Pipeline</div>
          <h2 className="glitch-text">Calibration + Timed Exam</h2>
          <p className="hero-subtitle">
            Capture a strict calibration reference first, then run a timed 3-question mock interview with live posture scoring.
          </p>

          <div className="mode-selector-grid">
            <div className="glass-morphism mode-card" onClick={() => setScreen("calibration")}>
              <div className="icon-wrapper">🎯</div>
              <h3>Calibration Room</h3>
              <p>Align face in circle, capture reference, freeze frame, start posture baseline.</p>
              <button className="start-btn posture-btn">Go to Calibration</button>
            </div>

            <div className="glass-morphism mode-card" onClick={() => setScreen("exam")}>
              <div className="icon-wrapper">🛡️</div>
              <h3>Exam Room</h3>
              <p>Random 3 questions, 3 minutes each, start-answer timer, score with posture errors.</p>
              <button className="start-btn proctor-btn">Go to Exam</button>
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
          <h1>{screen === "calibration" ? "🎯 Calibration Room" : "🛡️ Exam Room"}</h1>
          <span className="status-badge active">Backend: {data?.mode || "loading"}</span>
        </div>
        <div className="exam-controls">
          <button className="exam-btn secondary" onClick={() => setScreen(screen === "calibration" ? "exam" : "calibration")}>
            {screen === "calibration" ? "Go Exam" : "Go Calibration"}
          </button>
          <button className="toggle-btn btn-danger" onClick={leaveSession}>
            End Session
          </button>
        </div>
      </header>

      {toast ? <div className="cheating-banner-mini">{toast}</div> : null}

      <div className="main-content">
        <div className="left-panel">
          <div className="video-frame-container full-height">
            <img
              key={screen}
              src={streamSrc}
              alt="Live Feed"
              className="video-element"
            />
          </div>
        </div>

        <div className="dashboard-sidebar">
          {!backendOnline ? (
            <div className="card">
              <h3>Backend status</h3>
              <p>Backend seems offline or not responding.</p>
              <p style={{ opacity: 0.85 }}>
                Start the backend on port 8000, then reload this page.
              </p>
            </div>
          ) : null}
          {screen === "calibration" ? (
            <CalibrationPanel data={data} setToast={setToast} />
          ) : (
            <ExamPanel data={data} setToast={setToast} />
          )}
        </div>
      </div>
    </div>
  );
}

function CalibrationPanel({ data, setToast }) {
  const ratioPct = Math.round((data?.face_inside_ratio || 0) * 100);
  const ready = Boolean(data?.calibration_ready);
  const isCalibrating = data?.mode === "calibrating";

  const resetCalibration = async () => {
    try {
      await axios.get(`${API}/calibrate`);
      setToast("Calibration reset. Align face and capture reference.");
    } catch (err) {
      setToast(err?.response?.data?.message || "Could not reset calibration.");
    }
  };

  const captureReference = async () => {
    try {
      await axios.get(`${API}/capture_reference`);
      setToast("Reference captured. Circle removed for posture tracking.");
    } catch (err) {
      setToast(err?.response?.data?.message || "Could not capture reference.");
    }
  };

  return (
    <>
      <div className="card">
        <h3>Calibration Status</h3>
        <p>Face inside circle: <strong>{ratioPct}%</strong></p>
        <p>Capture rule: minimum 80%</p>
        <p>Mode: <strong>{data?.mode || "loading"}</strong></p>
        <p>Ready: <strong>{ready ? "Yes" : "No"}</strong></p>
        {!isCalibrating ? (
          <p style={{ color: "#f2c94c" }}>
            Click <strong>Reset</strong> to return to calibration mode.
          </p>
        ) : null}
      </div>

      <div className="card">
        <h3>Controls</h3>
        <div className="exam-controls">
          <button className="exam-btn secondary" onClick={resetCalibration}>Reset</button>
          <button className="exam-btn primary" onClick={captureReference} disabled={!ready || !isCalibrating}>
            Capture Reference
          </button>
        </div>
      </div>

      <div className="card">
        <h3>Live Suggestion</h3>
        <div className="advice-box">{data?.suggestion || "Waiting for camera data..."}</div>
      </div>

      <LiveTelemetry data={data} showScores={false} />
    </>
  );
}

function ExamPanel({ data, setToast }) {
  const [questions, setQuestions] = useState([]);
  const [questionIdx, setQuestionIdx] = useState(0);
  const [answering, setAnswering] = useState(false);
  const [timeLeft, setTimeLeft] = useState(QUESTION_SECONDS);
  const [result, setResult] = useState(null);
  const [examReady, setExamReady] = useState(false);

  useEffect(() => {
    const shuffled = [...hrQuestions].sort(() => Math.random() - 0.5).slice(0, 3);
    setQuestions(shuffled);
    setQuestionIdx(0);
    setAnswering(false);
    setTimeLeft(QUESTION_SECONDS);
    setResult(null);

    axios
      .get(`${API}/start_exam`)
      .then(() => setExamReady(true))
      .catch((err) => {
        setExamReady(false);
        setToast(err?.response?.data?.message || "Exam requires completed calibration.");
      });
  }, [setToast]);

  useEffect(() => {
    if (!answering) return;
    const t = setInterval(() => {
      setTimeLeft((s) => Math.max(0, s - 1));
    }, 1000);
    return () => clearInterval(t);
  }, [answering]);

  const finalizeQuestion = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/end_question`);
      setResult(res.data);
    } catch (err) {
      setToast(err?.response?.data?.message || "Could not compute question result.");
    } finally {
      setAnswering(false);
    }
  }, [setToast]);

  useEffect(() => {
    if (!answering || timeLeft !== 0) return;
    finalizeQuestion();
  }, [answering, timeLeft, finalizeQuestion]);

  const startAnswer = async () => {
    setResult(null);
    setTimeLeft(QUESTION_SECONDS);
    try {
      await axios.get(`${API}/start_question`);
      setAnswering(true);
    } catch (err) {
      setToast(err?.response?.data?.message || "Could not start answer timer.");
    }
  };

  const nextQuestion = () => {
    setQuestionIdx((q) => Math.min(q + 1, 2));
    setResult(null);
    setTimeLeft(QUESTION_SECONDS);
    setAnswering(false);
  };

  const finishExam = async () => {
    try {
      const res = await axios.get(`${API}/end_exam`);
      setToast(`Final score ${res.data.score}/100 (${res.data.label})`);
    } catch (err) {
      setToast(err?.response?.data?.message || "Could not end exam.");
    }
  };

  const q = questions[questionIdx];
  const timer = useMemo(() => {
    const mm = String(Math.floor(timeLeft / 60)).padStart(2, "0");
    const ss = String(timeLeft % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  }, [timeLeft]);

  return (
    <>
      <div className="card">
        <h3>Question {questionIdx + 1} / 3</h3>
        <p>{q?.question || "Preparing questions..."}</p>
      </div>

      <div className="card">
        <h3>Timer</h3>
        <h1 className={`score-number ${timeLeft <= 20 && answering ? "critical" : "stable"}`}>{timer}</h1>
        <div className="exam-controls">
          <button className="exam-btn primary" onClick={startAnswer} disabled={!examReady || answering || Boolean(result)}>
            Start Answer
          </button>
          <button className="exam-btn secondary" onClick={finalizeQuestion} disabled={!answering}>
            End Now
          </button>
        </div>
      </div>

      <div className="card">
        <h3>Live Suggestion</h3>
        <div className="advice-box">{data?.suggestion || "Waiting..."}</div>
        {data?.is_cheating ? <p style={{ color: "#f85149" }}>Eye contact lost.</p> : null}
      </div>

      {result ? (
        <div className="card">
          <h3>Question Score</h3>
          <h1 className={`score-number ${result.score < 70 ? "critical" : "stable"}`}>{result.score}/100</h1>
          <p>{result.label}</p>
          <div style={{ display: "grid", gap: 6 }}>
            {(result.errors || []).map((e) => (
              <div key={e.key} className="state-box">
                {e.description} ({e.percent_frames}%)
              </div>
            ))}
          </div>
          <div className="exam-controls" style={{ marginTop: 10 }}>
            {questionIdx < 2 ? (
              <button className="exam-btn primary" onClick={nextQuestion}>Next Question</button>
            ) : (
              <button className="exam-btn primary" onClick={finishExam}>Finish Exam</button>
            )}
          </div>
        </div>
      ) : null}

      <LiveTelemetry data={data} showScores />
    </>
  );
}

function LiveTelemetry({ data, showScores }) {
  const isCalibrating = data?.mode === "calibrating";
  return (
    <div className="card">
      <h3>RL Telemetry</h3>
      <p>State: {Array.isArray(data?.state) ? data.state.join(" | ") : "N/A"}</p>
      <p>Action: {data?.action || "N/A"}</p>
      <p>Reward: {showScores && !isCalibrating ? (data?.reward ?? 0) : "N/A (before calibration)"}</p>
      <p>Trust score: {showScores && !isCalibrating ? `${data?.trust_score ?? 0}%` : "N/A (before calibration)"}</p>
      <p>Source: {data?.identified_by || "N/A"}</p>
    </div>
  );
}

export default App;