import React, { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import useVisionData from "./hooks/useVisionData";
import hrQuestions from "./data/hrQuestions.json";
import "./App.css";

const API = "http://127.0.0.1:8000";
const QUESTION_SECONDS = 180;
const STATE_POLL_SECONDS = 0.5;

function App() {
  const [screen, setScreen] = useState("home");
  const [toast, setToast] = useState(null);
  const [examProgress, setExamProgress] = useState(100);
  const data = useVisionData(screen !== "home");
  const backendOnline = Boolean(data?.connected);
  const streamSrc = useMemo(() => `${API}/video_feed?screen=${screen}`, [screen]);
  const showToast = useCallback((message, type = "info") => {
    setToast({ message, type });
  }, []);

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

  const finishAndLeaveSession = async () => {
    await stopAll();
    setScreen("home");
  };

  useEffect(() => {
    if (!toast?.message) return;
    const t = setTimeout(() => setToast(null), 3000);
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
          <div className="badge-chip">Machine Vision Interview Pipeline</div>
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

      {toast?.message ? <div className={`toast-banner ${toast.type || "info"}`}>{toast.message}</div> : null}

      <div className="main-content">
        <div className="left-panel">
          <div className="video-frame-container full-height">
            <div className="video-title-strip">
              <span>Live Machine Vision Feed</span>
              <strong>{backendOnline ? "Streaming" : "Waiting"}</strong>
            </div>
            {screen === "exam" ? (
              <div className="camera-progress-shell">
                <div className="camera-progress-fill" style={{ width: `${examProgress}%` }} />
              </div>
            ) : null}
            {screen === "calibration" && data?.calibration_frozen ? (
              <div className="camera-freeze-overlay">
                <div className="freeze-spinner" />
                <p>Locking reference... {data.calibration_freeze_remaining?.toFixed(1)}s</p>
              </div>
            ) : null}
            <img
              key={screen}
              src={streamSrc}
              alt="Live Feed"
              className="video-element"
            />
            <VisionVideoOverlay data={data} />
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
            <CalibrationPanel data={data} showToast={showToast} />
          ) : (
            <ExamPanel
              data={data}
              showToast={showToast}
              setExamProgress={setExamProgress}
              onFinishSession={finishAndLeaveSession}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function VisionVideoOverlay({ data }) {
  const state = Array.isArray(data?.state) ? data.state : [];
  const position = data?.position || state[0] || "N/A";
  const head = data?.head || state[1] || "N/A";
  const gaze = data?.gaze || state[2] || "N/A";
  const movement = data?.movement_level || "low";
  const movementValue = data?.movement_value ?? data?.movement ?? 0;
  const attentionSeconds = ((data?.gaze_away_counter || 0) * STATE_POLL_SECONDS).toFixed(1);
  const suggestionAction = data?.action || "N/A";
  const suggestionClass = {
    no_action: "good",
    adjust_posture: "warning",
    look_screen: "danger",
    reduce_movement: "danger",
  }[suggestionAction] || "neutral";

  return (
    <div className="mv-video-overlay">
      <div>
        <span>Position</span>
        <strong>{position === "centered" ? "center" : position}</strong>
      </div>
      <div>
        <span>Head</span>
        <strong>{head}</strong>
      </div>
      <div>
        <span>Gaze</span>
        <strong>{gaze}</strong>
      </div>
      <div>
        <span>Movement</span>
        <strong>{movement} ({movementValue.toFixed(1)})</strong>
      </div>
      <div>
        <span>Away</span>
        <strong>{attentionSeconds}s</strong>
      </div>
      <div className={`overlay-suggestion ${suggestionClass}`}>
        <span>Suggestion</span>
        <strong>{suggestionAction}</strong>
      </div>
    </div>
  );
}

function CalibrationPanel({ data, showToast }) {
  const ratioPct = Math.round((data?.face_inside_ratio || 0) * 100);
  const ready = Boolean(data?.calibration_ready);
  const isCalibrating = data?.mode === "calibrating" || data?.mode === "calibration_freeze";
  const snapshot = data?.calibration_snapshot;
  const freezeActive = Boolean(data?.calibration_frozen);

  const resetCalibration = async () => {
    try {
      await axios.get(`${API}/calibrate`);
      showToast("Calibration reset. Align face and capture reference.", "info");
    } catch (err) {
      showToast(err?.response?.data?.message || "Could not reset calibration.", "error");
    }
  };

  const captureReference = async () => {
    try {
      await axios.get(`${API}/capture_reference`);
      showToast("Reference captured successfully.", "success");
    } catch (err) {
      showToast(err?.response?.data?.message || "Could not capture reference.", "error");
    }
  };

  return (
    <>
      <div className="card">
        <h3>Calibration Status</h3>
        {isCalibrating ? (
          <>
            <p>Face inside oval: <strong>{ratioPct}%</strong></p>
            <p>Capture rule: minimum 50%</p>
            <p>Freeze status: <strong>{freezeActive ? `Locking (${data?.calibration_freeze_remaining?.toFixed(1)}s)` : "Idle"}</strong></p>
            <p>Ready: <strong>{ready ? "Yes" : "No"}</strong></p>
            <p>Face Width: {Math.round(data?.face_width || 0)}</p>
            <p>Face Height: {Math.round(data?.face_height || 0)}</p>
            <p>Head Angle: {(data?.head_angle ?? 0).toFixed(2)}</p>
            <p>Eye Direction: {(data?.eye_dir ?? 0).toFixed(2)}</p>
            <p>Eye Ratio: {(data?.eye_ratio ?? 0).toFixed(2)}</p>
            <p>Eye Distance: {(data?.eye_distance ?? 0).toFixed(2)}</p>
          </>
        ) : (
          <p style={{ color: "#f2c94c" }}>
            Click <strong>Reset</strong> to return to calibration mode.
          </p>
        )}
        <p>Mode: <strong>{data?.mode || "loading"}</strong></p>
      </div>

      {snapshot ? (
        <div className="card">
          <h3>Captured Reference Ratios</h3>
          <p>Face Width: {Math.round(snapshot.face_w || 0)}</p>
          <p>Face Height: {Math.round(snapshot.face_h || 0)}</p>
          <p>Head Angle: {(snapshot.head_angle ?? 0).toFixed(2)}</p>
          <p>Eye Direction: {(snapshot.eye_dir ?? 0).toFixed(2)}</p>
          <p>Eye Ratio: {(snapshot.eye_ratio ?? 0).toFixed(2)}</p>
          <p>Eye Distance: {(snapshot.eye_dist ?? 0).toFixed(2)}</p>
        </div>
      ) : null}

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

      <MachineVisionInsights data={data} showInsights={!isCalibrating} />
    </>
  );
}

function ExamPanel({ data, showToast, setExamProgress, onFinishSession }) {
  const [questions, setQuestions] = useState([]);
  const [questionIdx, setQuestionIdx] = useState(0);
  const [answering, setAnswering] = useState(false);
  const [timeLeft, setTimeLeft] = useState(QUESTION_SECONDS);
  const [result, setResult] = useState(null);
  const [examReady, setExamReady] = useState(false);
  const baselineReady = ["posture", "exam", "exam_question"].includes(data?.mode);
  const inActiveQuestion = data?.mode === "exam_question";

  useEffect(() => {
    if (answering) {
      setExamProgress((timeLeft / QUESTION_SECONDS) * 100);
      return;
    }
    setExamProgress(result ? 0 : 100);
  }, [answering, timeLeft, result, setExamProgress]);

  useEffect(() => () => setExamProgress(100), [setExamProgress]);

  const ensureExamStarted = useCallback(async () => {
    try {
      await axios.get(`${API}/start_exam`);
      setExamReady(true);
      return true;
    } catch (err) {
      setExamReady(false);
      showToast(err?.response?.data?.message || "Exam requires completed calibration.", "error");
      return false;
    }
  }, [showToast]);

  useEffect(() => {
    const shuffled = [...hrQuestions].sort(() => Math.random() - 0.5).slice(0, 3);
    setQuestions(shuffled);
    setQuestionIdx(0);
    setAnswering(false);
    setTimeLeft(QUESTION_SECONDS);
    setResult(null);

    ensureExamStarted();
  }, [ensureExamStarted]);

  const restartSession = async () => {
    try {
      await axios.get(`${API}/end_exam`);
    } catch (err) {
      console.debug("end_exam before restart skipped:", err?.message);
    }

    const shuffled = [...hrQuestions].sort(() => Math.random() - 0.5).slice(0, 3);
    setQuestions(shuffled);
    setQuestionIdx(0);
    setAnswering(false);
    setTimeLeft(QUESTION_SECONDS);
    setResult(null);
    setExamReady(false);
    showToast("Session restarted.", "info");
  };

  useEffect(() => {
    if (!baselineReady || examReady) return;
    ensureExamStarted();
  }, [baselineReady, examReady, ensureExamStarted]);

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
      showToast(err?.response?.data?.message || "Could not compute question result.", "error");
    } finally {
      setAnswering(false);
    }
  }, [showToast]);

  useEffect(() => {
    if (!answering || timeLeft !== 0) return;
    finalizeQuestion();
  }, [answering, timeLeft, finalizeQuestion]);

  const startAnswer = async () => {
    setResult(null);
    setTimeLeft(QUESTION_SECONDS);
    try {
      const started = await ensureExamStarted();
      if (!started) return;
      await axios.get(`${API}/begin_answer`);
      setAnswering(true);
    } catch (err) {
      setExamReady(false);
      showToast(err?.response?.data?.message || "Could not start answer timer.", "error");
    }
  };

  const nextQuestion = () => {
    setQuestionIdx((q) => Math.min(q + 1, 2));
    setResult(null);
    setTimeLeft(QUESTION_SECONDS);
    setAnswering(false);
  };

  const redoQuestion = () => {
    setResult(null);
    setTimeLeft(QUESTION_SECONDS);
    setAnswering(false);
  };

  const finishExam = async () => {
    try {
      const res = await axios.get(`${API}/end_exam`);
      const scoreText = res.data.time_penalty > 0 
        ? `Final score ${res.data.score}/100 (${res.data.label}) - Base: ${res.data.base_score}%, Time penalty: ${res.data.time_penalty}%`
        : `Final score ${res.data.score}/100 (${res.data.label})`;
      showToast(scoreText, "success");
      await onFinishSession();
    } catch (err) {
      showToast(err?.response?.data?.message || "Could not end exam.", "error");
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
        {inActiveQuestion && data?.is_cheating ? <p style={{ color: "#f85149" }}>Eye contact lost.</p> : null}
      </div>

      {result ? (
        <div className="card">
          <h3>Question Score</h3>
          <h1 className={`score-number ${result.score < 70 ? "critical" : "stable"}`}>{result.score}/100</h1>
          <p>{result.label}</p>
          {result.base_score !== undefined && result.time_penalty > 0 && (
            <p style={{ fontSize: "0.9em", color: "#f85149" }}>
              Base score: {result.base_score}% - Time penalty: {result.time_penalty}% 
              (Answered in {result.elapsed_time}s)
            </p>
          )}
          {result.raw_error_percent !== undefined ? (
            <p style={{ fontSize: "0.9em", color: "#8b949e" }}>
              Timer-only scoring: {result.total_frames || 0} frames, {result.raw_error_percent}% drift,
              {` ${result.penalty_error_percent}% counted after ${result.error_tolerance_percent}% tolerance.`}
            </p>
          ) : null}
          <div style={{ display: "grid", gap: 6 }}>
            {(result.errors || []).map((e) => (
              <div key={e.key} className="state-box">
                {e.description} ({e.key === "time_penalty" ? `${e.percent_frames}% penalty` : `${e.percent_frames}%`})
              </div>
            ))}
          </div>
          <div className="exam-controls result-actions" style={{ marginTop: 10 }}>
            <button className="exam-btn secondary" onClick={redoQuestion} style={{ marginRight: 10 }}>
              Redo Question
            </button>
            {questionIdx < 2 ? (
              <button className="exam-btn primary" onClick={nextQuestion}>Next Question</button>
            ) : (
              <>
                <button className="exam-btn secondary" onClick={restartSession} style={{ marginRight: 10 }}>
                  Restart Session
                </button>
                <button className="exam-btn primary" onClick={finishExam}>Finish Exam</button>
              </>
            )}
          </div>
        </div>
      ) : null}

      <MachineVisionInsights data={data} showInsights />
    </>
  );
}

function MachineVisionInsights({ data, showInsights }) {
  const isCalibrating = data?.mode === "calibrating";
  const isReady = showInsights && !isCalibrating;
  const state = Array.isArray(data?.state) ? data.state : [];
  const position = data?.position || state[0] || "N/A";
  const displayPosition = position === "centered" ? "center" : position;
  const head = data?.head || state[1] || "N/A";
  const gaze = data?.gaze || state[2] || "N/A";
  const movementLevel = data?.movement_level || "low";
  const movementValue = data?.movement_value ?? data?.movement ?? 0;
  const processingFlags = data?.processing_flags || {};
  const movementLabel = {
    low: "Low (small movement)",
    medium: "Medium",
    high: "High (large movement)",
  }[movementLevel] || "N/A";
  const attentionSeconds = ((data?.gaze_away_counter || 0) * STATE_POLL_SECONDS).toFixed(1);
  const suggestionAction = data?.action || "N/A";
  const suggestionClass = {
    no_action: "good",
    adjust_posture: "warning",
    look_screen: "danger",
    reduce_movement: "danger",
  }[suggestionAction] || "neutral";

  return (
    <div className="card vision-insights-card">
      <h3>Machine Vision Insights</h3>
      <p className="mv-status-line">System running on Machine Vision (Rule-based Analysis)</p>

      <div className="insight-section">
        <h4>Current State Breakdown</h4>
        <div className="insight-grid">
          <div className="insight-row"><span>👤 Position</span><strong>{isReady ? displayPosition : "N/A"}</strong></div>
          <div className="insight-row"><span>🧠 Head</span><strong>{isReady ? head : "N/A"}</strong></div>
          <div className="insight-row"><span>👀 Gaze</span><strong>{isReady ? gaze : "N/A"}</strong></div>
        </div>
      </div>

      <div className="insight-row full">
        <span>📉 Movement</span>
        <strong>{isReady ? movementLabel : "N/A"}</strong>
      </div>

      <div className="insight-row full">
        <span>⏱ Attention</span>
        <strong>{isReady ? `Looking away for ${attentionSeconds} seconds` : "N/A"}</strong>
      </div>

      <div className="insight-section">
        <h4>Live MV Measurements</h4>
        <div className="measurement-grid">
          <span>Face</span><strong>{Math.round(data?.face_width || 0)} x {Math.round(data?.face_height || 0)}</strong>
          <span>Head angle</span><strong>{(data?.head_angle ?? 0).toFixed(2)}</strong>
          <span>Eye direction</span><strong>{(data?.eye_dir ?? 0).toFixed(2)}</strong>
          <span>Eye ratio</span><strong>{(data?.eye_ratio ?? 0).toFixed(2)}</strong>
          <span>Movement px</span><strong>{movementValue.toFixed(1)}</strong>
        </div>
      </div>

      <div className="insight-section">
        <h4>Vision Processing</h4>
        <div className="processing-grid">
          <div className={processingFlags.edge_detected ? "stage-chip active" : "stage-chip"}>
            <span>Edge Detection</span>
            <strong>{processingFlags.edge_detected ? "Active" : "Waiting"}</strong>
          </div>
          <div className={processingFlags.threshold_applied ? "stage-chip active" : "stage-chip"}>
            <span>Thresholding</span>
            <strong>{processingFlags.threshold_applied ? "Active" : "Waiting"}</strong>
          </div>
          <div className={processingFlags.corners_detected ? "stage-chip active" : "stage-chip"}>
            <span>Corner Detection</span>
            <strong>{processingFlags.corners_detected ? "Active" : "Waiting"}</strong>
          </div>
        </div>
      </div>

      <div className={`suggestion-pill ${suggestionClass}`}>
        <span>Suggestion</span>
        <strong>{isReady ? suggestionAction : "N/A"}</strong>
      </div>
    </div>
  );
}

export default App;
