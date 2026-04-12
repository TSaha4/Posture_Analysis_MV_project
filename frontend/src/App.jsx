import React, { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import useRLData from "./hooks/useRLData";
import hrQuestions from "./data/hrQuestions.json";
import "./App.css";

const API = "http://127.0.0.1:8000";
const QUESTION_SECONDS = 180;

function App() {
  const [screen, setScreen] = useState("home");
  const [toast, setToast] = useState(null);
  const [examProgress, setExamProgress] = useState(100);
  const data = useRLData(screen !== "home");
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
    axios
      .get(`${API}/calibrate`)
      .catch((err) => console.debug("calibrate on enter failed:", err?.message));
  }, [screen]);

  if (screen === "home") {
    return <WelcomeScreen setScreen={setScreen} />;
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand brand-button" onClick={() => setScreen("home")}>
          <div className="brand-mark">PA</div>
          <div>
            <p className="eyebrow">Realtime Suite</p>
            <h1>PostureAI</h1>
          </div>
        </button>

        <div className="nav-group">
          <p className="nav-label">Workspace</p>
          <button
            className={`nav-item ${screen === "calibration" ? "active" : ""}`}
            onClick={() => setScreen("calibration")}
          >
            <span className="nav-icon">C</span>
            Calibration
          </button>
          <button
            className={`nav-item ${screen === "exam" ? "active" : ""}`}
            onClick={() => setScreen("exam")}
          >
            <span className="nav-icon">I</span>
            Interview
          </button>
        </div>

        <div className="nav-group">
          <p className="nav-label">Session</p>
          <div className="sidebar-summary card-soft">
            <span className={`status-dot ${backendOnline ? "online" : "offline"}`} />
            <div>
              <strong>{backendOnline ? "Backend connected" : "Backend offline"}</strong>
              <p>{data?.mode || "waiting"}</p>
            </div>
          </div>
        </div>

        <button className="logout-btn" onClick={leaveSession}>
          End Session
        </button>
      </aside>

      <main className="workspace">
        <header className="workspace-header">
          <div>
            <p className="eyebrow">{screen === "calibration" ? "Calibration studio" : "Interview control room"}</p>
            <h2>{screen === "calibration" ? "Reference Capture & Alignment" : "Mock Interview Performance"}</h2>
          </div>
          <div className="workspace-actions">
            <div className={`pill-status ${backendOnline ? "success" : "danger"}`}>
              {backendOnline ? "Live Feed Ready" : "Waiting for Backend"}
            </div>
            <button
              className="ghost-btn"
              onClick={() => setScreen(screen === "calibration" ? "exam" : "calibration")}
            >
              {screen === "calibration" ? "Go to Interview" : "Go to Calibration"}
            </button>
          </div>
        </header>

        {toast?.message ? <div className={`toast-banner ${toast.type || "info"}`}>{toast.message}</div> : null}

        <div className="dashboard-grid">
          <section className="content-column">
            <SessionHero
              screen={screen}
              data={data}
              streamSrc={streamSrc}
              examProgress={examProgress}
            />
            <SuggestionBox data={data} screen={screen} />
          </section>

          <aside className="insight-column">
            <Scoreboard data={data} screen={screen} />
            <div className="below-camera-scroll">
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
              {screen === "exam" ? <VisionInsightsCard data={data} /> : null}
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

function WelcomeScreen({ setScreen }) {
  return (
    <div className="welcome-screen">
      <div className="welcome-backdrop" />
      <div className="welcome-aura aura-left" />
      <div className="welcome-aura aura-right" />
      <div className="welcome-panel">
        <section className="welcome-card">
          <p className="welcome-badge">Posture Analysis Studio</p>
          <h2>Calibrate. Start. Shine.</h2>
          <p className="hero-subtitle">Professional mock interviews with clean live posture feedback.</p>
          <div className="hero-actions welcome-actions">
            <button className="primary-btn" onClick={() => setScreen("calibration")}>
              Open Calibration
            </button>
            <button className="ghost-btn" onClick={() => setScreen("exam")}>
              Open Interview
            </button>
          </div>
          <div className="welcome-meta">
            <div className="meta-pill">
              <span className="meta-dot success" />
              Live Guidance
            </div>
            <div className="meta-pill">
              <span className="meta-dot warm" />
              Real-time Score
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function SessionHero({ screen, data, streamSrc, examProgress }) {
  return (
    <section className="hero-card">
      <div className="video-stage">
        {screen === "exam" ? (
          <div className="camera-progress-shell">
            <div className="camera-progress-fill" style={{ width: `${examProgress}%` }} />
          </div>
        ) : null}
        {screen === "calibration" && data?.calibration_frozen ? (
          <div className="camera-freeze-overlay">
            <div className="freeze-spinner" />
          </div>
        ) : null}
        <img key={screen} src={streamSrc} alt="Live Feed" className="video-element" />
      </div>
    </section>
  );
}

function CalibrationPanel({ data, showToast }) {
  const ratioPct = Math.round((data?.face_inside_ratio || 0) * 100);
  const ready = Boolean(data?.calibration_ready);
  const isCalibrating = data?.mode === "calibrating" || data?.mode === "calibration_freeze";
  const alignmentHealth = Math.min(100, Math.max(0, ratioPct));

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
    <section className="module-card">
      <div className="module-header">
        <div>
          <p className="eyebrow">Calibration</p>
          <h3>Reference capture</h3>
        </div>
        <div className={`pill-status ${ready ? "success" : "warning"}`}>{ready ? "Ready" : "Align face"}</div>
      </div>

      <div className="detail-grid compact">
        <Metric label="Alignment" value={`${alignmentHealth}%`} />
        <Metric label="Face in oval" value={`${ratioPct}%`} />
        <Metric label="Head angle" value={(data?.head_angle ?? 0).toFixed(2)} />
        <Metric label="Eye direction" value={(data?.eye_dir ?? 0).toFixed(2)} />
      </div>

      <div className="module-actions">
        <button className="ghost-btn" onClick={resetCalibration}>
          Reset
        </button>
        <button className="primary-btn" onClick={captureReference} disabled={!ready || !isCalibrating}>
          Capture Reference
        </button>
      </div>
    </section>
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
      const scoreText =
        res.data.time_penalty > 0
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
  const examStateLabel = answering ? "Answering" : examReady ? "Started" : "Pending";

  return (
    <>
      <section className="module-card interview-focus">
        <div className="module-header">
          <div>
            <p className="eyebrow">Current prompt</p>
            <h3>Question {questionIdx + 1} of 3</h3>
          </div>
          <div className={`pill-status ${answering ? "danger" : "success"}`}>{answering ? "Answer in Progress" : "Ready"}</div>
        </div>

        <p className="question-copy">{q?.question || "Preparing questions..."}</p>

        <div className="detail-grid exam-status-grid">
          <Metric label="Timer" value={timer} highlight={timeLeft <= 20 && answering} />
          <Metric label="Exam state" value={examStateLabel} />
        </div>

        <div className="module-actions">
          <button className="primary-btn" onClick={startAnswer} disabled={!examReady || answering || Boolean(result)}>
            Start Answer
          </button>
          <button className="ghost-btn" onClick={finalizeQuestion} disabled={!answering}>
            End Now
          </button>
        </div>
      </section>

      {result ? (
        <section className="module-card">
          <div className="module-header">
            <div>
              <p className="eyebrow">Question result</p>
              <h3>Performance snapshot</h3>
            </div>
            <div className={`pill-status ${result.score < 70 ? "danger" : "success"}`}>{result.label}</div>
          </div>

          <div className="result-summary">
            <div className={`score-pill ${result.score < 70 ? "danger" : "success"}`}>{result.score}/100</div>
            <div className="result-copy">
              <p>
                {result.base_score !== undefined && result.time_penalty > 0
                  ? `Base score ${result.base_score}% with a ${result.time_penalty}% time penalty after ${result.elapsed_time}s.`
                  : "No time penalty applied for this answer."}
              </p>
            </div>
          </div>

          <div className="error-list">
            {(result.errors || []).map((e) => (
              <div key={e.key} className="error-item">
                <span>{e.description}</span>
                <strong>{e.key === "time_penalty" ? `${e.percent_frames}% penalty` : `${e.percent_frames}%`}</strong>
              </div>
            ))}
          </div>

          <div className="module-actions">
            <button className="ghost-btn" onClick={redoQuestion}>
              Redo Question
            </button>
            {questionIdx < 2 ? (
              <button className="primary-btn" onClick={nextQuestion}>
                Next Question
              </button>
            ) : (
              <>
                <button className="ghost-btn" onClick={restartSession}>
                  Restart Session
                </button>
                <button className="primary-btn" onClick={finishExam}>
                  Finish Exam
                </button>
              </>
            )}
          </div>
        </section>
      ) : null}
    </>
  );
}

function Scoreboard({ data, screen }) {
  const trustScore = Math.round(data?.trust_score ?? 0);
  const rewardValue = Number(data?.reward ?? 0);
  const rewardScore = Math.max(0, Math.min(100, Math.round(((rewardValue + 1) / 2) * 100)));
  const modeLabel = screen === "calibration" ? "Alignment" : "Interview";

  return (
    <section className="scoreboard-card">
      <div className="score-header">
        <div>
          <p className="eyebrow">Live scores</p>
          <h3>Session quality</h3>
        </div>
      </div>

      <div className="score-summary-grid">
        <ScoreMetric label="Trust" value={`${trustScore}%`} percent={trustScore} />
        <ScoreMetric label="Reward" value={`${rewardScore}%`} percent={rewardScore} />
        <ScoreMetric label="Mode" value={modeLabel} />
      </div>
    </section>
  );
}

function SuggestionBox({ data, screen }) {
  const fallback =
    screen === "calibration"
      ? "Align your face inside the frame and keep steady."
      : "Maintain eye contact and continue with steady posture.";
  const message = data?.suggestion || fallback;
  const normalized = message.toLowerCase();
  const isPositive =
    normalized.includes("good posture") ||
    normalized.includes("ready") ||
    normalized.includes("steady");
  const toneClass = isPositive ? "positive" : "alert";

  return (
    <section className={`suggestion-card ${toneClass}`}>
      <p className="eyebrow">Suggestion</p>
      <strong>{message}</strong>
    </section>
  );
}

function ScoreMetric({ label, value, percent = null }) {
  const safePercent = percent == null ? null : Math.max(0, Math.min(100, percent));
  return (
    <div className="score-metric">
      {safePercent == null ? null : (
        <div
          className="score-circle"
          style={{ background: `conic-gradient(var(--violet) ${safePercent * 3.6}deg, #efe9ff 0deg)` }}
        >
          <div>{value}</div>
        </div>
      )}
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function VisionInsightsCard({ data }) {
  const action = data?.action || "no_action";
  const actionTone =
    action === "no_action"
      ? "good"
      : action === "adjust_posture"
        ? "warn"
        : "danger";

  const stateRows = [
    ["Position", data?.position || "N/A"],
    ["Head", data?.head || "N/A"],
    ["Gaze", data?.gaze || "N/A"],
  ];

  const processingRows = [
    ["Edge Detection", data?.vision_processing?.edge_detection],
    ["Thresholding", data?.vision_processing?.thresholding],
    ["Corner Detection", data?.vision_processing?.corner_detection],
  ];

  return (
    <section className="module-card">
      <div className="module-header">
        <div>
          <p className="eyebrow">Machine Vision Insights</p>
          <h3>Essential signals</h3>
        </div>
      </div>

      <div className="insight-section">
        <p className="eyebrow">Motion</p>
        <div className="insight-grid">
          <div className="insight-row">
            <span>Movement Level</span>
            <strong>{data?.movement_level || "low"}</strong>
          </div>
        </div>
      </div>

      <div className="insight-section">
        <p className="eyebrow">Attention</p>
        <div className="insight-grid">
          <div className="insight-row full">
            <span>Looking away</span>
            <strong>{Number(data?.attention_duration ?? 0).toFixed(1)}s</strong>
          </div>
        </div>
      </div>

      <div className="insight-section">
        <p className="eyebrow">System Suggestion</p>
        <div className={`suggestion-pill ${actionTone}`}>
          <strong>{action}</strong>
          <span>{data?.suggestion || "Waiting..."}</span>
        </div>
      </div>

      <details className="insight-details">
        <summary>Details</summary>
        <div className="detail-list">
          {stateRows.map(([label, value]) => (
            <div key={label} className="insight-row">
              <span>{label}</span>
              <strong>{value}</strong>
            </div>
          ))}
          <div className="insight-row">
            <span>Movement Value</span>
            <strong>{Number(data?.movement_value ?? 0).toFixed(2)}</strong>
          </div>
          <div className="insight-row">
            <span>Normalized Face X</span>
            <strong>{Number(data?.normalized_face_x ?? 0).toFixed(3)}</strong>
          </div>
          <div className="insight-row">
            <span>Normalized Face Y</span>
            <strong>{Number(data?.normalized_face_y ?? 0).toFixed(3)}</strong>
          </div>
          <div className="insight-row">
            <span>Normalized Head Angle</span>
            <strong>{Number(data?.normalized_head_angle ?? 0).toFixed(3)}</strong>
          </div>
          <div className="insight-row">
            <span>Normalized Eye Dir</span>
            <strong>{Number(data?.normalized_eye_dir ?? 0).toFixed(3)}</strong>
          </div>
          {processingRows.map(([label, active]) => (
            <div key={label} className="insight-row">
              <span>{label}</span>
              <strong>{active ? "Active" : "Idle"}</strong>
            </div>
          ))}
        </div>
      </details>
    </section>
  );
}

function Metric({ label, value, highlight = false }) {
  return (
    <div className={`metric-card ${highlight ? "highlight" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
