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
        <div className="brand">
          <div className="brand-mark">PA</div>
          <div>
            <p className="eyebrow">Realtime Suite</p>
            <h1>PostureAI</h1>
          </div>
        </div>

        <div className="nav-group">
          <p className="nav-label">Workspace</p>
          <button className="nav-item" onClick={() => setScreen("home")}>
            <span className="nav-icon">H</span>
            Overview
          </button>
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
            </div>
          </section>

          <aside className="insight-column">
            <Scoreboard data={data} screen={screen} />
            <ConversationPanel data={data} screen={screen} />
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
  const badgeText = screen === "calibration" ? "Calibration Live" : "Interview Live";
  const subtext =
    screen === "calibration"
      ? "Align the face inside the frame and capture a clean reference."
      : data?.suggestion || "Maintain eye contact and continue your answer with steady posture.";

  return (
    <section className="hero-card">
      <div className="hero-card-header">
        <div>
          <p className="eyebrow">People attending the call</p>
          <h3>Candidate View</h3>
        </div>
      </div>

      <div className="video-stage">
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
        <img key={screen} src={streamSrc} alt="Live Feed" className="video-element" />

        <div className="video-caption">
          <div className="recording-pill">Session Active</div>
        </div>

      </div>

      <div className="live-brief">
        <div className="wave-mark" />
        <div>
          <p className="eyebrow">Conversation now</p>
          <strong>{subtext}</strong>
        </div>
      </div>
    </section>
  );
}

function CalibrationPanel({ data, showToast }) {
  const ratioPct = Math.round((data?.face_inside_ratio || 0) * 100);
  const ready = Boolean(data?.calibration_ready);
  const isCalibrating = data?.mode === "calibrating" || data?.mode === "calibration_freeze";
  const snapshot = data?.calibration_snapshot;
  const freezeActive = Boolean(data?.calibration_frozen);
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
    <>
      <section className="panel-grid two-up">
        <InfoCard
          title="Alignment score"
          value={`${alignmentHealth}%`}
          subtitle={ready ? "Ready to capture" : "Keep face inside the oval"}
        >
          <ProgressBar value={alignmentHealth} tone="violet" />
        </InfoCard>
        <InfoCard
          title="Freeze state"
          value={freezeActive ? "Locking" : "Idle"}
          subtitle={freezeActive ? `${data?.calibration_freeze_remaining?.toFixed(1)}s remaining` : "Awaiting capture"}
        >
          <ProgressBar value={freezeActive ? 100 : 24} tone="amber" />
        </InfoCard>
      </section>

      <section className="module-card">
        <div className="module-header">
          <div>
            <p className="eyebrow">Calibration controls</p>
            <h3>Reference capture workflow</h3>
          </div>
          <div className={`pill-status ${ready ? "success" : "warning"}`}>{ready ? "Reference Eligible" : "Needs Alignment"}</div>
        </div>

        <div className="detail-grid">
          <Metric label="Mode" value={data?.mode || "loading"} />
          <Metric label="Face in oval" value={`${ratioPct}%`} />
          <Metric label="Head angle" value={(data?.head_angle ?? 0).toFixed(2)} />
          <Metric label="Eye direction" value={(data?.eye_dir ?? 0).toFixed(2)} />
          <Metric label="Eye ratio" value={(data?.eye_ratio ?? 0).toFixed(2)} />
          <Metric label="Eye distance" value={(data?.eye_distance ?? 0).toFixed(2)} />
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

      <section className="panel-grid two-up">
        <InfoCard
          title="Live suggestion"
          value="Coach note"
          subtitle={data?.suggestion || "Waiting for camera data..."}
        />
        <InfoCard
          title="Captured snapshot"
          value={snapshot ? "Available" : "Pending"}
          subtitle={
            snapshot
              ? `Face ${Math.round(snapshot.face_w || 0)} x ${Math.round(snapshot.face_h || 0)}`
              : "No reference captured yet"
          }
        />
      </section>

      <TelemetryPanel data={data} showScores={!isCalibrating} />
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

        <div className="detail-grid compact">
          <Metric label="Timer" value={timer} highlight={timeLeft <= 20 && answering} />
          <Metric label="Backend mode" value={data?.mode || "loading"} />
          <Metric label="Eye contact" value={inActiveQuestion && data?.is_cheating ? "Lost" : "Good"} />
          <Metric label="Exam state" value={examReady ? "Started" : "Pending"} />
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

      <section className="panel-grid two-up">
        <InfoCard title="Live suggestion" value="Coach note" subtitle={data?.suggestion || "Waiting..."} />
        <InfoCard
          title="Risk watch"
          value={inActiveQuestion && data?.is_cheating ? "Attention drift" : "Stable"}
          subtitle={
            inActiveQuestion && data?.is_cheating
              ? "Eye contact dropped during the current answer."
              : "Candidate posture and gaze are within acceptable range."
          }
        />
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

      <TelemetryPanel data={data} showScores />
    </>
  );
}

function Scoreboard({ data, screen }) {
  const trustScore = Math.round(data?.trust_score ?? 0);
  const rewardValue = Number(data?.reward ?? 0);
  const rewardScore = Math.max(0, Math.min(100, Math.round((rewardValue + 1) * 50)));
  const modeLabel = screen === "calibration" ? "Alignment" : "Interview";

  return (
    <section className="scoreboard-card">
      <div className="score-header">
        <div>
          <p className="eyebrow">Live scores</p>
          <h3>Session quality</h3>
        </div>
      </div>

      <div className="ring-grid">
        <ProgressRing value={trustScore} label="Trust Score" tone="violet" />
        <ProgressRing value={rewardScore} label="Reward Index" tone="amber" />
      </div>

      <div className="score-tags">
        <div className="tag-row">
          <span>Tracking mode</span>
          <strong>{modeLabel}</strong>
        </div>
        <div className="tag-row">
          <span>Signal source</span>
          <strong>{data?.identified_by || "N/A"}</strong>
        </div>
      </div>
    </section>
  );
}

function ConversationPanel({ data, screen }) {
  const messages = useMemo(() => {
    if (screen === "calibration") {
      return [
        { author: "System", text: "Center your face inside the oval and hold still for reference capture.", accent: false },
        { author: "Coach", text: data?.suggestion || "Waiting for camera data...", accent: true },
        {
          author: "Status",
          text: data?.calibration_ready ? "Reference can be captured now." : "Keep adjusting until the alignment threshold is reached.",
          accent: false,
        },
      ];
    }

    return [
      { author: "Interviewer", text: "Continue answering with calm posture and consistent eye contact.", accent: false },
      { author: "Coach", text: data?.suggestion || "Waiting...", accent: true },
      {
        author: "Monitor",
        text: data?.is_cheating ? "Eye contact drift detected in the current answer." : "Posture signal looks stable right now.",
        accent: false,
      },
    ];
  }, [data?.calibration_ready, data?.is_cheating, data?.suggestion, screen]);

  return (
    <section className="chat-card">
      <div className="chat-tabs">Chat</div>

      <div className="chat-thread">
        {messages.map((message) => (
          <div key={`${message.author}-${message.text}`} className={`chat-bubble ${message.accent ? "accent" : ""}`}>
            <p className="chat-author">{message.author}</p>
            <p>{message.text}</p>
          </div>
        ))}
      </div>

      <div className="chat-input">Insights stream is read-only during the live session.</div>
    </section>
  );
}

function TelemetryPanel({ data, showScores }) {
  const isCalibrating = data?.mode === "calibrating";

  return (
    <section className="module-card">
      <div className="module-header">
        <div>
          <p className="eyebrow">RL telemetry</p>
          <h3>Realtime posture signals</h3>
        </div>
      </div>

      <div className="detail-grid">
        <Metric label="State" value={Array.isArray(data?.state) ? data.state.join(" | ") : "N/A"} />
        <Metric label="Action" value={data?.action || "N/A"} />
        <Metric label="Reward" value={showScores && !isCalibrating ? data?.reward ?? 0 : "N/A"} />
        <Metric
          label="Trust score"
          value={showScores && !isCalibrating ? `${data?.trust_score ?? 0}%` : "N/A"}
        />
      </div>
    </section>
  );
}

function ProgressBar({ value, tone = "violet" }) {
  return (
    <div className="progress-track">
      <div className={`progress-fill ${tone}`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}

function ProgressRing({ value, label, tone = "violet" }) {
  const safeValue = Math.max(0, Math.min(100, value));
  const degrees = safeValue * 3.6;

  return (
    <div className="ring-stat">
      <div
        className={`ring-visual ${tone}`}
        style={{ background: `conic-gradient(var(--ring-color) ${degrees}deg, #efe9ff ${degrees}deg)` }}
      >
        <div className="ring-inner">{safeValue}%</div>
      </div>
      <strong>{label}</strong>
    </div>
  );
}

function InfoCard({ title, value, subtitle, children }) {
  return (
    <section className="info-card">
      <p className="eyebrow">{title}</p>
      <h3>{value}</h3>
      <p className="info-subtitle">{subtitle}</p>
      {children}
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
