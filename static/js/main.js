// Clockwise dartboard order starting at 20 (12 o'clock)
const SECTORS = [
  20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5,
];

// ---- Boot parameters from server (no behaviour change) ----
const __BOOT__ =
  typeof window !== "undefined" && window.GAME_BOOT ? window.GAME_BOOT : {};
const BOOT_MODE = typeof __BOOT__.game === "string" ? __BOOT__.game : "around";
const BOOT_X01_START = Number(__BOOT__.x01Start || 501);
const BOOT_DOUBLE_OUT =
  __BOOT__.doubleOut === true || __BOOT__.doubleOut === "true";
try {
  console.info("[BOOT]", { BOOT_MODE, BOOT_X01_START, BOOT_DOUBLE_OUT });
} catch (_) {}

// Persist whether the calibration panel is shown
const CAL_UI_KEY = "ardarts.calui";

// ---- Persist calibration between sessions ----
const CAL_KEY = "ardarts.cal";
function getSavedCal() {
  try {
    return JSON.parse(localStorage.getItem(CAL_KEY) || "{}");
  } catch (_) {
    return {};
  }
}
function saveCal() {
  try {
    localStorage.setItem(
      CAL_KEY,
      JSON.stringify({
        R: BOARD_RADIUS_FUDGE,
        CX: CENTER_X_FUDGE,
        CY: CENTER_Y_FUDGE,
        ROT: ROT_OFFSET_DEG,
        IMG_ROT: BOARD_IMG_ROT_DEG,
        IMG_SCALE: BOARD_IMG_SCALE,
        IMG_X: BOARD_IMG_X,
        IMG_Y: BOARD_IMG_Y,
      }),
    );
  } catch (_) {}
}

// --- Calibration save confirmation and button wiring ---
function stampSaved(msg = "Saved") {
  const el = document.getElementById("cal-saved");
  if (!el) return;
  el.textContent = msg + " • " + new Date().toLocaleTimeString();
  setTimeout(() => {
    el.textContent = "";
  }, 2500);
}

(function wireCalSave() {
  const btn = document.getElementById("cal-save");
  if (!btn) return;
  btn.addEventListener("click", () => {
    saveCal();
    stampSaved();
  });
})();
function loadCalUi() {
  try {
    return JSON.parse(localStorage.getItem(CAL_UI_KEY) || "false");
  } catch (_) {
    return false;
  }
}
function saveCalUi(isShown) {
  try {
    localStorage.setItem(CAL_UI_KEY, JSON.stringify(!!isShown));
  } catch (_) {}
}

// ---- Around-the-World targets ----
const TARGETS = [
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  16,
  17,
  18,
  19,
  20,
  "bull",
];

// ---------------------------------------------------------------------------------
// GAME STATE (plain object so HUD code can keep working unchanged)
// ---------------------------------------------------------------------------------
let game = {
  mode: BOOT_MODE,
  x01Start: BOOT_X01_START,
  doubleOut: BOOT_DOUBLE_OUT,
  players: [
    {
      name: "Player 1",
      idx: 0,
      score: BOOT_X01_START,
      done: false,
      lastHit: null,
      turnThrows: [],
    },
    {
      name: "Player 2",
      idx: 0,
      score: BOOT_X01_START,
      done: false,
      lastHit: null,
      turnThrows: [],
    },
  ],
  turn: 0,
  throwsLeft: 3,
  turnStartScore: BOOT_X01_START,
  history: [],
  active: false,
  winner: null,
};

// --- Calibration knobs (tweak live with arrow keys) ---
let BOARD_RADIUS_FUDGE = 0.92;
let CENTER_X_FUDGE = 0;
let CENTER_Y_FUDGE = 0;

// Sector math / hit detection rotation
let ROT_OFFSET_DEG = 2;

// Visual-only board image transform
let BOARD_IMG_ROT_DEG = 0;
let BOARD_IMG_SCALE = 1.0;
let BOARD_IMG_X = 0;
let BOARD_IMG_Y = 0;

// Initial visibility of guides / panel from storage
let SHOW_CAL_PANEL = loadCalUi();

// Apply any saved calibration
(function applySavedCal() {
  const s = getSavedCal();
  if (typeof s.R === "number") BOARD_RADIUS_FUDGE = s.R;
  if (typeof s.CX === "number") CENTER_X_FUDGE = s.CX;
  if (typeof s.CY === "number") CENTER_Y_FUDGE = s.CY;
  if (typeof s.ROT === "number") ROT_OFFSET_DEG = s.ROT;
  if (typeof s.IMG_ROT === "number") BOARD_IMG_ROT_DEG = s.IMG_ROT;
  if (typeof s.IMG_SCALE === "number") BOARD_IMG_SCALE = s.IMG_SCALE;
  if (typeof s.IMG_X === "number") BOARD_IMG_X = s.IMG_X;
  if (typeof s.IMG_Y === "number") BOARD_IMG_Y = s.IMG_Y;
})();

// --- Visual guides toggle & ring ratios ---
let SHOW_GUIDES = true;
const RATIOS = {
  outer: 1.0,
  doubleInner: 0.95,
  tripleOuter: 0.63,
  tripleInner: 0.57,
  bullOuter: 0.09,
  bullInner: 0.035,
};

const container = document.getElementById("board-container");
const overlay = document.getElementById("overlay");
const board = document.getElementById("dartboard");
const statusEl = document.getElementById("status");

// ---------------------------------------------------------
// Board image rotation (decoupled from ROT_OFFSET_DEG)
// ---------------------------------------------------------
function applyBoardImageTransform() {
  if (!board) return;
  board.style.transformOrigin = "50% 50%";
  board.style.transform =
    `translate(${BOARD_IMG_X}px, ${BOARD_IMG_Y}px) ` +
    `scale(${BOARD_IMG_SCALE}) ` +
    `rotate(${BOARD_IMG_ROT_DEG}deg)`;
}

const ctx = overlay.getContext("2d");

// --- Simple SFX wiring (hit / bust / win) with WebAudio fallback ---
const SFX = { hit: null, bust: null, win: null };
const AUDIO = { ctx: null };
function initSfx() {
  SFX.hit = document.getElementById("sfx-hit") || null;
  SFX.bust = document.getElementById("sfx-bust") || null;
  SFX.win = document.getElementById("sfx-win") || null;
}
function ensureAudio() {
  if (!AUDIO.ctx) {
    try {
      AUDIO.ctx = new (window.AudioContext || window.webkitAudioContext)();
    } catch (_) {}
  }
  if (AUDIO.ctx && AUDIO.ctx.state === "suspended") {
    AUDIO.ctx.resume();
  }
}
function unmuteSfx() {
  [SFX.hit, SFX.bust, SFX.win].forEach((el) => {
    if (el) el.muted = false;
  });
  ensureAudio();
}
function beep(freq = 880, dur = 0.08, type = "sine", gain = 0.08) {
  const ctx = AUDIO.ctx;
  if (!ctx) return;
  const osc = ctx.createOscillator();
  const g = ctx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, ctx.currentTime);
  g.gain.setValueAtTime(gain, ctx.currentTime);
  g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + dur);
  osc.connect(g).connect(ctx.destination);
  osc.start();
  osc.stop(ctx.currentTime + dur);
}
function playSfx(kind) {
  const el = SFX[kind];
  if (el) {
    try {
      el.currentTime = 0;
      void el.play();
      return;
    } catch (_) {}
  }
  ensureAudio();
  if (!AUDIO.ctx) return;
  if (kind === "hit") {
    beep(880, 0.06, "square", 0.06);
  } else if (kind === "bust") {
    beep(180, 0.2, "sawtooth", 0.08);
  } else if (kind === "win") {
    beep(660, 0.09, "sine", 0.07);
    setTimeout(() => beep(880, 0.12, "sine", 0.07), 100);
    setTimeout(() => beep(1320, 0.16, "sine", 0.07), 230);
  }
}

// ---------------------------------------------
// PLAYER CARD STATE HELPERS
// ---------------------------------------------
function recordLastHit(label) {
  const p = game.players[game.turn];
  p.lastHit = label;
}

function updateLastHitUI(playerIndex, label) {
  const el = document.getElementById(playerIndex === 0 ? "p1-last" : "p2-last");
  if (el) {
    el.textContent = `Last hit: ${label}`;
  }
}

function updateTargetUI(playerIndex) {
  const el = document.getElementById(
    playerIndex === 0 ? "p1-target" : "p2-target",
  );
  if (!el) return;
  const p = game.players[playerIndex];
  if (game.mode === "x01") {
    el.textContent = `Need: ${p.score}`;
  } else {
    const t = TARGETS[p.idx];
    el.textContent = `To hit: ${t === "bull" ? "Bull" : t}`;
  }
}

function updateThrowsUI(playerIndex) {
  const el = document.getElementById(
    playerIndex === 0 ? "p1-throws" : "p2-throws",
  );
  if (!el) return;
  const p = game.players[playerIndex];
  const icons = [];
  for (let i = 0; i < 3; i++) {
    if (p.turnThrows[i] === "hit") icons.push("✅");
    else if (p.turnThrows[i] === "miss") icons.push("❌");
    else icons.push("🎯");
  }
  el.textContent = `Throws: ${icons.join(" ")}`;
}

function resizeOverlay() {
  const rect = container.getBoundingClientRect();
  overlay.style.width = rect.width + "px";
  overlay.style.height = rect.height + "px";
  overlay.width = Math.max(1, Math.round(rect.width));
  overlay.height = Math.max(1, Math.round(rect.height));
  drawFade();
}

function drawFade() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const { cx, cy, radius } = boardCenterAndRadius();
  const rect = overlay.getBoundingClientRect();
  const g = ctx.createRadialGradient(
    cx - rect.left,
    cy - rect.top,
    Math.max(1, radius * 0.2),
    cx - rect.left,
    cy - rect.top,
    Math.max(overlay.width, overlay.height) * 0.6,
  );
  g.addColorStop(0, "rgba(0,0,0,0.0)");
  g.addColorStop(1, "rgba(0,0,0,0.10)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, overlay.width, overlay.height);
  drawGuides(cx - rect.left, cy - rect.top, radius);
  drawATWTargetHighlight(cx - rect.left, cy - rect.top, radius);
}

function ringFromRadiusFrac(r) {
  if (r <= 0.05) return "inner_bull";
  if (r <= 0.12) return "outer_bull";
  if (r >= 0.94 && r <= 1.0) return "double";
  if (r >= 0.57 && r <= 0.65) return "treble";
  if (r > 1.0) return "miss";
  return "single";
}

function sectorIndexFromAngle(a) {
  a = a + Math.PI / 2 - (ROT_OFFSET_DEG * Math.PI) / 180;
  a = ((a % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
  return Math.floor((a / (Math.PI * 2)) * 20) % 20;
}

function boardCenterAndRadius() {
  const r = board.getBoundingClientRect();
  let cx = (r.left + r.right) / 2 + CENTER_X_FUDGE;
  let cy = (r.top + r.bottom) / 2 + CENTER_Y_FUDGE;
  let radius = (Math.min(r.width, r.height) / 2) * BOARD_RADIUS_FUDGE;
  return { cx, cy, radius };
}

function drawMarker(x, y, label) {
  ctx.save();
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#0ff";
  ctx.font = "14px system-ui";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(label, x + 10, y + 10);
  ctx.restore();
}

function drawGuides(cxLocal, cyLocal, radius) {
  if (!SHOW_GUIDES) return;
  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(0, 255, 255, 1)";
  const rings = [
    radius * RATIOS.outer,
    radius * RATIOS.doubleInner,
    radius * RATIOS.tripleOuter,
    radius * RATIOS.tripleInner,
    radius * RATIOS.bullOuter,
    radius * RATIOS.bullInner,
  ];
  rings.forEach((r) => {
    ctx.beginPath();
    ctx.arc(cxLocal, cyLocal, r, 0, Math.PI * 2);
    ctx.stroke();
  });
  ctx.strokeStyle = "rgba(255,255,0,0.7)";
  ctx.beginPath();
  ctx.moveTo(cxLocal - 8, cyLocal);
  ctx.lineTo(cxLocal + 8, cyLocal);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cxLocal, cyLocal - 8);
  ctx.lineTo(cxLocal, cyLocal + 8);
  ctx.stroke();

  const zero = -Math.PI / 2 + (ROT_OFFSET_DEG * Math.PI) / 180;
  const zx = cxLocal + radius * Math.cos(zero);
  const zy = cyLocal + radius * Math.sin(zero);
  ctx.strokeStyle = "rgba(0,255,255,0.85)";
  ctx.beginPath();
  ctx.moveTo(cxLocal, cyLocal);
  ctx.lineTo(zx, zy);
  ctx.stroke();
  ctx.restore();
}

function drawATWTargetHighlight(cxLocal, cyLocal, radius) {
  if (game.mode !== "around") return;
  if (!game.active) return;
  const p = game.players[game.turn];
  const target = TARGETS[p.idx];
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  if (target === "bull") {
    const rOuter = radius * RATIOS.bullOuter * 1.35;
    const rInner = 0;
    const g = ctx.createRadialGradient(
      cxLocal,
      cyLocal,
      rInner,
      cxLocal,
      cyLocal,
      rOuter,
    );
    g.addColorStop(0, "rgba(0,255,200,0.25)");
    g.addColorStop(1, "rgba(0, 255, 200, 1)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cxLocal, cyLocal, rOuter, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(0,255,200,0.65)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cxLocal, cyLocal, radius * RATIOS.bullOuter, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    const i = SECTORS.indexOf(target);
    if (i >= 0) {
      const step = (Math.PI * 2) / 20;
      const aStart = i * step;
      const aEnd = (i + 1) * step;
      const thetaStart =
        aStart - Math.PI / 2 + (ROT_OFFSET_DEG * Math.PI) / 180;
      const thetaEnd = aEnd - Math.PI / 2 + (ROT_OFFSET_DEG * Math.PI) / 180;
      const rOut = radius * RATIOS.outer * 1.01;
      ctx.fillStyle = "rgba(180, 255, 0, 1)";
      ctx.strokeStyle = "rgba(255, 47, 47, 1)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(cxLocal, cyLocal);
      ctx.arc(cxLocal, cyLocal, rOut, thetaStart, thetaEnd, false);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
  }
  ctx.restore();
}

function hookCalibrationPanel() {
  const rx = document.getElementById("cal-radius");
  const dx = document.getElementById("cal-dx");
  const dy = document.getElementById("cal-dy");
  const rot = document.getElementById("cal-rot");
  const tog = document.getElementById("cal-toggle");
  const imgRot = document.getElementById("cal-img-rot");
  const imgRotVal = document.getElementById("cal-img-rot-val");
  const panel = document.getElementById("cal-panel");
  const btnCal = document.getElementById("btn-cal");
  const imgScale = document.getElementById("cal-img-scale");
  const imgX = document.getElementById("cal-img-x");
  const imgY = document.getElementById("cal-img-y");
  if (!panel || !btnCal) return;
  if (panel) {
    panel.classList.toggle("hidden", !SHOW_CAL_PANEL);
  }
  if (btnCal) {
    btnCal.addEventListener("click", () => {
      SHOW_CAL_PANEL = !SHOW_CAL_PANEL;
      if (panel) panel.classList.toggle("hidden", !SHOW_CAL_PANEL);
      saveCalUi(SHOW_CAL_PANEL);
    });
  }
  const rVal = document.getElementById("cal-r-val");
  const dxVal = document.getElementById("cal-dx-val");
  const dyVal = document.getElementById("cal-dy-val");
  const rotVal = document.getElementById("cal-rot-val");
  const sync = () => {
    if (rVal) rVal.textContent = BOARD_RADIUS_FUDGE.toFixed(3);
    if (dxVal) dxVal.textContent = CENTER_X_FUDGE;
    if (dyVal) dyVal.textContent = CENTER_Y_FUDGE;
    if (rotVal) rotVal.textContent = ROT_OFFSET_DEG.toFixed(1);
    if (imgRotVal) imgRotVal.textContent = BOARD_IMG_ROT_DEG.toFixed(1);
  };
  if (rx) rx.value = BOARD_RADIUS_FUDGE.toFixed(3);
  if (dx) dx.value = CENTER_X_FUDGE;
  if (dy) dy.value = CENTER_Y_FUDGE;
  if (rot) rot.value = ROT_OFFSET_DEG;
  if (imgRot) imgRot.value = BOARD_IMG_ROT_DEG;
  if (tog) tog.checked = SHOW_GUIDES;
  if (imgScale) imgScale.value = BOARD_IMG_SCALE.toFixed(3);
  if (imgX) imgX.value = BOARD_IMG_X;
  if (imgY) imgY.value = BOARD_IMG_Y;
  sync();
  if (rx) {
    rx.addEventListener("input", () => {
      BOARD_RADIUS_FUDGE = parseFloat(rx.value) || BOARD_RADIUS_FUDGE;
      drawFade();
      saveCal();
    });
  }
  if (dx) {
    dx.addEventListener("input", () => {
      CENTER_X_FUDGE = parseInt(dx.value || "0", 10);
      drawFade();
      saveCal();
    });
  }
  if (dy) {
    dy.addEventListener("input", () => {
      CENTER_Y_FUDGE = parseInt(dy.value || "0", 10);
      drawFade();
      saveCal();
    });
  }
  if (rot) {
    rot.addEventListener("input", () => {
      ROT_OFFSET_DEG = parseFloat(rot.value) || 0;
      drawFade();
      sync();
      saveCal();
    });
  }
  if (imgRot) {
    imgRot.addEventListener("input", () => {
      BOARD_IMG_ROT_DEG = parseFloat(imgRot.value) || 0;
      applyBoardImageTransform();
      sync();
      saveCal();
    });
  }
  if (imgScale) {
    imgScale.addEventListener("input", () => {
      BOARD_IMG_SCALE = parseFloat(imgScale.value) || BOARD_IMG_SCALE;
      applyBoardImageTransform();
      saveCal();
    });
  }
  if (imgX) {
    imgX.addEventListener("input", () => {
      BOARD_IMG_X = parseInt(imgX.value || "0");
      applyBoardImageTransform();
      saveCal();
    });
  }
  if (imgY) {
    imgY.addEventListener("input", () => {
      BOARD_IMG_Y = parseInt(imgY.value || "0");
      applyBoardImageTransform();
      saveCal();
    });
  }
  if (tog) {
    tog.addEventListener("change", () => {
      SHOW_GUIDES = !!tog.checked;
      drawFade();
      saveCal();
    });
  }
}

async function logToServer(payload) {
  try {
    const res = await fetch("/hit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await res.json();
  } catch (e) {
    console.warn("Log failed (OK if backend not running):", e);
  }
}

// -------------------------------------------------------------------------------------------------
// Camera endpoints and detection helpers
// -------------------------------------------------------------------------------------------------

// Flag: true while detectDartFromCamera() is in flight.
// Poller skips when this is set so the same hit is never scored twice.
let _manualDetecting = false;

async function captureBoardBefore() {
  try {
    const res = await fetch("/capture-before", { method: "POST" });
    const data = await res.json();
    if (!data.ok) throw new Error("capture failed");
    if (statusEl) {
      statusEl.textContent =
        "Board captured. Throw a dart, then press Detect Dart.";
    }
  } catch (e) {
    console.warn("capture-before failed:", e);
    if (statusEl) {
      statusEl.textContent = "Capture failed – check Pi logs.";
    }
  }
}

function normaliseDetectionHit(hit) {
  if (!hit) return { ring: "miss", sector: null };
  const t = hit.type;
  if (t === "inner_bull" || t === "outer_bull") {
    return { ring: t, sector: 25 };
  }
  if (t === "miss" || hit.sector == null) {
    return { ring: "miss", sector: null };
  }
  return { ring: t, sector: hit.sector };
}

async function detectDartFromCamera() {
  _manualDetecting = true;
  try {
    const res = await fetch("/detect", { method: "POST" });
    const data = await res.json();
    if (!data.ok) {
      throw new Error("detect endpoint returned !ok");
    }
    // Drain _latest_hit so the poller doesn't re-process this same detection
    fetch("/latest-hit").catch(() => {});
    if (!data.hit) {
      if (statusEl) statusEl.textContent = "No impact detected.";
      const throwingPlayer = game.turn;
      recordLastHit("Miss");
      updateLastHitUI(throwingPlayer, "Miss");
      game.players[throwingPlayer].turnThrows.push("miss");
      updateThrowsUI(throwingPlayer);
      return;
    }

    const { ring, sector } = normaliseDetectionHit(data.hit);

    if (ring === "miss") {
      if (statusEl) statusEl.textContent = "Detected: Miss";
      const throwingPlayer = game.turn;
      recordLastHit("Miss");
      updateLastHitUI(throwingPlayer, "Miss");
      game.players[throwingPlayer].turnThrows.push("miss");
      updateThrowsUI(throwingPlayer);
      return;
    }

    if (!game.active) {
      startGame();
    }

    unmuteSfx();
    ensureAudio();

    const label = ring.includes("bull")
      ? ring.replace("_", " ")
      : `${ring} ${sector}`;

    if (statusEl) {
      statusEl.textContent = `Detected: ${label}`;
    }

    const throwingPlayer = game.turn;
    recordLastHit(label);
    updateLastHitUI(throwingPlayer, label);
    game.players[throwingPlayer].turnThrows.push("hit");
    updateThrowsUI(throwingPlayer);
    applyHit(ring, sector);
    updateTargetUI(throwingPlayer);

    logToServer({
      source: "camera",
      ring,
      sector,
      cx: data.hit.x,
      cy: data.hit.y,
    });
  } catch (e) {
    console.warn("detect failed:", e);
    if (statusEl) {
      statusEl.textContent = "Detect failed – check Pi logs.";
    }
  } finally {
    _manualDetecting = false;
  }
}

// -------------------------------------------------------------------------------------------------
// MODEL LAYER: GameEngine + Modes
// -------------------------------------------------------------------------------------------------
class GameEngine {
  constructor(gameState) {
    this.g = gameState;
    this.mode = null;
  }
  setMode(name) {
    if (name === "around") this.mode = AroundMode;
    else if (name === "x01") this.mode = X01Mode;
    else this.mode = AroundMode;
    this.g.mode = name;
    console.info("[Engine] mode set:", this.mode.name);
  }
  start() {
    if (!this.mode) this.setMode(this.g.mode || "around");
    this.g.active = true;
    this.g.turn = 0;
    this.g.throwsLeft = 3;
    this.g.history = [];
    this.g.winner = null;
    this.g.players.forEach((p) => {
      p.done = false;
      p.idx = 0;
      p.score = this.g.x01Start;
      p.turnThrows = [];
    });
    this.g.turnStartScore = this.g.players[0].score;
    this.mode.onStart(this);
    drawFade();
    unmuteSfx();
    console.info("[Engine] start", {
      mode: this.mode.name,
      state: this.snapshot(),
    });
  }
  reset() {
    this.g.active = false;
    this.g.turn = 0;
    this.g.throwsLeft = 3;
    this.g.history = [];
    this.g.winner = null;
    this.g.players.forEach((p) => {
      p.done = false;
      p.idx = 0;
      p.score = this.g.x01Start;
      p.turnThrows = [];
    });
    this.g.turnStartScore = this.g.x01Start;
    console.info("[Engine] reset");
  }
  nextPlayer() {
    this.g.turn = (this.g.turn + 1) % this.g.players.length;
    this.g.throwsLeft = 3;
    this.g.players[this.g.turn].turnThrows = [];
    updateThrowsUI(this.g.turn);
    this.g.turnStartScore = this.g.players[this.g.turn].score;
    drawFade();
    try {
      const badge = document.getElementById("turn-badge");
      if (badge) badge.textContent = `Player ${this.g.turn + 1} to throw`;
    } catch (_) {}
    try {
      const cur = this.g.players[this.g.turn];
      if (typeof statusEl !== "undefined" && statusEl) {
        if (this.g.mode === "around") {
          statusEl.textContent = `Player ${this.g.turn + 1} — to hit: ${atwTargetText(cur)}`;
        } else {
          statusEl.textContent = `Player ${this.g.turn + 1} — need ${cur.score}`;
        }
      }
    } catch (_) {}
    fetch("/reset-baseline", { method: "POST" }).catch(() => {});
    // Disarm mic — waits for Ready button before re-arming
    fetch("/disarm-audio", { method: "POST" }).catch(() => {});
    showReadyButton();
    console.info("[Engine] nextPlayer", { turn: this.g.turn });
  }
  pushSnapshot() {
    const cur = this.g.players[this.g.turn];
    this.g.history.push({
      turn: this.g.turn,
      throwsLeft: this.g.throwsLeft,
      idx: cur.idx,
      score: cur.score,
      done: cur.done,
      turnStartScore: this.g.turnStartScore,
    });
  }
  undo() {
    const last = this.g.history.pop();
    if (!last) return;
    this.g.turn = last.turn;
    this.g.throwsLeft = last.throwsLeft;
    const p = this.g.players[this.g.turn];
    p.idx = last.idx;
    p.score = last.score;
    p.done = last.done;
    this.g.turnStartScore = last.turnStartScore;
    console.info("[Engine] undo", { restored: last, state: this.snapshot() });
  }
  end(winnerIdx = null, note = "") {
    this.g.active = false;
    this.g.winner = typeof winnerIdx === "number" ? winnerIdx : null;
    console.info("[Engine] end", { winnerIdx, note, state: this.snapshot() });
  }
  applyHit(ring, sector) {
    if (!this.g.active || !this.mode) return;
    this.pushSnapshot();
    const result = this.mode.applyHit(this, { ring, sector });

    playSfx("hit");

    if (!result || result.consumeThrow !== false) {
      this.g.throwsLeft -= 1;
    }

    let ended = false;
    if (result && result.win) {
      this.end(this.g.turn, result.note || "win");
      ended = true;
    }

    if (result && result.win) {
      playSfx("win");
    } else if (result && result.bust) {
      playSfx("bust");
    }

    drawFade();

    if (!ended && this.g.throwsLeft <= 0) {
      this.nextPlayer();
    }

    console.info("[Engine] hit", {
      ring,
      sector,
      result,
      state: this.snapshot(),
    });
  }
  snapshot() {
    const p = this.g.players.map((x) => ({
      name: x.name,
      idx: x.idx,
      score: x.score,
      done: x.done,
    }));
    return {
      mode: this.g.mode,
      turn: this.g.turn,
      throwsLeft: this.g.throwsLeft,
      players: p,
      turnStartScore: this.g.turnStartScore,
      winner: this.g.winner,
      active: this.g.active,
    };
  }
}

// --- Around-the-World mode ---
const AroundMode = {
  name: "around",
  onStart(engine) {
    const g = engine.g;
    if (typeof statusEl !== "undefined" && statusEl) {
      statusEl.textContent = `Player ${g.turn + 1} — to hit: 1`;
    }
  },
  applyHit(engine, hit) {
    const { ring, sector } = hit;
    const g = engine.g;
    const p = g.players[g.turn];
    if (p.done) return { note: "already finished" };

    const target = TARGETS[p.idx];
    let advance = 0;

    if (target === "bull") {
      if (ring === "inner_bull" || ring === "outer_bull") advance = 1;
    } else if (sector === target) {
      if (ring === "treble") advance = 3;
      else if (ring === "double") advance = 2;
      else if (ring === "single") advance = 1;
    }

    if (advance > 0) {
      p.idx = Math.min(p.idx + advance, TARGETS.length - 1);
    }

    const dartsLeftAfter = g.throwsLeft - 1;
    const nextTgt = TARGETS[p.idx] === "bull" ? "Bull" : TARGETS[p.idx];

    let win = false;
    if (
      p.idx === TARGETS.length - 1 &&
      (ring === "inner_bull" || ring === "outer_bull")
    ) {
      p.done = true;
      win = true;
    }

    if (typeof statusEl !== "undefined" && statusEl) {
      const hitLabel = ring.includes("bull")
        ? ring.replace("_", " ")
        : `${ring} ${sector}`;
      const afterZero = dartsLeftAfter <= 0;
      const nextPlayerNum = ((g.turn + 1) % g.players.length) + 1;
      if (win) {
        statusEl.textContent = `Hit: ${hitLabel} • advanced +${advance} • Winner: ${p.name} 🎯`;
      } else if (advance === 0) {
        statusEl.textContent = afterZero
          ? `Hit: ${hitLabel} • no advance • Turn complete. Player ${nextPlayerNum} is up!`
          : `Hit: ${hitLabel} • no advance • still to hit: ${nextTgt}`;
      } else {
        statusEl.textContent = afterZero
          ? `Hit: ${hitLabel} • advanced +${advance} • Turn complete. Player ${nextPlayerNum} is up!`
          : `Hit: ${hitLabel} • advanced +${advance} • now to hit: ${nextTgt}`;
      }
    }

    return { advanced: advance, win };
  },
};

// --- X01 mode ---
const X01Mode = {
  name: "x01",
  onStart(engine) {
    const g = engine.g;
    g.players.forEach((p) => {
      p.score = g.x01Start;
      p.done = false;
      p.idx = 0;
    });
    g.turnStartScore = g.players[g.turn].score;
    if (typeof statusEl !== "undefined" && statusEl) {
      statusEl.textContent = `Player ${g.turn + 1} — need ${g.players[g.turn].score}`;
    }
  },
  applyHit(engine, hit) {
    const { ring, sector } = hit;
    const g = engine.g;
    const p = g.players[g.turn];

    const delta = ringScore(ring, sector);
    const next = p.score - delta;
    const hitLabel = ring.includes("bull")
      ? ring.replace("_", " ")
      : `${ring} ${sector}`;
    const dartsLeftAfter = g.throwsLeft - 1;

    if (next < 0 || (next === 0 && g.doubleOut && !isDouble(ring))) {
      p.score = g.turnStartScore;
      g.throwsLeft = 1;
      if (typeof statusEl !== "undefined" && statusEl) {
        const nextPlayerNum = ((g.turn + 1) % g.players.length) + 1;
        const msg =
          next === 0 && g.doubleOut && !isDouble(ring)
            ? `Hit: ${hitLabel} • not a double → Bust! Turn complete. Player ${nextPlayerNum} is up!`
            : `Hit: ${hitLabel} • Bust! Turn complete. Player ${nextPlayerNum} is up!`;
        statusEl.textContent = msg;
      }
      return { bust: true };
    }

    if (next === 0) {
      if (!g.doubleOut || isDouble(ring)) {
        p.score = 0;
        p.done = true;
        if (typeof statusEl !== "undefined" && statusEl) {
          statusEl.textContent = `Hit: ${hitLabel} • Checkout! Winner: ${p.name} 🎯`;
        }
        return { win: true, note: "checkout" };
      }
    }

    p.score = next;
    if (typeof statusEl !== "undefined" && statusEl) {
      const nextPlayerNum = ((g.turn + 1) % g.players.length) + 1;
      statusEl.textContent =
        dartsLeftAfter <= 0
          ? `Hit: ${hitLabel} • need ${next} • Turn complete. Player ${nextPlayerNum} is up!`
          : `Hit: ${hitLabel} • need ${next}`;
    }

    return {};
  },
};

// ---- Helpers ----
function ringScore(ring, sector) {
  if (ring === "inner_bull") return 50;
  if (ring === "outer_bull") return 25;
  if (ring === "double") return sector * 2;
  if (ring === "treble") return sector * 3;
  if (ring === "single") return sector;
  return 0;
}
function isDouble(ring) {
  return ring === "double" || ring === "inner_bull";
}
function atwTargetText(p) {
  const t = TARGETS[p.idx];
  return t === "bull" ? "Bull" : String(t);
}
function showReadyButton() {
  const badge = document.getElementById("turn-badge");
  if (!badge) return;
  badge.innerHTML = "";
  const btn = document.createElement("button");
  btn.textContent = `Player ${game.turn + 1} — Ready to throw?`;
  btn.className = "btn-ready";
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Setting up...";
    await fetch("/arm-audio", { method: "POST" }).catch(() => {});
    badge.textContent = `Player ${game.turn + 1} to throw`;
  });
  badge.appendChild(btn);
}
// -------------------------------------------------------------------------------------------------
// UI glue
// -------------------------------------------------------------------------------------------------
function renderPlayers() {
  const host = document.getElementById("players");
  if (!host) return;
  host.innerHTML = "";
  game.players.forEach((p, i) => {
    const d = document.createElement("span");
    d.className =
      "player-pill" +
      (p.done ? " done" : "") +
      (i === game.turn ? " active" : "");
    if (game.mode === "x01") {
      d.textContent = `${p.name} • ${p.score}`;
    } else {
      const t = TARGETS[p.idx];
      d.textContent = `${p.name} • ${p.done ? "✓ finished" : "→ " + (t === "bull" ? "Bull" : t)}`;
    }
    host.appendChild(d);
  });
  const cur = game.players[game.turn];
  document.getElementById("hud-player").textContent = cur.name;
  document.getElementById("hud-throws").textContent = String(game.throwsLeft);
  if (game.mode === "x01") {
    document.getElementById("hud-target").textContent = cur.score;
  } else {
    const tgt = TARGETS[cur.idx] === "bull" ? "Bull" : TARGETS[cur.idx];
    document.getElementById("hud-target").textContent = `To hit: ${tgt}`;
  }

  if (!game.active && game.winner !== null) {
    const w = game.players[game.winner];
    if (statusEl) statusEl.textContent = `Winner: ${w.name} 🎯`;
  }

  const undoBtn = document.getElementById("btn-undo");
  if (undoBtn) undoBtn.disabled = game.history.length === 0;

  const p1 = game.players[0];
  const p2 = game.players[1];

  const p1NameEl = document.getElementById("p1-name");
  const p2NameEl = document.getElementById("p2-name");
  const p1TargetEl = document.getElementById("p1-target");
  const p2TargetEl = document.getElementById("p2-target");
  const p1Card = document.getElementById("p1-card");
  const p2Card = document.getElementById("p2-card");
  const p1LastEl = document.getElementById("p1-last");
  const p2LastEl = document.getElementById("p2-last");

  const formatTarget = (player) => {
    if (game.mode === "x01") return `Need: ${player.score}`;
    const t = TARGETS[player.idx];
    return `To hit: ${t === "bull" ? "Bull" : t}`;
  };

  if (p1NameEl) p1NameEl.textContent = p1.name;
  if (p2NameEl) p2NameEl.textContent = p2.name;
  if (p1TargetEl) p1TargetEl.textContent = formatTarget(p1);
  if (p2TargetEl) p2TargetEl.textContent = formatTarget(p2);

  if (p1LastEl) {
    p1LastEl.textContent = p1.lastHit
      ? `Last hit: ${p1.lastHit}`
      : "Last hit: —";
  }
  if (p2LastEl) {
    p2LastEl.textContent = p2.lastHit
      ? `Last hit: ${p2.lastHit}`
      : "Last hit: —";
  }

  if (p1Card) {
    p1Card.classList.remove("active", "inactive");
    p1Card.classList.add(game.turn === 0 ? "active" : "inactive");
  }
  if (p2Card) {
    p2Card.classList.remove("active", "inactive");
    p2Card.classList.add(game.turn === 1 ? "active" : "inactive");
  }

  const badge = document.getElementById("turn-badge");
  if (badge) {
    if (game.winner !== null) {
      badge.textContent = `Winner: ${game.players[game.winner].name} 🎯`;
    } else if (game.active) {
      if (!badge.querySelector(".btn-ready")) {
        badge.textContent = `Player ${game.turn + 1} to throw`;
      }
    }
  }
}

// Engine instance and UI delegates
let ENGINE = new GameEngine(game);

function resetGame() {
  ENGINE.reset();
  renderPlayers();
  updateThrowsUI(0);
  updateThrowsUI(1);
  applyBoardImageTransform();
}
function startGame() {
  game.mode = BOOT_MODE;
  game.x01Start = BOOT_X01_START;
  game.doubleOut = BOOT_DOUBLE_OUT;
  ENGINE.setMode(game.mode);
  ENGINE.start();
  fetch("/arm-audio", { method: "POST" }).catch(() => {});
  drawFade();
  if (statusEl) {
    if (game.mode === "around") {
      statusEl.textContent = `Player 1 — to hit: 1`;
    } else {
      statusEl.textContent = `Player 1 — starting at ${game.x01Start}`;
    }
  }
  renderPlayers();
  updateThrowsUI(0);
  updateThrowsUI(1);
  applyBoardImageTransform();
}
function nextPlayer() {
  ENGINE.nextPlayer();
  renderPlayers();
}
function applyHit(ring, sector) {
  ENGINE.applyHit(ring, sector);
  renderPlayers();
}
function undo() {
  ENGINE.undo();
  renderPlayers();
}

function handleClick(e) {
  const rect = overlay.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  const { cx, cy, radius } = boardCenterAndRadius();
  const cxLocal = cx - rect.left;
  const cyLocal = cy - rect.top;

  const dx = x - cxLocal,
    dy = y - cyLocal;
  const rFrac = Math.hypot(dx, dy) / Math.max(1, radius);
  const ring = ringFromRadiusFrac(rFrac);
  const angle = Math.atan2(dy, dx);
  const secIdx = sectorIndexFromAngle(angle);
  const sectorNum = SECTORS[secIdx];

  if (!game.active) {
    startGame();
  }
  unmuteSfx();
  ensureAudio();

  drawFade();
  drawMarker(
    x,
    y,
    ring.includes("bull") ? ring.replace("_", " ") : `${ring} ${sectorNum}`,
  );

  const hitLabel = ring.includes("bull")
    ? ring.replace("_", " ")
    : `${ring} ${sectorNum}`;

  if (statusEl) {
    statusEl.textContent = `Hit: ${hitLabel}`;
  }

  const throwingPlayer = game.turn;
  recordLastHit(hitLabel);
  updateLastHitUI(throwingPlayer, hitLabel);
  game.players[throwingPlayer].turnThrows.push("hit");
  updateThrowsUI(throwingPlayer);
  applyHit(ring, sectorNum);
  updateTargetUI(throwingPlayer);

  console.log("CLICK", { x, y, ring, secIdx, sectorNum });
  logToServer({ ring, sectorIndex: secIdx, px: x, py: y });
}

// Simple live calibration with arrow keys and +/-
window.addEventListener("keydown", (e) => {
  let changed = false;
  const stepPx = 1;
  switch (e.key) {
    case "ArrowLeft":
      CENTER_X_FUDGE -= stepPx;
      changed = true;
      break;
    case "ArrowRight":
      CENTER_X_FUDGE += stepPx;
      changed = true;
      break;
    case "ArrowUp":
      CENTER_Y_FUDGE -= stepPx;
      changed = true;
      break;
    case "ArrowDown":
      CENTER_Y_FUDGE += stepPx;
      changed = true;
      break;
    case "+":
    case "=":
      BOARD_RADIUS_FUDGE += 0.005;
      changed = true;
      break;
    case "-":
      BOARD_RADIUS_FUDGE -= 0.005;
      changed = true;
      break;
    case "[":
      BOARD_IMG_ROT_DEG -= 0.5;
      changed = true;
      break;
    case "]":
      BOARD_IMG_ROT_DEG += 0.5;
      changed = true;
      break;
  }
  if (changed) {
    applyBoardImageTransform();
    drawFade();
    statusEl.textContent = `Cal: R=${BOARD_RADIUS_FUDGE.toFixed(3)}  dx=${CENTER_X_FUDGE}  dy=${CENTER_Y_FUDGE}  rot=${ROT_OFFSET_DEG.toFixed(1)}°`;
    saveCal();
  }
});

window.addEventListener("keydown", (e) => {
  if ((e.key === "c" || e.key === "C") && (e.ctrlKey || e.metaKey)) {
    const panel = document.getElementById("cal-panel");
    if (panel) {
      SHOW_CAL_PANEL = !SHOW_CAL_PANEL;
      panel.classList.toggle("hidden", !SHOW_CAL_PANEL);
      saveCalUi(SHOW_CAL_PANEL);
      e.preventDefault();
    }
  }
});

window.addEventListener("resize", resizeOverlay);
board.addEventListener("load", resizeOverlay);
overlay.addEventListener("click", handleClick);

// -------------------------------------------------------------------------------------------------
// AUDIO TRIGGER POLLING
// -------------------------------------------------------------------------------------------------
let _pollInterval = null;

function startPolling() {
  // Always clear any existing interval first — prevents duplicate pollers on page refresh
  if (_pollInterval) {
    clearInterval(_pollInterval);
    _pollInterval = null;
  }
  _pollInterval = setInterval(async () => {
    if (!game.active || _manualDetecting) return;
    try {
      const res = await fetch("/latest-hit");
      const data = await res.json();
      if (!data.ok || !data.hit) return;

      const { ring, sector } = normaliseDetectionHit(data.hit);

      unmuteSfx();
      ensureAudio();

      const label = ring.includes("bull")
        ? ring.replace("_", " ")
        : `${ring} ${sector}`;

      if (statusEl) statusEl.textContent = `Detected: ${label}`;

      const throwingPlayer = game.turn;
      recordLastHit(label);
      updateLastHitUI(throwingPlayer, label);

      if (ring === "miss") {
        game.players[throwingPlayer].turnThrows.push("miss");
      } else {
        game.players[throwingPlayer].turnThrows.push("hit");
        applyHit(ring, sector);
        updateTargetUI(throwingPlayer);
      }
      updateThrowsUI(throwingPlayer);
      renderPlayers();
    } catch (_) {
      // silently ignore poll errors
    }
  }, 1000);
}

// -------------------------------------------------------------------------------------------------
// SINGLE DOMContentLoaded — all init here
// -------------------------------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  resizeOverlay();
  hookCalibrationPanel();
  drawFade();
  initSfx();
  applyBoardImageTransform();

  const s = document.getElementById("btn-start");
  const r = document.getElementById("btn-reset");
  const u = document.getElementById("btn-undo");
  const c = document.getElementById("btn-capture");
  const d = document.getElementById("btn-detect");
  if (s) s.addEventListener("click", startGame);
  if (r) r.addEventListener("click", resetGame);
  if (u) u.addEventListener("click", undo);
  if (c)
    c.addEventListener("click", () => {
      captureBoardBefore();
    });
  if (d)
    d.addEventListener("click", () => {
      detectDartFromCamera();
    });

  ENGINE.setMode(game.mode);
  renderPlayers();
  updateThrowsUI(0);
  updateThrowsUI(1);

  // Start polling — exactly once
  startPolling();
});
