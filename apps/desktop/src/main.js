import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";
import { Command } from "@tauri-apps/plugin-shell";
import "./style.css";

const $ = (selector) => document.querySelector(selector);
const elements = {
  drop: $("#drop"),
  fileLabel: $("#file-label"),
  engine: $("#engine"),
  speakers: $("#speakers"),
  countRow: $("#count-row"),
  speakerCount: $("#speaker-count"),
  start: $("#start"),
  cancel: $("#cancel"),
  openResult: $("#open-result"),
  runtime: $("#runtime"),
  status: $("#status"),
  statusText: $("#status-text"),
  elapsed: $("#elapsed"),
  progressBar: $("#progress-bar"),
  log: $("#log"),
};

let mediaPath = null;
let child = null;
let resultPath = null;
let startedAt = null;
let timer = null;
let stderrLines = [];
let runtimeReady = false;
let speakerReady = false;

function refreshStart() {
  elements.start.disabled = !runtimeReady || !mediaPath || Boolean(child);
}

function fileName(path) {
  return path.split(/[\\/]/).pop();
}

function setMedia(path) {
  mediaPath = path;
  elements.fileLabel.textContent = fileName(path);
  elements.drop.classList.add("selected");
  elements.openResult.classList.add("hidden");
  resultPath = null;
  refreshStart();
}

function appendLog(line) {
  const clean = line.trim();
  if (!clean) return;
  elements.log.textContent = `${elements.log.textContent}${clean}\n`;
  elements.log.scrollTop = elements.log.scrollHeight;
}

function setRunning(running) {
  elements.start.classList.toggle("hidden", running);
  elements.cancel.classList.toggle("hidden", !running);
  elements.drop.disabled = running;
  elements.engine.disabled = running;
  elements.speakers.disabled = running || !speakerReady;
  elements.progressBar.classList.toggle("active", running);
  if (running) {
    startedAt = Date.now();
    timer = window.setInterval(() => {
      const seconds = Math.floor((Date.now() - startedAt) / 1000);
      const minutes = Math.floor(seconds / 60);
      elements.elapsed.textContent = `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
    }, 250);
  } else {
    window.clearInterval(timer);
    timer = null;
    refreshStart();
  }
}

async function chooseFile() {
  const selected = await open({
    multiple: false,
    filters: [{
      name: "Audio and video",
      extensions: ["mp3", "wav", "m4a", "flac", "ogg", "aac", "mp4", "webm", "mkv", "avi", "mov"],
    }],
  });
  if (selected) setMedia(selected);
}

async function checkRuntime() {
  try {
    const output = await Command.create("ft", ["doctor"]).execute();
    if (output.code !== 0) throw new Error(output.stderr);
    const doctor = JSON.parse(output.stdout);
    const engines = ["qwen", "parakeet"];
    const available = engines.filter((name) => doctor.ready[name]);
    for (const name of engines) {
      elements.engine.querySelector(`option[value="${name}"]`).disabled = !doctor.ready[name];
    }
    if (!doctor.ready[elements.engine.value] && available.length) {
      elements.engine.value = available[0];
    }
    runtimeReady = available.length > 0;
    speakerReady = Boolean(doctor.ready.speaker_transcription);
    elements.engine.disabled = !runtimeReady;
    elements.speakers.disabled = !speakerReady;
    if (!speakerReady) {
      elements.speakers.checked = false;
      elements.countRow.classList.add("hidden");
    }
    elements.runtime.textContent = available.length
      ? `Local · ${available.join(" · ")}${speakerReady ? " · speakers" : ""}`
      : "CLI installed · add an engine profile";
  } catch {
    runtimeReady = false;
    speakerReady = false;
    elements.runtime.textContent = "Install the free-transcribe backend first";
  }
  refreshStart();
}

async function transcribe() {
  if (!mediaPath || child) return;
  const args = [mediaPath, "--engine", elements.engine.value];
  if (elements.speakers.checked) {
    const count = elements.speakerCount.value.trim();
    args.push("--speakers");
    if (count) args.push(count);
  }

  stderrLines = [];
  resultPath = null;
  elements.log.textContent = "";
  elements.statusText.textContent = "Starting…";
  elements.elapsed.textContent = "0:00";
  elements.progressBar.style.width = "";
  elements.openResult.classList.add("hidden");
  setRunning(true);

  const command = Command.create("ft", args);
  command.stdout.on("data", appendLog);
  command.stderr.on("data", (line) => {
    stderrLines.push(line.trim());
    appendLog(line);
    const match = line.match(/^\[([^\]]+)\]\s+(.*)$/);
    if (match) elements.statusText.textContent = match[2];
  });
  command.on("error", (error) => {
    elements.statusText.textContent = String(error);
    setRunning(false);
    child = null;
  });
  command.on("close", ({ code }) => {
    child = null;
    setRunning(false);
    if (code === 0) {
      resultPath = [...stderrLines].reverse().find((line) => line.toLowerCase().endsWith(".md")) ?? null;
      elements.statusText.textContent = "Complete";
      elements.progressBar.style.width = "100%";
      elements.openResult.classList.toggle("hidden", !resultPath);
    } else {
      elements.statusText.textContent = `Stopped with code ${code}`;
    }
  });
  try {
    child = await command.spawn();
  } catch (error) {
    child = null;
    elements.statusText.textContent = String(error);
    setRunning(false);
  }
}

elements.drop.addEventListener("click", chooseFile);
elements.start.addEventListener("click", transcribe);
elements.cancel.addEventListener("click", async () => child?.kill());
elements.openResult.addEventListener("click", () => resultPath && openPath(resultPath));
elements.speakers.addEventListener("change", () => {
  elements.countRow.classList.toggle("hidden", !elements.speakers.checked);
});

await getCurrentWebview().onDragDropEvent((event) => {
  elements.drop.classList.toggle("dragging", event.payload.type === "over");
  if (event.payload.type === "drop" && event.payload.paths.length) {
    setMedia(event.payload.paths[0]);
  }
});

checkRuntime();
