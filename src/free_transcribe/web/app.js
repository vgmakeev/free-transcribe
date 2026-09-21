const $ = (selector) => document.querySelector(selector);

const elements = {
  runtime: $("#runtime"), authRow: $("#auth-row"), token: $("#token"),
  drop: $("#drop"), file: $("#file"), fileLabel: $("#file-label"),
  speakers: $("#speakers"), countRow: $("#count-row"), speakerCount: $("#speaker-count"),
  namesRow: $("#names-row"), speakerNames: $("#speaker-names"),
  gemini: $("#gemini"), start: $("#start"), progress: $("#progress"),
  stages: $("#stages"), status: $("#status"), percent: $("#percent"), bar: $("#bar"),
  result: $("#result"), transcript: $("#transcript"), copy: $("#copy"), download: $("#download"),
};

let media = null;
let running = false;
let parakeetReady = false;
let speakersReady = false;
let geminiReady = false;
elements.token.value = sessionStorage.getItem("free-transcribe-token") ?? "";

function headers() {
  const token = elements.token.value.trim();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function refreshStart() {
  elements.start.disabled = running || !media || !parakeetReady;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function setMedia(file) {
  media = file;
  elements.fileLabel.textContent = `${file.name} · ${formatBytes(file.size)}`;
  elements.drop.classList.add("selected");
  refreshStart();
}

function refreshSpeakerOptions() {
  const enabled = elements.speakers.checked;
  elements.countRow.classList.toggle("hidden", !enabled);
  elements.namesRow.classList.toggle("hidden", !enabled);
}

function isVideo(file) {
  return file.type.startsWith("video/") || /\.(mp4|webm|mkv|avi|mov)$/i.test(file.name);
}

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function encodeMonoWav(audio, sampleRate = 16000) {
  const outputLength = Math.ceil(audio.duration * sampleRate);
  const buffer = new ArrayBuffer(44 + outputLength * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + outputLength * 2, true);
  writeAscii(view, 8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, outputLength * 2, true);

  const channels = Array.from(
    { length: audio.numberOfChannels },
    (_, index) => audio.getChannelData(index),
  );
  const ratio = audio.sampleRate / sampleRate;
  for (let outputIndex = 0; outputIndex < outputLength; outputIndex += 1) {
    const position = outputIndex * ratio;
    const left = Math.min(Math.floor(position), audio.length - 1);
    const right = Math.min(left + 1, audio.length - 1);
    const fraction = position - left;
    let sample = 0;
    for (const channel of channels) {
      sample += channel[left] + (channel[right] - channel[left]) * fraction;
    }
    sample = Math.max(-1, Math.min(1, sample / channels.length));
    view.setInt16(44 + outputIndex * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}

async function extractAudio(file) {
  if (!isVideo(file)) return file;
  setProgress("Extracting audio locally in your browser…", { stage: "extract" });
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) throw new Error("This browser cannot extract audio from video");
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    const wav = encodeMonoWav(decoded);
    const stem = file.name.replace(/\.[^.]+$/, "");
    return new File([wav], `${stem}.wav`, { type: "audio/wav" });
  } catch (error) {
    throw new Error(
      `The browser could not decode this video's audio. Convert it to WAV or FLAC locally. (${error.message})`,
    );
  } finally {
    await context.close();
  }
}

function setStage(active, completed = []) {
  for (const item of elements.stages.querySelectorAll("li")) {
    const step = item.dataset.step;
    item.classList.toggle("active", step === active);
    item.classList.toggle("complete", completed.includes(step));
  }
}

function setProgress(message, { percent = null, stage = null, completed = [] } = {}) {
  elements.status.textContent = message;
  elements.percent.textContent = percent === null ? "" : `${Math.round(percent)}%`;
  elements.bar.classList.toggle("indeterminate", percent === null);
  elements.bar.style.width = percent === null ? "" : `${percent}%`;
  if (stage) setStage(stage, completed);
}

function setRunning(value) {
  running = value;
  elements.drop.disabled = value;
  elements.speakers.disabled = value || !speakersReady;
  elements.speakerCount.disabled = value || !speakersReady;
  elements.speakerNames.disabled = value || !speakersReady;
  elements.gemini.disabled = value || !geminiReady;
  refreshStart();
}

async function errorMessage(response) {
  try {
    const body = await response.json();
    return body.detail ?? response.statusText;
  } catch {
    return response.statusText;
  }
}

function submit(form) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/v1/transcriptions");
    for (const [name, value] of Object.entries(headers())) request.setRequestHeader(name, value);
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      setProgress("Uploading audio…", {
        percent: (event.loaded / event.total) * 100,
        stage: "upload",
        completed: ["extract"],
      });
    });
    request.addEventListener("load", () => {
      let body = {};
      try { body = JSON.parse(request.responseText); } catch { /* empty error */ }
      if (request.status >= 200 && request.status < 300) resolve(body);
      else reject(new Error(body.detail ?? request.statusText));
    });
    request.addEventListener("error", () => reject(new Error("Upload failed")));
    request.send(form);
  });
}

function renderJob(job) {
  if (job.status === "queued") {
    setProgress(job.progress.message, { stage: "model", completed: ["extract", "upload"] });
  } else if (job.status === "running") {
    const stage = job.progress.stage;
    if (["device", "preparing", "loading"].includes(stage)) {
      setProgress(job.progress.message, { stage: "model", completed: ["extract", "upload"] });
    } else if (stage === "transcribing") {
      setProgress(job.progress.message, {
        percent: job.progress.percent ?? null,
        stage: "text",
        completed: ["extract", "upload", "model"],
      });
    } else if (stage.startsWith("diarization")) {
      setProgress(job.progress.message, {
        percent: job.progress.percent ?? null,
        stage: "speakers",
        completed: ["extract", "upload", "model", "text"],
      });
    } else if (stage === "correcting") {
      setProgress(job.progress.message, {
        stage: "correct",
        completed: ["extract", "upload", "model", "text", "speakers"],
      });
    }
  }
  if (job.status === "failed") throw new Error(job.error ?? "Transcription failed");
}

async function observe(job) {
  const response = await fetch(`/v1/transcriptions/${job.id}/events`, {
    headers: { ...headers(), Accept: "text/event-stream" },
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  if (!response.body) throw new Error("SSE is not supported by this browser");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const data = frame.split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!data) continue;
      job = JSON.parse(data);
      renderJob(job);
      if (job.status === "succeeded") return job;
    }
    if (done) break;
  }
  throw new Error("Progress stream ended before transcription completed");
}

async function transcribe() {
  if (!media || running) return;
  sessionStorage.setItem("free-transcribe-token", elements.token.value.trim());
  setRunning(true);
  elements.result.classList.add("hidden");
  elements.progress.classList.remove("hidden");
  setProgress(isVideo(media) ? "Preparing local audio extraction…" : "Preparing upload…", {
    percent: 0,
    stage: "extract",
  });
  try {
    const upload = await extractAudio(media);
    const form = new FormData();
    form.append("file", upload);
    form.append("engine", "parakeet");
    form.append("speakers", String(elements.speakers.checked));
    form.append("gemini", String(elements.gemini.checked));
    if (elements.speakers.checked) {
      const speakerCount = elements.speakerCount.value.trim();
      if (speakerCount) form.append("speaker_count", speakerCount);
      const speakerNames = elements.speakerNames.value.trim();
      if (speakerNames) form.append("speaker_names", speakerNames);
    }
    setProgress("Uploading audio…", { percent: 0, stage: "upload", completed: ["extract"] });
    const submitted = await submit(form);
    const job = await observe(submitted);
    const response = await fetch(job.result_url, { headers: headers() });
    if (!response.ok) throw new Error(await errorMessage(response));
    elements.transcript.value = await response.text();
    fetch(`/v1/transcriptions/${job.id}`, { method: "DELETE", headers: headers() }).catch(() => {});
    elements.result.classList.remove("hidden");
    setProgress("Transcription complete", {
      percent: 100,
      stage: "done",
      completed: ["extract", "upload", "model", "text", "speakers", "correct", "done"],
    });
  } catch (error) {
    setProgress(error.message || String(error), { percent: 0 });
    if (/token|unauthorized/i.test(error.message)) elements.authRow.classList.remove("hidden");
  } finally {
    setRunning(false);
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error();
    const health = await response.json();
    parakeetReady = Boolean(health.ready.engines.parakeet);
    speakersReady = Boolean(health.ready.speakers);
    geminiReady = Boolean(health.ready.gemini);
    elements.speakers.disabled = !speakersReady;
    if (!speakersReady) elements.speakers.checked = false;
    elements.gemini.disabled = !geminiReady;
    if (!geminiReady) elements.gemini.checked = false;
    elements.runtime.textContent = `Server ${health.version} · NVIDIA CUDA · ${health.queue.running} running, ${health.queue.queued} queued`;
    elements.authRow.classList.toggle("hidden", !health.authentication);
    refreshStart();
  } catch {
    elements.runtime.textContent = "Server unavailable";
  }
}

elements.drop.addEventListener("click", () => elements.file.click());
elements.file.addEventListener("change", () => elements.file.files[0] && setMedia(elements.file.files[0]));
for (const eventName of ["dragenter", "dragover"]) {
  elements.drop.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.drop.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.drop.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.drop.classList.remove("dragging");
  });
}
elements.drop.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) setMedia(file);
});
elements.speakers.addEventListener("change", () => {
  refreshSpeakerOptions();
});
elements.start.addEventListener("click", transcribe);
elements.copy.addEventListener("click", async () => {
  if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(elements.transcript.value);
  else {
    elements.transcript.select();
    document.execCommand("copy");
  }
  elements.copy.textContent = "Copied";
  window.setTimeout(() => { elements.copy.textContent = "Copy"; }, 1200);
});
elements.download.addEventListener("click", () => {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([elements.transcript.value], { type: "text/markdown" }));
  link.download = "transcript.md";
  link.click();
  URL.revokeObjectURL(link.href);
});

refreshSpeakerOptions();
checkHealth();
