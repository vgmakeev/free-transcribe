const $ = (selector) => document.querySelector(selector);

const elements = {
  runtime: $("#runtime"),
  authRow: $("#auth-row"),
  token: $("#token"),
  drop: $("#drop"),
  file: $("#file"),
  fileLabel: $("#file-label"),
  profile: $("#profile"),
  profileInfo: $("#profile-info"),
  speakers: $("#speakers"),
  countRow: $("#count-row"),
  speakerCount: $("#speaker-count"),
  start: $("#start"),
  progress: $("#progress"),
  stages: $("#stages"),
  status: $("#status"),
  percent: $("#percent"),
  bar: $("#bar"),
  result: $("#result"),
  transcript: $("#transcript"),
  copy: $("#copy"),
  download: $("#download"),
};

const PROFILES = {
  parakeet: {
    engine: "parakeet",
    model: "",
    size: "2.3 GB",
    note: "fastest on Apple Silicon",
  },
  quality: { engine: "qwen", model: "", size: "4.7 GB", note: "best accuracy" },
};

let media = null;
let running = false;
let readyEngines = new Set();

elements.token.value = sessionStorage.getItem("free-transcribe-token") ?? "";

function headers() {
  const token = elements.token.value.trim();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function setMedia(file) {
  media = file;
  elements.fileLabel.textContent = `${file.name} · ${formatBytes(file.size)}`;
  elements.drop.classList.add("selected");
  elements.start.disabled = running || !readyEngines.has(PROFILES[elements.profile.value].engine);
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function refreshProfile() {
  const profile = PROFILES[elements.profile.value];
  const aligner = elements.speakers.checked && profile.engine === "qwen"
    ? " · +1.8 GB word aligner"
    : "";
  elements.profileInfo.textContent = `First use: ~${profile.size}${aligner} · ${profile.note}. Cached afterward.`;
  elements.countRow.classList.toggle("hidden", !elements.speakers.checked);
  elements.start.disabled = running || !media || !readyEngines.has(profile.engine);
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
  elements.start.disabled = value || !media;
  elements.drop.disabled = value;
  elements.profile.disabled = value;
  elements.speakers.disabled = value;
  elements.speakerCount.disabled = value;
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
    for (const [name, value] of Object.entries(headers())) {
      request.setRequestHeader(name, value);
    }
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      const percent = (event.loaded / event.total) * 100;
      setProgress("Uploading media…", { percent, stage: "upload" });
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
      setProgress(job.progress.message, {
        stage: "model",
        completed: ["upload"],
      });
    } else if (job.status === "running") {
      const stage = job.progress.stage;
      if (["device", "preparing", "loading"].includes(stage)) {
        setProgress(job.progress.message, { stage: "model", completed: ["upload"] });
      } else if (stage === "transcribing") {
        setProgress(job.progress.message, {
          percent: job.progress.percent ?? null,
          stage: "text",
          completed: ["upload", "model"],
        });
      } else if (stage.startsWith("diarization")) {
        setProgress(job.progress.message, {
          percent: job.progress.percent ?? null,
          stage: "speakers",
          completed: ["upload", "model", "text"],
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
  setProgress("Uploading media…", { percent: 0, stage: "upload" });

  const profile = PROFILES[elements.profile.value];
  const form = new FormData();
  form.append("file", media);
  form.append("engine", profile.engine);
  if (profile.model) form.append("model", profile.model);
  form.append("speakers", String(elements.speakers.checked));
  const speakerCount = elements.speakerCount.value.trim();
  if (speakerCount) form.append("speaker_count", speakerCount);

  try {
    const submitted = await submit(form);
    const job = await observe(submitted);
    const response = await fetch(job.result_url, { headers: headers() });
    if (!response.ok) throw new Error(await errorMessage(response));
    elements.transcript.value = await response.text();
    fetch(`/v1/transcriptions/${job.id}`, {
      method: "DELETE",
      headers: headers(),
    }).catch(() => {});
    elements.result.classList.remove("hidden");
    setProgress("Transcription complete", {
      percent: 100,
      stage: "done",
      completed: ["upload", "model", "text", "speakers", "done"],
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
    readyEngines = new Set(
      Object.entries(health.ready.engines)
        .filter(([, ready]) => ready)
        .map(([engine]) => engine),
    );
    for (const option of elements.profile.options) {
      const ready = readyEngines.has(PROFILES[option.value].engine);
      option.disabled = !ready;
      option.hidden = !ready;
    }
    if (!readyEngines.has(PROFILES[elements.profile.value].engine)) {
      const fallback = [...elements.profile.options].find((option) => !option.disabled);
      if (fallback) elements.profile.value = fallback.value;
    }
    elements.speakers.disabled = !health.ready.speakers;
    if (!health.ready.speakers) elements.speakers.checked = false;
    elements.runtime.textContent = `Server ${health.version} · ${health.cuda_required ? "CUDA GPU" : "local"} · ${health.queue.running} running, ${health.queue.queued} queued`;
    elements.authRow.classList.toggle("hidden", !health.authentication);
    refreshProfile();
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
elements.profile.addEventListener("change", refreshProfile);
elements.speakers.addEventListener("change", refreshProfile);
elements.start.addEventListener("click", transcribe);
elements.copy.addEventListener("click", async () => {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(elements.transcript.value);
  } else {
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

refreshProfile();
checkHealth();
