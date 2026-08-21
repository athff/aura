// AURA chat frontend (vanilla JS, no framework).
// Talks ONLY to this app's /api/chat endpoint.

(function () {
  "use strict";

    const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const messagesEl = document.getElementById("messages");
  const liveBtn = document.getElementById("live-btn");
  const stopLiveBtn = document.getElementById("stop-live-btn");
  const liveStatusEl = document.getElementById("live-status");
  const stateLabelEl = document.getElementById("stateLabel");
  const body = document.body;

  let busy = false;
  let ws = null;
  let liveMode = false;
  let currentPartialEl = null;
  let currentAudio = null;
  let audioQueue = [];

  let mediaRecorder = null;
  let recordedChunks = [];
  let shouldProcessRecording = true;
  let micAutoResumeEnabled = true;
  let micPermissionDenied = false;

  let audioContext = null;
  let analyserNode = null;
  let vadInterval = null;
  let vadVoiceDetected = false;
  let lastVoiceTime = 0;
  const VAD_THRESHOLD = 0.012;
  const VAD_SILENCE_MS = 700;

  function isWsOpen() {
    return ws && ws.readyState === WebSocket.OPEN;
  }

  function isRecordingActive() {
    return !!(mediaRecorder && mediaRecorder.state && mediaRecorder.state !== "inactive");
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function createBubble(role, text) {
    const article = document.createElement("article");
    article.className = "msg msg--" + role;

    if (role === "assistant") {
      const orbSpan = document.createElement("span");
      orbSpan.className = "orb orb--msg";
      orbSpan.setAttribute("aria-hidden", "true");
      orbSpan.innerHTML = '<svg class="aura-logo" aria-hidden="true"><use href="#auraOrb"/></svg>';
      article.appendChild(orbSpan);
    }

    const bodyEl = document.createElement("div");
    bodyEl.className = "msg__body";

    const meta = document.createElement("span");
    meta.className = "msg__meta";
    if (role === "user") {
      meta.textContent = "You Â· now";
    } else {
      meta.textContent = "AURA Â· now";
    }
    bodyEl.appendChild(meta);

    const p = document.createElement("p");
    if (text !== null && text !== undefined) {
      p.textContent = text;
    }
    bodyEl.appendChild(p);

    article.appendChild(bodyEl);
    messagesEl.appendChild(article);
    scrollToBottom();
    return article;
  }

  function setBusy(value) {
    busy = value;
    sendBtn.disabled = value;
    input.disabled = value;
  }

  function resetInputHeight() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 180) + "px";
  }

    // ----- Visual state registry (mirrors prototype state toolbar) -----
  const STATES = [
    { id: "ready",        label: "READY",          liveStatus: "Ready when you are." },
    { id: "listening",    label: "LISTENING",      liveStatus: "I'm listeningâ€¦" },
    { id: "transcribing", label: "TRANSCRIBING",   liveStatus: "Hearing youâ€¦" },
    { id: "thinking",     label: "THINKING",       liveStatus: "Thinkingâ€¦" },
    { id: "speaking",     label: "SPEAKING",       liveStatus: "Speakingâ€¦" },
    { id: "error",        label: "ERROR",          liveStatus: "Something went wrong." },
    { id: "disconnected", label: "DISCONNECTED",   liveStatus: "Reconnectingâ€¦" }
  ];

  function byStateId(id) {
    for (let i = 0; i < STATES.length; i++) {
      if (STATES[i].id === id) return STATES[i];
    }
    return STATES[0];
  }

  function syncLive() {
    const meta = byStateId(body.getAttribute("data-state"));
    if (stateLabelEl) stateLabelEl.textContent = meta.label;
    if (liveStatusEl) liveStatusEl.textContent = meta.liveStatus;
  }

  // Canonical visual state setter â€” called by WebSocket status messages
  // and by the dev toolbar visual-only buttons. Real WebSocket state always
  // overrides any visual/dev state.
  function setLiveStatus(state) {
    const normalized = state ? String(state).toLowerCase() : "ready";
    body.setAttribute("data-state", normalized);
    syncLive();
    // Hide all state animation elements and show the appropriate one for the current state
    const transcribingEl = document.getElementById("transcribing-process");
    const speakingEl = document.getElementById("speaking-wave");
    const thinkingEl = document.getElementById("thinking-process");
    if (transcribingEl) transcribingEl.hidden = true;
    if (speakingEl) speakingEl.hidden = true;
    if (thinkingEl) thinkingEl.hidden = true;
    if (normalized === "transcribing") {
      if (transcribingEl) transcribingEl.hidden = false;
    } else if (normalized === "speaking") {
      if (speakingEl) speakingEl.hidden = false;
    } else if (normalized === "thinking") {
      if (thinkingEl) thinkingEl.hidden = false;
    } else {
      // "listening" and "ready" — keep animations hidden; status text suffices
      if (transcribingEl) transcribingEl.hidden = true;
      if (speakingEl) speakingEl.hidden = true;
      if (thinkingEl) thinkingEl.hidden = true;
    }
    // brief cross-fade when the voice state changes (180â€“250ms)
    const stage = document.querySelector(".live__stage");
    if (stage) {
      stage.classList.remove("fx-faded");
      void stage.offsetWidth; /* reflow to restart the animation */
      stage.classList.add("fx-faded");
      setTimeout(function () { stage.classList.remove("fx-faded"); }, 260);
    }
  }



  async function sendMessage() {
    const text = input.value.trim();
    if (busy || !text) return;

    input.value = "";
    resetInputHeight();
    createBubble("user", text);

    if (liveMode && isWsOpen()) {
      try {
        currentPartialEl = createBubble("assistant", null);
        ws.send(JSON.stringify({ type: "text", text: text }));
      } catch (err) {
        createBubble("error", "Live connection error: could not send message");
      }
      return;
    }

    const loading = createBubble("assistant", null);
    setBusy(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      const data = await response.json().catch(() => ({}));
      loading.remove();

      if (!response.ok) {
        createBubble("error", data.detail || "Something went wrong. Please try again.");
      } else {
        createBubble("assistant", data.reply || "");
      }
    } catch (err) {
      loading.remove();
      createBubble("error", "Network error: could not reach AURA. Is the server running?");
    } finally {
      setBusy(false);
      input.focus();
      scrollToBottom();
    }
  }

  function cleanupVAD() {
    if (vadInterval) {
      clearInterval(vadInterval);
      vadInterval = null;
    }
    vadVoiceDetected = false;
    lastVoiceTime = 0;
    if (analyserNode) {
      try {
        analyserNode.disconnect();
      } catch (e) {
        // ignore
      }
      analyserNode = null;
    }
    if (audioContext) {
      try {
        audioContext.close();
      } catch (e) {
        // ignore
      }
      audioContext = null;
    }
  }

  function stopRecording(sendForTranscription) {
    if (typeof sendForTranscription !== "boolean") {
      sendForTranscription = true;
    }
    shouldProcessRecording = sendForTranscription;
    cleanupVAD();

    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      try {
        mediaRecorder.stop();
      } catch (e) {
        // ignore
      }
    }
    if (mediaRecorder && mediaRecorder.stream) {
      try {
        mediaRecorder.stream.getTracks().forEach((track) => track.stop());
      } catch (e) {
        // ignore
      }
    }

  }

  function maybeResumeListening(delayMs) {
    const delay = typeof delayMs === "number" ? delayMs : 250;
    setTimeout(() => {
      if (!liveMode || !isWsOpen() || !micAutoResumeEnabled || micPermissionDenied) return;
      if (currentAudio || audioQueue.length) return;
      if (isRecordingActive()) return;
      startRecording();
    }, delay);
  }

  function stopCurrentAudio() {
    audioQueue = [];
    if (!currentAudio) return;
    try {
      currentAudio.pause();
      currentAudio.src = "";
    } catch (e) {
      // ignore
    }
    currentAudio = null;
  }

  // Plays the streamed per-sentence audio messages back-to-back so speech begins
  // before the whole reply is finished (streamed TTS). When the queue drains, the
  // microphone auto-resume runs again.
  function playNextAudio() {
    if (currentAudio) return;
    const item = audioQueue.shift();
    if (!item) {
      // Queue drained -> listening can resume.
      setLiveStatus("Listening");
      maybeResumeListening(320);
      return;
    }
    const audio = new Audio("data:audio/" + (item.fmt || "wav") + ";base64," + item.b64);
    currentAudio = audio;
    audio.addEventListener("playing", () => setLiveStatus("Speaking"));
    audio.addEventListener("ended", () => {
      currentAudio = null;
      playNextAudio();
    });
    audio.play().catch(() => {
      currentAudio = null;
      playNextAudio();
    });
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  function encodeWAVFromAudioBuffer(audioBuffer, targetRate) {
    const rate = targetRate || 16000;
    const channels = audioBuffer.numberOfChannels;
    const base = audioBuffer.getChannelData(0);
    const mono = new Float32Array(base.length);
    mono.set(base);

    if (channels > 1) {
      const ch1 = audioBuffer.getChannelData(1);
      const len = Math.min(mono.length, ch1.length);
      for (let i = 0; i < len; i++) {
        mono[i] = (mono[i] + ch1[i]) / 2;
      }
    }

    const srcRate = audioBuffer.sampleRate;
    let resampled = mono;
    if (srcRate !== rate) {
      const ratio = rate / srcRate;
      const newLen = Math.max(1, Math.round(mono.length * ratio));
      const out = new Float32Array(newLen);
      for (let i = 0; i < newLen; i++) {
        const pos = i / ratio;
        const i0 = Math.floor(pos);
        const i1 = Math.min(i0 + 1, mono.length - 1);
        const frac = pos - i0;
        out[i] = mono[i0] * (1 - frac) + mono[i1] * frac;
      }
      resampled = out;
    }

    const buffer = new ArrayBuffer(44 + resampled.length * 2);
    const view = new DataView(buffer);
    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + resampled.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, rate, true);
    view.setUint32(28, rate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, "data");
    view.setUint32(40, resampled.length * 2, true);

    let offset = 44;
    for (let i = 0; i < resampled.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, resampled[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buffer;
  }

  function arrayBufferToBase64(buffer) {
    let binary = "";
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
    }
    return btoa(binary);
  }

  function startVAD(stream) {
    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const src = audioContext.createMediaStreamSource(stream);
      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 2048;
      src.connect(analyserNode);

      const data = new Float32Array(analyserNode.fftSize);
      vadVoiceDetected = false;
      lastVoiceTime = 0;

      vadInterval = setInterval(() => {
        if (!analyserNode) return;
        analyserNode.getFloatTimeDomainData(data);
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
          sum += data[i] * data[i];
        }
        const rms = Math.sqrt(sum / data.length);
        const now = Date.now();
        if (rms > VAD_THRESHOLD) {
          vadVoiceDetected = true;
          lastVoiceTime = now;
          if (liveMode && isWsOpen()) setLiveStatus("Listening");
        } else if (vadVoiceDetected && now - lastVoiceTime > VAD_SILENCE_MS) {
          stopRecording(true);
        }
      }, 150);
    } catch (err) {
      // If VAD isn't available, user can still manually stop recording.
    }
  }

  function startRecording() {
    if (!liveMode || !isWsOpen()) return;
    if (currentAudio || audioQueue.length) return;
    if (isRecordingActive()) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      createBubble("error", "Microphone not supported in this browser");
      setLiveStatus("Error");
      return;
    }

    recordedChunks = [];
    shouldProcessRecording = true;
    setLiveStatus("Listening");

    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        micPermissionDenied = false;
        let recorder = null;
        try {
          recorder = new MediaRecorder(stream);
        } catch (err) {
          createBubble("error", "Cannot create MediaRecorder: " + err);
          setLiveStatus("Error");

          return;
        }

        mediaRecorder = recorder;
        recorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) {
            recordedChunks.push(event.data);
          }
        };

        recorder.onstop = async () => {
          cleanupVAD();
          const processThisRecording = shouldProcessRecording;


          if (!processThisRecording || !recordedChunks.length) {
            mediaRecorder = null;
            if (liveMode && isWsOpen() && !currentAudio) {
              setLiveStatus("Listening");
              maybeResumeListening(150);
            }
            return;
          }

          setLiveStatus("Transcribing");
          try {
            const blob = new Blob(recordedChunks, { type: "audio/webm" });
            const srcBuffer = await blob.arrayBuffer();
            const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
            const decoded = await decodeCtx.decodeAudioData(srcBuffer);
            const wav = encodeWAVFromAudioBuffer(decoded, 16000);
            try {
              decodeCtx.close();
            } catch (e) {
              // ignore
            }

            if (isWsOpen()) {
              ws.send(JSON.stringify({ type: "audio", format: "wav", data: arrayBufferToBase64(wav) }));
            } else {
              createBubble("error", "Live connection closed before sending audio");
              setLiveStatus("Disconnected");
            }
          } catch (err) {
            createBubble("error", "Could not process recording: " + err);
            setLiveStatus("Error");
            maybeResumeListening(600);
          } finally {
            mediaRecorder = null;
          }
        };

        recorder.start();
        startVAD(stream);
      })
      .catch(() => {
        micPermissionDenied = true;

        createBubble("error", "Microphone permission denied or unavailable");
        setLiveStatus("Error");
      });
  }

  function connectLive() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/ws/live");

    ws.addEventListener("open", () => {
      liveMode = true;
      micAutoResumeEnabled = true;
      liveBtn.hidden = true;
      stopLiveBtn.hidden = false;

      setLiveStatus("Listening");
document.body.classList.add("live-open");
      maybeResumeListening(60);
    });

    ws.addEventListener("message", (event) => {
      let msg = null;
      try {
        msg = JSON.parse(event.data);
      } catch (e) {
        return;
      }
      if (!msg || typeof msg.type !== "string") return;

      if (msg.type === "status") {
        setLiveStatus(msg.state || "Thinking");
      } else if (msg.type === "transcript") {
        createBubble("user", msg.text || "");
        setLiveStatus("Thinking");
      } else if (msg.type === "partial") {
        if (!currentPartialEl) {
          currentPartialEl = createBubble("assistant", "");
        }
        const content = currentPartialEl.querySelector(".bubble > div");
        if (content) content.textContent = msg.text || "";
        setLiveStatus("Thinking");
      } else if (msg.type === "complete") {
        if (currentPartialEl) {
          const content = currentPartialEl.querySelector(".bubble > div");
          if (content) content.textContent = msg.text || "";
          currentPartialEl = null;
        } else {
          createBubble("assistant", msg.text || "");
        }
        setLiveStatus("Ready");
      } else if (msg.type === "audio") {
        stopRecording(false);

        const b64 = msg.data || "";
        const fmt = msg.format || "wav";
        if (!b64) {
          createBubble("error", "Audio payload was empty");
          setLiveStatus("Error");
          maybeResumeListening(500);
          return;
        }
        // Queue each sentence's audio and play them back-to-back so speech starts
        // before the whole reply is finished (streamed TTS).
        audioQueue.push({ b64: b64, fmt: fmt });
        setLiveStatus("Speaking");
        playNextAudio();
      } else if (msg.type === "barge_in") {
        // User spoke while AURA was streaming Kokoro audio.
        // Stop current playback immediately and clear the queue so no further
        // synthesized sentences are sent or played.
        stopCurrentAudio();
        setLiveStatus("Listening");
        maybeResumeListening(320);
      } else if (msg.type === "error") {
        createBubble("error", msg.detail || "Live error");
        setLiveStatus("Error");
        maybeResumeListening(700);
      }
    });

    ws.addEventListener("close", () => {
      liveMode = false;
      currentPartialEl = null;
      micAutoResumeEnabled = false;
      stopRecording(false);
      stopCurrentAudio();
      liveBtn.hidden = false;
      stopLiveBtn.hidden = true;
      setLiveStatus("Disconnected");
      ws = null;
document.body.classList.remove("live-open");
    });

    ws.addEventListener("error", () => {
      setLiveStatus("Error");
    });
  }

  function disconnectLive() {
    micAutoResumeEnabled = false;
    stopRecording(false);
    stopCurrentAudio();
    if (ws) {
      try {
        ws.close();
      } catch (e) {
        // ignore
      }
    }
    ws = null;
    liveMode = false;
    currentPartialEl = null;
    liveBtn.hidden = false;
    stopLiveBtn.hidden = true;
    setLiveStatus("Ready");
  document.body.classList.remove("live-open");
}

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });

  input.addEventListener("input", resetInputHeight);

  liveBtn.addEventListener("click", (event) => {
    event.preventDefault();
    connectLive();
  });

  stopLiveBtn.addEventListener("click", (event) => {
    event.preventDefault();
    disconnectLive();
  });




  setLiveStatus("Ready");
  input.focus();
})();
