// AURA chat frontend (vanilla JS, no framework).
// Talks ONLY to this app's /api/chat endpoint. It never contacts a cloud
// provider directly, and it never sees any API keys.

(function () {
  "use strict";

  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const messagesEl = document.getElementById("messages");

  let busy = false;

  // --- helpers -------------------------------------------------------------

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function createBubble(role, text) {
    const wrapper = document.createElement("div");
    wrapper.className = "message " + role;

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (role === "assistant" && text === null) {
      // Loading indicator: three bouncing dots.
      const dots = document.createElement("div");
      dots.className = "dots";
      dots.innerHTML = "<span></span><span></span><span></span>";
      bubble.appendChild(dots);
    } else {
      const content = document.createElement("div");
      content.textContent = text; // XSS-safe: never inject raw HTML
      bubble.appendChild(content);
    }

    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
  }

  function setBusy(value) {
    busy = value;
    sendBtn.disabled = value;
    input.disabled = value;
  }

  function resetInputHeight() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }

  // --- sending -------------------------------------------------------------

  async function sendMessage() {
    const text = input.value.trim();
    if (busy || !text) return;

    input.value = "";
    resetInputHeight();
    createBubble("user", text);

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

  // --- events --------------------------------------------------------------

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    sendMessage();
  });

  // Enter sends; Shift+Enter makes a new line.
  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });

  // Auto-grow the textarea up to a max height.
  input.addEventListener("input", resetInputHeight);

  // Start with focus on the input.
  input.focus();
})();