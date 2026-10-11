// Local pairing page: shown by the EvoLoRA agent when a website asks to connect.

const params = new URLSearchParams(location.search);
const origin = params.get("origin") || "";
const originEl = document.querySelector("#pair-origin");
const errorEl = document.querySelector("#pair-error");
const allow = document.querySelector("#pair-allow");

function fail(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
}

let validOrigin = false;
try {
  const url = new URL(origin);
  validOrigin = url.origin === origin && (url.protocol === "https:" || ["localhost", "127.0.0.1"].includes(url.hostname));
} catch {
  validOrigin = false;
}
originEl.textContent = validOrigin ? origin : "Unknown website";
if (!validOrigin || !window.opener) {
  allow.disabled = true;
  fail(validOrigin ? "Open this from the EvoLoRA website's Connect button." : "This request has no valid website origin.");
}

allow.addEventListener("click", async () => {
  allow.disabled = true;
  try {
    const response = await fetch("/api/pair", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ origin }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `Pairing failed (${response.status})`);
    // Only the approved origin can receive the token.
    window.opener.postMessage({ type: "evolora-paired", token: body.token }, origin);
    window.close();
  } catch (error) {
    allow.disabled = false;
    fail(error.message);
  }
});

document.querySelector("#pair-deny").addEventListener("click", () => {
  window.opener?.postMessage({ type: "evolora-pair-denied" }, validOrigin ? origin : "*");
  window.close();
});
