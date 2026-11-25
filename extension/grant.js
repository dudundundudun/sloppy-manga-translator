"use strict";

function isSafeOriginPattern(pattern) {
  if (!/^https?:\/\/[^/\s]+\/\*$/.test(pattern)) {
    return false;
  }
  try {
    const parsed = new URL(pattern.slice(0, -1));
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return false;
    }
    const host = (parsed.hostname || "")
      .trim()
      .toLowerCase()
      .replace(/^\[|\]$/g, "")
      .replace(/\.+$/, "");
    if (!host || host === "localhost" || host.endsWith(".localhost") || host.endsWith(".local")) {
      return false;
    }
    if (/^\d+$/.test(host) || /^0x[0-9a-f]+$/i.test(host)) {
      return false;
    }
    if (
      host === "0.0.0.0" ||
      /^127\./.test(host) ||
      /^10\./.test(host) ||
      /^192\.168\./.test(host) ||
      /^169\.254\./.test(host) ||
      /^0\./.test(host) ||
      /^172\.(1[6-9]|2\d|3[01])\./.test(host) ||
      /^100\.(6[4-9]|[7-9]\d|1([01]\d|2[0-7]))\./.test(host)
    ) {
      return false;
    }
    if (
      host.includes(":") &&
      (
        host === "::1" ||
        host === "::" ||
        host.startsWith("fe80:") ||
        /^f[cd][0-9a-f]{2}:/i.test(host) ||
        host.includes("::ffff:")
      )
    ) {
      return false;
    }
    return true;
  } catch (_) {
    return false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const originEl = document.getElementById("origin-label");
  const statusEl = document.getElementById("status");
  const grantBtn = document.getElementById("btn-grant");
  const cancelBtn = document.getElementById("btn-cancel");

  const params = new URLSearchParams(window.location.search);
  const pattern = String(params.get("origin") || "").trim();

  if (!isSafeOriginPattern(pattern)) {
    originEl.textContent = pattern || "(некорректный origin)";
    statusEl.textContent = "Ошибка: недопустимый origin для выдачи доступа.";
    grantBtn.disabled = true;
  } else {
    originEl.textContent = pattern;
  }

  grantBtn.addEventListener("click", async () => {
    if (!isSafeOriginPattern(pattern)) return;
    grantBtn.disabled = true;
    statusEl.textContent = "Запрос разрешения в Chrome...";
    try {
      const granted = await chrome.permissions.request({ origins: [pattern] });
      if (granted) {
        statusEl.textContent = "Доступ разрешён! Можно закрывать окно.";
        setTimeout(() => window.close(), 350);
      } else {
        statusEl.textContent = "Пользователь отклонил запрос доступа.";
        grantBtn.disabled = false;
      }
    } catch (err) {
      statusEl.textContent = `Ошибка запроса: ${(err && err.message) || String(err)}`;
      grantBtn.disabled = false;
    }
  });

  cancelBtn.addEventListener("click", () => {
    window.close();
  });
});
