/**
 * Background Service Worker (v1.0.0)
 *
 * Изменения v1.0.0:
 * 1. Поддержка параметров `minFontSize`, `maxFontSize`, `detectFreeText` и целевого языка `romaji`.
 * 2. Поддержка остановки пакетного перевода папки (`STOP_FOLDER_TRANSLATION`) и продолжения с места остановки (`skip_existing: true`).
 */

const SERVER_URLS = ["http://127.0.0.1:5000", "http://localhost:5000"];
let activeServerBase = SERVER_URLS[0];

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode.apply(null, chunk);
  }
  return btoa(binary);
}

async function blobToDataUrl(blob) {
  const nativeMime = (blob.type || "").toLowerCase();
  if (
    nativeMime === "image/jpeg" ||
    nativeMime === "image/jpg" ||
    nativeMime === "image/webp" ||
    nativeMime === "image/png"
  ) {
    const buffer = await blob.arrayBuffer();
    return `data:${nativeMime};base64,${arrayBufferToBase64(buffer)}`;
  }

  try {
    if (typeof OffscreenCanvas !== "undefined" && typeof createImageBitmap !== "undefined") {
      const bitmap = await createImageBitmap(blob);
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = canvas.getContext("2d");
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
      const webpBlob = await canvas.convertToBlob({ type: "image/webp", quality: 0.92 });
      const buffer = await webpBlob.arrayBuffer();
      return `data:image/webp;base64,${arrayBufferToBase64(buffer)}`;
    }
  } catch (_) {}

  const mime = nativeMime.startsWith("image/") ? nativeMime : "image/png";
  const buffer = await blob.arrayBuffer();
  return `data:${mime};base64,${arrayBufferToBase64(buffer)}`;
}

async function callLocalServer(endpoint, options = {}, timeoutMs = 180000) {
  let lastError = null;

  for (const baseUrl of [activeServerBase, ...SERVER_URLS.filter((u) => u !== activeServerBase)]) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const res = await fetch(`${baseUrl}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {}),
        },
      });
      clearTimeout(timer);

      if (res.status === 502 || res.status === 503) {
        lastError = new Error(
          `HTTP ${res.status}: VPN в браузере перехватывает локальный адрес ${baseUrl}. Отключите прокси для 127.0.0.1.`
        );
        continue;
      }

      const contentType = res.headers.get("content-type") || "";
      if (!contentType.includes("application/json")) {
        const text = await res.text();
        throw new Error(`Некорректный ответ сервера (${res.status}): ${text.slice(0, 120)}`);
      }

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || data.error || `Ошибка сервера HTTP ${res.status}`);
      }

      activeServerBase = baseUrl;
      return data;
    } catch (err) {
      clearTimeout(timer);
      if (err && err.name === "AbortError") {
        lastError = new Error(`Превышен таймаут ожидания ответа от ${baseUrl}${endpoint}`);
      } else {
        lastError = err;
      }
    }
  }

  throw lastError || new Error("Локальный сервер (127.0.0.1:5000) недоступен");
}

async function fetchImageAsDataUrl(imageUrl) {
  let lastErr = null;

  for (const credMode of ["omit", "include"]) {
    try {
      const res = await fetch(imageUrl, {
        method: "GET",
        credentials: credMode,
        cache: "no-cache",
      });

      if (!res.ok) {
        lastErr = new Error(`HTTP ${res.status} при скачивании картинки`);
        continue;
      }

      const blob = await res.blob();
      if (!blob || blob.size < 100) {
        lastErr = new Error("Скачан пустой файл изображения");
        continue;
      }

      return await blobToDataUrl(blob);
    } catch (err) {
      lastErr = err;
    }
  }

  throw lastErr || new Error("Не удалось скачать изображение со страницы");
}

async function sendMessageToTab(tabId, payload) {
  try {
    await chrome.tabs.sendMessage(tabId, payload);
  } catch (_) {
    try {
      await chrome.scripting.executeScript({
        target: { tabId },
        files: ["content.js"],
      });
      setTimeout(() => {
        chrome.tabs.sendMessage(tabId, payload).catch(() => {});
      }, 150);
    } catch (injErr) {
      console.warn("[MangaTranslator] Не удалось внедрить скрипт:", injErr);
    }
  }
}

chrome.action.onClicked.addListener((tab) => {
  if (tab && tab.id) {
    sendMessageToTab(tab.id, { action: "TOGGLE_PANEL" });
  }
});

chrome.commands.onCommand.addListener(async (command) => {
  if (command === "translate-page") {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.id) {
      sendMessageToTab(tab.id, { action: "TRIGGER_TRANSLATE_PAGE" });
    }
  }
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      switch (msg.action) {
        case "CHECK_HEALTH": {
          const data = await callLocalServer("/health", { method: "GET" }, 25000);
          sendResponse({ ok: true, data });
          break;
        }

        case "GET_MODELS": {
          const refreshParam = msg.refresh ? "?refresh=true" : "?refresh=false";
          const data = await callLocalServer(`/models${refreshParam}`, { method: "GET" }, 25000);
          sendResponse({ ok: true, data });
          break;
        }

        case "SAVE_SERVER_SETTINGS": {
          const data = await callLocalServer(
            "/settings",
            {
              method: "POST",
              body: JSON.stringify({
                unsloth_url: msg.unslothUrl,
                unsloth_api_key: msg.unslothApiKey,
                selected_model: msg.selectedModel,
                auto_save_web_pages: msg.autoSaveWeb !== false,
                web_save_dir: msg.webSaveDir || null,
              }),
            },
            25000
          );
          sendResponse({ ok: true, data });
          break;
        }

        case "PICK_WEB_SAVE_FOLDER": {
          const data = await callLocalServer("/pick_web_save_folder", { method: "POST" }, 60000);
          sendResponse({ ok: true, data });
          break;
        }

        case "OPEN_WEB_SAVE_FOLDER": {
          const q = msg.path ? `?path=${encodeURIComponent(msg.path)}` : "";
          const data = await callLocalServer(`/open_web_save_folder${q}`, { method: "POST" }, 15000);
          sendResponse({ ok: true, data });
          break;
        }

        case "STOP_SERVER": {
          const data = await callLocalServer("/shutdown", { method: "POST" }, 5000);
          sendResponse({ ok: true, data });
          break;
        }

        case "TRANSLATE_IMAGE": {
          let base64Image = msg.imageBase64;
          if (!base64Image) {
            if (!msg.imageUrl) {
              throw new Error("Не найден URL картинки");
            }
            base64Image = await fetchImageAsDataUrl(msg.imageUrl);
          }

          const cleanModel = msg.model && msg.model !== "auto" ? msg.model : null;

          const data = await callLocalServer(
            "/translate",
            {
              method: "POST",
              body: JSON.stringify({
                image: base64Image,
                lang: msg.lang || "auto",
                target_lang: msg.targetLang || "en",
                ollama_model: cleanModel,
                model: cleanModel,
                uppercase: msg.uppercase !== false,
                min_font_size: msg.minFontSize || 12,
                max_font_size: msg.maxFontSize || 32,
                detect_free_text: Boolean(msg.detectFreeText),
                text_filter_mode: msg.textFilterMode || "dialogue_only",
                skip_sighs: msg.skipSighs !== false,
                skip_sfx: msg.skipSfx !== false,
                min_text_length: msg.minTextLength || 2,
                return_image: true,
                image_format: "webp",
                save_debug: false,
                auto_save: msg.autoSave !== false,
                save_dir: msg.saveDir || null,
                chapter_title: msg.chapterTitle || null,
                page_url: msg.pageUrl || null,
                image_url: msg.imageUrl || null,
                page_index: msg.pageIndex || null,
              }),
            },
            240000
          );
          sendResponse({ ok: true, data });
          break;
        }

        case "START_FOLDER_TRANSLATION": {
          const cleanModel = msg.model && msg.model !== "auto" ? msg.model : null;
          const data = await callLocalServer(
            "/translate_folder",
            {
              method: "POST",
              body: JSON.stringify({
                lang: msg.lang || "auto",
                target_lang: msg.targetLang || "en",
                ollama_model: cleanModel,
                model: cleanModel,
                uppercase: msg.uppercase !== false,
                min_font_size: msg.minFontSize || 12,
                max_font_size: msg.maxFontSize || 32,
                detect_free_text: msg.detectFreeText !== false,
                skip_existing: true,
                open_when_done: true,
              }),
            },
            60000
          );
          sendResponse({ ok: true, data });
          break;
        }

        case "STOP_FOLDER_TRANSLATION": {
          const data = await callLocalServer("/folder_stop", { method: "POST" }, 10000);
          sendResponse({ ok: true, data });
          break;
        }

        case "GET_FOLDER_STATUS": {
          const data = await callLocalServer("/folder_status", { method: "GET" }, 10000);
          sendResponse({ ok: true, data });
          break;
        }

        default:
          sendResponse({ ok: false, error: `Неизвестная команда: ${msg.action}` });
      }
    } catch (err) {
      console.error(`[MangaTranslator BG Error] (${msg.action}):`, err);
      sendResponse({
        ok: false,
        error: err && err.message ? err.message : String(err),
      });
    }
  })();

  return true;
});
