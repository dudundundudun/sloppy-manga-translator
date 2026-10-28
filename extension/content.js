/**
 * Content Script (v1.0.8) — Локальный и облачный AI-переводчик Манги / Манхвы / Маньхуа
 *
 * Ключевые улучшения v1.0.8:
 * 1. Полностью переработан интерфейс выбора ИИ-моделей:
 *    - Информативная карточка активной модели с бейджем провайдера (✨ Авто / ☁️ Cloud / 🦙 Ollama),
 *      открывающая сгруппированный список моделей с быстрым поиском.
 *    - Исправлен баг подсветки кнопок быстрого выбора (`✨ Авто`, `⚡ Flash`, `🧠 Pro`, `🦙 Локальная`):
 *      больше никаких ложных срабатываний `startsWith("unsloth::")`, из-за которых клик по `⚡ Flash`
 *      подсвечивал кнопку `🧠 Pro`.
 *    - Переключение модели использует изолированный вызов `SET_ACTIVE_MODEL` с защитой от гонок запросов
 *      и больше НЕ затирает сохраненный `unsloth_api_key` на сервере.
 *    - Поддержка скачивания рекомендованных локальных моделей Ollama в 1 клик (`PULL_MODEL`) с живым прогресс-баром.
 * 2. Улучшено управление переводом:
 *    - Перетаскиваемая шапка панели (с сохранением позиции) и кнопка сворачивания (`—`).
 *    - Кнопка `🎯 Кадр` (перевод только видимых на экране страниц) и кнопка `⏹ Стоп` для остановки перевода страницы или папки.
 *    - Корректная замена и восстановление `<picture><source srcset>` на современных читалках манги.
 */

(() => {
  if (typeof window.__mtCleanup === "function") {
    try {
      window.__mtCleanup();
    } catch (_) {}
  }

  const oldPanel = document.getElementById("mt-floating-panel");
  if (oldPanel) oldPanel.remove();

  const MIN_WIDTH = 280;
  const MIN_HEIGHT = 350;

  const DEFAULT_SETTINGS = {
    lang: "auto",
    targetLang: "en",
    model: "auto",
    lastLocalModel: "",
    uppercase: true,
    minFontSize: 12,
    maxFontSize: 32,
    overflowMode: "shrink",
    textFilterMode: "dialogue_only",
    skipSighs: true,
    skipSfx: true,
    detectFreeText: false,
    minTextLength: 2,
    autoSaveWeb: true,
    webSaveDir: "",
    unslothUrl: "https://moduleapi.pro/v1",
    unslothApiKey: "",
    minimized: false,
    panelPos: null,
  };

  let settings = { ...DEFAULT_SETTINGS };
  let availableModelsCache = [];
  let resolvedServerModel = "";
  let serverHasApiKey = false;
  let showingOriginal = false;
  let folderPollTimer = null;
  let pullPollTimer = null;
  let isCheckingServer = false;
  let pendingForceRefresh = false;
  let modelSwitchSeq = 0;
  let isModelSwitching = false;
  let pageBatchRunning = false;
  let pageStopRequested = false;

  function migrateAndLoadSettings(onLoaded) {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      chrome.storage.local.get(
        [
          "mt_source_lang",
          "mt_target_lang",
          "mt_model_v8",
          "mt_last_local_model",
          "mt_uppercase",
          "mt_min_font_size",
          "mt_max_font_size",
          "mt_overflow_mode",
          "mt_text_filter_mode",
          "mt_skip_sighs",
          "mt_skip_sfx",
          "mt_detect_free_text",
          "mt_min_text_length",
          "mt_auto_save_web",
          "mt_web_save_dir",
          "mt_unsloth_url",
          "mt_unsloth_key",
          "mt_minimized",
          "mt_panel_pos",
        ],
        (stored) => {
          const s = stored || {};
          settings.lang = s.mt_source_lang || DEFAULT_SETTINGS.lang;
          settings.targetLang = s.mt_target_lang || DEFAULT_SETTINGS.targetLang;
          settings.model = s.mt_model_v8 || DEFAULT_SETTINGS.model;
          settings.lastLocalModel = s.mt_last_local_model || "";
          settings.uppercase =
            s.mt_uppercase !== undefined ? Boolean(s.mt_uppercase) : DEFAULT_SETTINGS.uppercase;
          settings.minFontSize =
            parseInt(String(s.mt_min_font_size || DEFAULT_SETTINGS.minFontSize), 10) || 12;
          settings.maxFontSize =
            parseInt(String(s.mt_max_font_size || DEFAULT_SETTINGS.maxFontSize), 10) || 32;
          settings.overflowMode = s.mt_overflow_mode || "shrink";
          settings.textFilterMode = s.mt_text_filter_mode || "dialogue_only";
          settings.skipSighs = s.mt_skip_sighs !== undefined ? Boolean(s.mt_skip_sighs) : true;
          settings.skipSfx = s.mt_skip_sfx !== undefined ? Boolean(s.mt_skip_sfx) : true;
          settings.detectFreeText =
            s.mt_detect_free_text !== undefined ? Boolean(s.mt_detect_free_text) : false;
          settings.minTextLength = parseInt(String(s.mt_min_text_length || 2), 10) || 2;
          settings.autoSaveWeb =
            s.mt_auto_save_web !== undefined ? Boolean(s.mt_auto_save_web) : true;
          settings.webSaveDir = s.mt_web_save_dir || "";
          settings.unslothUrl = s.mt_unsloth_url || DEFAULT_SETTINGS.unslothUrl;
          settings.unslothApiKey = s.mt_unsloth_key || DEFAULT_SETTINGS.unslothApiKey;
          settings.minimized = Boolean(s.mt_minimized);
          settings.panelPos = s.mt_panel_pos || null;
          if (typeof onLoaded === "function") onLoaded();
        }
      );
    } else if (typeof onLoaded === "function") {
      onLoaded();
    }
  }

  function saveLocalSettings() {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      try {
        chrome.storage.local.set({
          mt_source_lang: settings.lang,
          mt_target_lang: settings.targetLang,
          mt_model_v8: settings.model,
          mt_last_local_model: settings.lastLocalModel || "",
          mt_uppercase: Boolean(settings.uppercase),
          mt_min_font_size: Number(settings.minFontSize) || 12,
          mt_max_font_size: Number(settings.maxFontSize) || 32,
          mt_overflow_mode: settings.overflowMode || "shrink",
          mt_text_filter_mode: settings.textFilterMode || "dialogue_only",
          mt_skip_sighs: Boolean(settings.skipSighs),
          mt_skip_sfx: Boolean(settings.skipSfx),
          mt_detect_free_text: Boolean(settings.detectFreeText),
          mt_min_text_length: Number(settings.minTextLength) || 2,
          mt_auto_save_web: Boolean(settings.autoSaveWeb),
          mt_web_save_dir: settings.webSaveDir || "",
          mt_unsloth_url: settings.unslothUrl,
          mt_unsloth_key: settings.unslothApiKey,
          mt_minimized: Boolean(settings.minimized),
          mt_panel_pos: settings.panelPos || null,
        });
      } catch (_) {}
    }
  }

  function safeSendMessage(payload) {
    return new Promise((resolve) => {
      try {
        if (!chrome || !chrome.runtime || !chrome.runtime.id) {
          resolve({
            ok: false,
            error: "Расширение было обновлено. Обновите вкладку (F5).",
          });
          return;
        }
        chrome.runtime.sendMessage(payload, (response) => {
          if (chrome.runtime.lastError) {
            resolve({
              ok: false,
              error: chrome.runtime.lastError.message || "Ошибка связи с фоновым скриптом",
            });
            return;
          }
          resolve(response || { ok: false, error: "Пустой ответ от фонового скрипта" });
        });
      } catch (err) {
        resolve({
          ok: false,
          error: err && err.message ? err.message : "Контекст расширения недействителен (нажмите F5)",
        });
      }
    });
  }

  function triggerCustomProtocol(url) {
    const a = document.createElement("a");
    a.href = url;
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    setTimeout(() => a.remove(), 500);
  }

  const oldStyle = document.getElementById("mt-floating-styles");
  if (oldStyle) oldStyle.remove();

  const style = document.createElement("style");
  style.id = "mt-floating-styles";
  style.textContent = `
    #mt-floating-panel {
      position: fixed;
      top: 14px;
      right: 18px;
      z-index: 2147483647;
      display: none;
      flex-direction: column;
      background: rgba(15, 16, 24, 0.98);
      color: #f3f4f6;
      border-radius: 14px;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 13px;
      box-shadow: 0 16px 42px rgba(0, 0, 0, 0.78);
      border: 1px solid rgba(255, 255, 255, 0.16);
      backdrop-filter: blur(16px);
      width: 356px;
      max-height: calc(100vh - 28px);
      overflow: hidden;
      user-select: none;
    }
    #mt-floating-panel.mt-open {
      display: flex !important;
    }
    #mt-floating-panel.mt-minimized {
      width: 268px;
    }
    #mt-floating-panel.mt-minimized #mt-panel-body {
      display: none !important;
    }
    .mt-header-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      padding: 10px 13px 8px 13px;
      cursor: grab;
      background: rgba(255, 255, 255, 0.02);
      flex-shrink: 0;
    }
    .mt-header-row:active {
      cursor: grabbing;
    }
    #mt-floating-panel.mt-minimized .mt-header-row {
      border-bottom: none;
      padding-bottom: 10px;
    }
    #mt-panel-body {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 10px 14px 13px 14px;
      overflow-y: auto;
      max-height: calc(100vh - 76px);
      scrollbar-width: thin;
      scrollbar-color: rgba(129, 140, 248, 0.4) transparent;
    }
    .mt-title {
      font-weight: 700;
      font-size: 13px;
      color: #a5b4fc;
      display: flex;
      align-items: center;
      gap: 7px;
      pointer-events: none;
    }
    .mt-ver-tag {
      font-size: 10px;
      font-weight: 700;
      padding: 1px 5px;
      border-radius: 5px;
      background: rgba(99, 102, 241, 0.25);
      color: #c7d2fe;
      border: 1px solid rgba(129, 140, 248, 0.35);
    }
    .mt-server-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #ef4444;
      display: inline-block;
      box-shadow: 0 0 6px #ef4444;
      flex-shrink: 0;
    }
    .mt-server-dot.online {
      background: #22c55e;
      box-shadow: 0 0 8px #22c55e;
    }
    .mt-icon-btn {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #e5e7eb;
      border-radius: 6px;
      padding: 3px 7px;
      cursor: pointer;
      font-size: 12px;
      line-height: 1.2;
      transition: background 0.15s, border-color 0.15s;
    }
    .mt-icon-btn:hover {
      background: rgba(255, 255, 255, 0.18);
      border-color: rgba(255, 255, 255, 0.3);
    }
    .mt-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .mt-label {
      font-size: 11.5px;
      color: #9ca3af;
      white-space: nowrap;
    }
    .mt-select, .mt-input {
      background: #1c1d29;
      color: #fff;
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 7px;
      padding: 5px 8px;
      font-size: 12px;
      outline: none;
      flex: 1;
      min-width: 0;
      cursor: pointer;
      pointer-events: auto !important;
    }
    .mt-select:focus, .mt-input:focus {
      border-color: #818cf8;
    }
    .mt-input {
      cursor: text;
      user-select: text;
    }
    /* Карточка выбора активной модели */
    #mt-active-model-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      background: linear-gradient(135deg, rgba(30, 32, 48, 0.95), rgba(22, 24, 36, 0.95));
      border: 1px solid rgba(129, 140, 248, 0.35);
      border-radius: 9px;
      padding: 7px 10px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    #mt-active-model-bar:hover {
      border-color: #818cf8;
      background: linear-gradient(135deg, rgba(40, 43, 66, 0.98), rgba(26, 28, 44, 0.98));
    }
    #mt-active-model-bar.mt-open {
      border-color: #a5b4fc;
      box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25);
    }
    .mt-model-bar-left {
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 0;
      flex: 1;
    }
    .mt-model-bar-top {
      display: flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
    }
    .mt-provider-badge {
      font-size: 10px;
      font-weight: 700;
      padding: 1.5px 6px;
      border-radius: 5px;
      text-transform: uppercase;
      letter-spacing: 0.3px;
      white-space: nowrap;
      flex-shrink: 0;
    }
    .mt-provider-badge.auto {
      background: rgba(99, 102, 241, 0.3);
      color: #c7d2fe;
      border: 1px solid rgba(129, 140, 248, 0.45);
    }
    .mt-provider-badge.cloud {
      background: rgba(16, 185, 129, 0.25);
      color: #6ee7b7;
      border: 1px solid rgba(52, 211, 153, 0.45);
    }
    .mt-provider-badge.ollama {
      background: rgba(245, 158, 11, 0.25);
      color: #fcd34d;
      border: 1px solid rgba(251, 191, 36, 0.45);
    }
    #mt-active-model-title {
      font-size: 12px;
      font-weight: 700;
      color: #f9fafb;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    #mt-active-model-sub {
      font-size: 10.5px;
      color: #9ca3af;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .mt-model-bar-chevron {
      font-size: 11px;
      color: #a5b4fc;
      font-weight: 700;
      padding: 2px 5px;
      border-radius: 5px;
      background: rgba(255, 255, 255, 0.06);
      flex-shrink: 0;
    }
    /* Быстрые пресет-кнопки */
    .mt-provider-pills {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 5px;
      width: 100%;
    }
    .mt-pill-btn {
      background: rgba(255, 255, 255, 0.06);
      color: #d1d5db;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 7px;
      padding: 5px 4px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      text-align: center;
      transition: all 0.15s ease;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .mt-pill-btn:hover:not(:disabled) {
      background: rgba(255, 255, 255, 0.15);
      border-color: rgba(255, 255, 255, 0.28);
    }
    .mt-pill-btn:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .mt-pill-btn.active {
      background: rgba(79, 70, 229, 0.42);
      border-color: #818cf8;
      color: #ffffff;
      box-shadow: 0 0 10px rgba(99, 102, 241, 0.3);
    }
    .mt-pill-btn.flash.active {
      background: rgba(14, 165, 233, 0.38);
      border-color: #38bdf8;
      color: #ffffff;
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.28);
    }
    .mt-pill-btn.pro.active {
      background: rgba(5, 150, 105, 0.44);
      border-color: #34d399;
      color: #ffffff;
      box-shadow: 0 0 10px rgba(52, 211, 153, 0.28);
    }
    .mt-pill-btn.local.active {
      background: rgba(217, 119, 6, 0.42);
      border-color: #fbbf24;
      color: #ffffff;
      box-shadow: 0 0 10px rgba(251, 191, 36, 0.28);
    }
    /* Выпадающий список моделей с категориями и поиском */
    #mt-custom-model-picker {
      display: none;
      flex-direction: column;
      gap: 5px;
      max-height: 235px;
      overflow-y: auto;
      background: #12131c;
      border: 1px solid rgba(129, 140, 248, 0.35);
      border-radius: 9px;
      padding: 7px;
      scrollbar-width: thin;
    }
    .mt-model-search {
      background: #1b1c28;
      color: #f3f4f6;
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 6px;
      padding: 5px 8px;
      font-size: 11.5px;
      outline: none;
      width: 100%;
      box-sizing: border-box;
      margin-bottom: 2px;
      user-select: text;
    }
    .mt-model-search:focus {
      border-color: #818cf8;
    }
    .mt-model-group-title {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      color: #9ca3af;
      padding: 4px 4px 2px 4px;
      border-top: 1px solid rgba(255, 255, 255, 0.07);
      margin-top: 2px;
    }
    .mt-model-group-title:first-of-type {
      border-top: none;
      margin-top: 0;
    }
    .mt-model-item {
      background: rgba(255, 255, 255, 0.04);
      color: #f3f4f6;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 11.5px;
      text-align: left;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 6px;
      transition: background 0.12s, border-color 0.12s;
    }
    .mt-model-item:hover {
      background: rgba(99, 102, 241, 0.24);
      border-color: rgba(129, 140, 248, 0.45);
    }
    .mt-model-item.selected {
      background: rgba(79, 70, 229, 0.36);
      border-color: #818cf8;
      font-weight: 600;
    }
    .mt-model-item.downloadable {
      background: rgba(245, 158, 11, 0.08);
      border-color: rgba(245, 158, 11, 0.22);
      color: #fde68a;
    }
    .mt-model-item.downloadable:hover {
      background: rgba(245, 158, 11, 0.2);
      border-color: rgba(251, 191, 36, 0.5);
    }
    .mt-model-item-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1;
    }
    .mt-model-item-tag {
      font-size: 10px;
      padding: 1px 5px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.1);
      color: #d1d5db;
      white-space: nowrap;
      flex-shrink: 0;
    }
    .mt-btn {
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 8px 10px;
      font-weight: 600;
      font-size: 12.5px;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.1s;
      text-align: center;
    }
    .mt-btn:hover:not(:disabled) { opacity: 0.93; }
    .mt-btn:active:not(:disabled) { transform: scale(0.98); }
    .mt-btn:disabled { opacity: 0.55; cursor: not-allowed; }
    .mt-btn.secondary {
      background: rgba(255, 255, 255, 0.1);
      font-weight: 500;
    }
    .mt-btn.danger {
      background: linear-gradient(135deg, #dc2626, #b91c1c);
      color: #fff;
    }
    .mt-btn.folder {
      background: linear-gradient(135deg, #059669, #047857);
    }
    .mt-btn.server-start {
      background: linear-gradient(135deg, #ea580c, #c2410c);
    }
    .mt-status {
      font-size: 11.5px;
      color: #d1d5db;
      text-align: center;
      min-height: 16px;
      word-break: break-word;
      padding-top: 2px;
    }
    #mt-settings-drawer {
      display: none;
      flex-direction: column;
      gap: 7px;
      background: rgba(0, 0, 0, 0.38);
      padding: 9px;
      border-radius: 9px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      margin-top: 2px;
    }
    .mt-progress-box {
      display: none;
      flex-direction: column;
      gap: 5px;
      background: rgba(255, 255, 255, 0.06);
      padding: 7px 9px;
      border-radius: 8px;
      font-size: 11px;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .mt-progress-bar-bg {
      width: 100%;
      height: 6px;
      background: rgba(255, 255, 255, 0.15);
      border-radius: 4px;
      overflow: hidden;
    }
    .mt-progress-bar-fill {
      width: 0%;
      height: 100%;
      background: #10b981;
      transition: width 0.25s ease;
    }
    #mt-badges-layer {
      position: fixed !important;
      top: 0 !important;
      left: 0 !important;
      width: 0 !important;
      height: 0 !important;
      overflow: visible !important;
      z-index: 2147483640 !important;
      pointer-events: none !important;
    }
    .mt-translate-badge {
      position: fixed !important;
      top: -9999px;
      right: auto;
      z-index: 2147483640 !important;
      background: rgba(17, 24, 39, 0.9) !important;
      color: #fff !important;
      border: 1px solid rgba(255, 255, 255, 0.25) !important;
      border-radius: 7px !important;
      padding: 5px 10px !important;
      font-size: 12px !important;
      font-weight: 600 !important;
      line-height: 1.2 !important;
      cursor: pointer !important;
      opacity: 0;
      pointer-events: none !important;
      transition: opacity 0.18s ease !important;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5) !important;
    }
    .mt-translate-badge.mt-visible,
    .mt-translate-badge:hover,
    .mt-translate-badge.mt-loading,
    .mt-translate-badge.mt-error {
      opacity: 1 !important;
      pointer-events: auto !important;
    }
  `;
  document.head.appendChild(style);

  const panel = document.createElement("div");
  panel.id = "mt-floating-panel";
  panel.innerHTML = `
    <div id="mt-header-drag" class="mt-header-row" title="Перетащите за шапку в удобное место экрана">
      <div class="mt-title">
        <span id="mt-server-dot" class="mt-server-dot" title="Статус локального сервера"></span>
        <span>Manga Translator</span>
        <span class="mt-ver-tag">v1.0.8</span>
      </div>
      <div style="display:flex; gap:4px;">
        <button id="mt-web-ui-btn" class="mt-icon-btn" title="Открыть локальную Веб-Студию (http://127.0.0.1:5000/)">🌐</button>
        <button id="mt-refresh-btn" class="mt-icon-btn" title="Обновить статус и список моделей">🔄</button>
        <button id="mt-settings-toggle" class="mt-icon-btn" title="Настройки API, шрифта, фильтрации и сохранений">⚙️</button>
        <button id="mt-minimize-btn" class="mt-icon-btn" title="Свернуть/развернуть панель">—</button>
        <button id="mt-close-panel-btn" class="mt-icon-btn" title="Скрыть окно">✕</button>
      </div>
    </div>

    <div id="mt-panel-body">
      <div class="mt-row">
        <span class="mt-label">Языки:</span>
        <select id="mt-source-lang" class="mt-select" title="Язык оригинала">
          <option value="auto">✨ Авто (JA/KO/ZH)</option>
          <option value="ja">🇯🇵 Японский</option>
          <option value="ko">🇰🇷 Корейский</option>
          <option value="zh">🇨🇳 Китайский</option>
        </select>
        <span style="color:#6b7280; font-size:12px;">→</span>
        <select id="mt-target-lang" class="mt-select" title="Язык перевода">
          <option value="en">🇬🇧 English</option>
          <option value="ru">🇷🇺 Русский</option>
          <option value="romaji">🔤 Romaji</option>
        </select>
      </div>

      <!-- Карточка текущей ИИ-модели -->
      <div id="mt-active-model-bar" title="Нажмите, чтобы выбрать любую облачную или локальную ИИ-модель">
        <div class="mt-model-bar-left">
          <div class="mt-model-bar-top">
            <span id="mt-provider-badge" class="mt-provider-badge auto">✨ АВТО</span>
            <span id="mt-active-model-title">Автовыбор лучшей модели</span>
          </div>
          <div id="mt-active-model-sub">Проверка доступных моделей...</div>
        </div>
        <span id="mt-model-chevron" class="mt-model-bar-chevron">📋 ▾</span>
      </div>

      <!-- 4 четкие кнопки быстрого переключения без конфликтов подсветки -->
      <div class="mt-provider-pills">
        <button id="mt-pill-auto" type="button" class="mt-pill-btn" title="Автоматический выбор лучшей доступной модели">✨ Авто</button>
        <button id="mt-pill-flash" type="button" class="mt-pill-btn flash" title="DeepSeek V4 Flash — быстрый и экономный облачный перевод">⚡ Flash</button>
        <button id="mt-pill-pro" type="button" class="mt-pill-btn pro" title="DeepSeek V4 Pro — максимальное литературное качество">🧠 Pro</button>
        <button id="mt-pill-local" type="button" class="mt-pill-btn local" title="Локальная модель Ollama (100% на вашей видеокарте без цензуры)">🦙 Локальная</button>
      </div>

      <!-- Выпадающий список всех моделей с поиском и категориями -->
      <div id="mt-custom-model-picker">
        <input id="mt-model-search-input" class="mt-model-search" type="text" placeholder="🔍 Поиск модели (qwen, shisa, deepseek, gemma)..." />
        <div id="mt-model-list-items" style="display:flex; flex-direction:column; gap:4px;"></div>
      </div>

      <!-- Прогресс скачивания локальной модели Ollama -->
      <div id="mt-pull-progress-wrap" class="mt-progress-box">
        <div id="mt-pull-progress-text">📥 Скачивание модели в Ollama...</div>
        <div class="mt-progress-bar-bg"><div id="mt-pull-progress-fill" class="mt-progress-bar-fill" style="background:#f59e0b;"></div></div>
      </div>

      <button id="mt-start-server-btn" class="mt-btn server-start" style="display:none;">
        🚀 Запустить локальный сервер
      </button>

      <div style="display:flex; gap:6px;">
        <button id="mt-translate-all-btn" class="mt-btn" style="flex:1;" title="Перевести все страницы манги на вкладке (Alt+T)">⚡ Перевести стр. (Alt+T)</button>
        <button id="mt-translate-viewport-btn" class="mt-btn secondary" title="Перевести только кадры, которые сейчас видны на экране">🎯 Кадр</button>
        <button id="mt-stop-page-btn" class="mt-btn danger" style="display:none;" title="Остановить пакетный перевод текущей страницы">⏹ Стоп</button>
      </div>

      <div id="mt-page-progress-wrap" class="mt-progress-box">
        <div id="mt-page-progress-text">Перевод страниц...</div>
        <div class="mt-progress-bar-bg"><div id="mt-page-progress-fill" class="mt-progress-bar-fill" style="background:#6366f1;"></div></div>
      </div>

      <div style="display:flex; gap:6px;">
        <button id="mt-folder-btn" class="mt-btn folder" style="flex:1;">📁 Перевести папку</button>
        <button id="mt-open-saved-btn" class="mt-btn secondary" title="Открыть папку на ПК, куда автоматически сохраняется переведенная манга из браузера">📂 Сохранения</button>
      </div>

      <div id="mt-folder-progress-wrap" class="mt-progress-box">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:6px;">
          <div id="mt-folder-progress-text" style="flex:1;">Подготовка...</div>
          <button id="mt-folder-stop-btn" type="button" class="mt-icon-btn" style="color:#fca5a5; border-color:rgba(248,113,113,0.4);" title="Остановить перевод папки (прогресс сохранится)">⏹ Стоп</button>
        </div>
        <div class="mt-progress-bar-bg"><div id="mt-progress-bar-fill" class="mt-progress-bar-fill"></div></div>
      </div>

      <button id="mt-toggle-orig-btn" class="mt-btn secondary" style="display:none;">
        👁 Показать оригинал
      </button>

      <div id="mt-settings-drawer">
        <div class="mt-row">
          <label class="mt-label" style="cursor:pointer; display:flex; align-items:center; gap:6px; color:#34d399; font-weight:600;">
            <input id="mt-autosave-chk" type="checkbox" checked />
            💾 Автосохранение перевода на ПК
          </label>
        </div>
        <div class="mt-row" id="mt-save-dir-row">
          <span class="mt-label" title="Папка автосохранения манги из браузера">Папка:</span>
          <input id="mt-web-save-dir" class="mt-input" type="text" placeholder="По умолчанию: Saved_Manga" />
          <button id="mt-pick-save-dir-btn" type="button" class="mt-icon-btn" title="Выбрать папку на компьютере">📁</button>
        </div>
        <div class="mt-row">
          <span class="mt-label">Cloud API URL:</span>
          <input id="mt-unsloth-url" class="mt-input" type="text" placeholder="https://moduleapi.pro/v1" />
        </div>
        <div class="mt-row">
          <span class="mt-label">API Key:</span>
          <input id="mt-unsloth-key" class="mt-input" type="password" placeholder="Оставьте пустым, чтобы не менять" />
        </div>
        <div class="mt-row">
          <span class="mt-label">ID модели:</span>
          <input id="mt-manual-model" class="mt-input" type="text" placeholder="auto или ollama::qwen2.5:14b" />
        </div>
        <div class="mt-row">
          <span class="mt-label">Мин. шрифт: <b id="mt-min-font-val">12</b>px</span>
          <input id="mt-min-font-slider" type="range" min="8" max="36" step="1" value="12" style="flex:1;" />
        </div>
        <div class="mt-row">
          <span class="mt-label">Макс. шрифт: <b id="mt-font-val">32</b>px</span>
          <input id="mt-font-slider" type="range" min="14" max="64" step="2" value="32" style="flex:1;" />
        </div>
        <div class="mt-row">
          <span class="mt-label">Переполнение:</span>
          <select id="mt-overflow-mode-sel" class="mt-select">
            <option value="shrink">📐 Уменьшать шрифт (внутри баббла)</option>
            <option value="expand_bubble">🎈 Расширять тесный баббл</option>
            <option value="overflow">🔓 Разрешить выход за края</option>
          </select>
        </div>
        <div class="mt-row">
          <span class="mt-label">Объем текста:</span>
          <select id="mt-filter-mode-sel" class="mt-select">
            <option value="dialogue_only">💬 Только диалоги (без вздохов/SFX)</option>
            <option value="bubbles_no_sighs">💭 Диалоги + фон (без вздохов/SFX)</option>
            <option value="long_only">🛡 Только длинные фразы (4+ симв.)</option>
            <option value="bubbles_only">🗨 Все бабблы (со вздохами)</option>
            <option value="all">🔊 Переводить вообще всё</option>
          </select>
        </div>
        <div class="mt-row">
          <label class="mt-label" style="cursor:pointer; display:flex; align-items:center; gap:6px; color:#e5e7eb;">
            <input id="mt-skip-sighs-chk" type="checkbox" checked />
            🚫 Не переводить вздохи (а, о, ах, ох, мм)
          </label>
        </div>
        <div class="mt-row">
          <label class="mt-label" style="cursor:pointer; display:flex; align-items:center; gap:6px; color:#e5e7eb;">
            <input id="mt-skip-sfx-chk" type="checkbox" checked />
            🚫 Не переводить звуки фона (SFX)
          </label>
        </div>
        <div class="mt-row">
          <label class="mt-label" style="cursor:pointer; display:flex; align-items:center; gap:6px; color:#e5e7eb;">
            <input id="mt-free-text-chk" type="checkbox" />
            🔍 Переводить текст ВНЕ бабблов (на фоне)
          </label>
        </div>
        <div class="mt-row">
          <label class="mt-label" style="cursor:pointer; display:flex; align-items:center; gap:6px; color:#e5e7eb;">
            <input id="mt-uppercase-chk" type="checkbox" checked />
            Заглавные буквы (КОМИКС-СТИЛЬ)
          </label>
        </div>
        <div class="mt-row" style="margin-top:2px;">
          <button id="mt-save-settings-btn" class="mt-btn secondary" style="flex:1; padding:6px;">💾 Сохранить настройки</button>
          <button id="mt-stop-server-btn" class="mt-btn secondary" style="color:#fca5a5; padding:6px 9px;" title="Выключить локальный сервер">⏹ Сервер</button>
        </div>
      </div>

      <div id="mt-status-text" class="mt-status">Нажмите 🔄 для проверки сервера</div>
    </div>
  `;
  document.body.appendChild(panel);

  ["mousedown", "mouseup", "click", "dblclick", "keydown"].forEach((evtName) => {
    panel.addEventListener(evtName, (e) => {
      e.stopPropagation();
    });
  });

  const headerDrag = document.getElementById("mt-header-drag");
  const serverDot = document.getElementById("mt-server-dot");
  const minimizeBtn = document.getElementById("mt-minimize-btn");
  const sourceLangSelect = document.getElementById("mt-source-lang");
  const targetLangSelect = document.getElementById("mt-target-lang");
  const activeModelBar = document.getElementById("mt-active-model-bar");
  const providerBadge = document.getElementById("mt-provider-badge");
  const activeModelTitle = document.getElementById("mt-active-model-title");
  const activeModelSub = document.getElementById("mt-active-model-sub");
  const modelChevron = document.getElementById("mt-model-chevron");
  const pillAuto = document.getElementById("mt-pill-auto");
  const pillFlash = document.getElementById("mt-pill-flash");
  const pillPro = document.getElementById("mt-pill-pro");
  const pillLocal = document.getElementById("mt-pill-local");
  const customModelPicker = document.getElementById("mt-custom-model-picker");
  const modelSearchInput = document.getElementById("mt-model-search-input");
  const modelListItems = document.getElementById("mt-model-list-items");
  const pullProgressWrap = document.getElementById("mt-pull-progress-wrap");
  const pullProgressText = document.getElementById("mt-pull-progress-text");
  const pullProgressFill = document.getElementById("mt-pull-progress-fill");
  const startServerBtn = document.getElementById("mt-start-server-btn");
  const translateAllBtn = document.getElementById("mt-translate-all-btn");
  const translateViewportBtn = document.getElementById("mt-translate-viewport-btn");
  const stopPageBtn = document.getElementById("mt-stop-page-btn");
  const pageProgressWrap = document.getElementById("mt-page-progress-wrap");
  const pageProgressText = document.getElementById("mt-page-progress-text");
  const pageProgressFill = document.getElementById("mt-page-progress-fill");
  const folderBtn = document.getElementById("mt-folder-btn");
  const folderProgressWrap = document.getElementById("mt-folder-progress-wrap");
  const folderProgressText = document.getElementById("mt-folder-progress-text");
  const folderStopBtn = document.getElementById("mt-folder-stop-btn");
  const progressBarFill = document.getElementById("mt-progress-bar-fill");
  const toggleOrigBtn = document.getElementById("mt-toggle-orig-btn");
  const settingsDrawer = document.getElementById("mt-settings-drawer");
  const unslothUrlInput = document.getElementById("mt-unsloth-url");
  const unslothKeyInput = document.getElementById("mt-unsloth-key");
  const manualModelInput = document.getElementById("mt-manual-model");
  const minFontSlider = document.getElementById("mt-min-font-slider");
  const minFontVal = document.getElementById("mt-min-font-val");
  const fontSlider = document.getElementById("mt-font-slider");
  const fontVal = document.getElementById("mt-font-val");
  const overflowModeSel = document.getElementById("mt-overflow-mode-sel");
  const autoSaveChk = document.getElementById("mt-autosave-chk");
  const webSaveDirInput = document.getElementById("mt-web-save-dir");
  const pickSaveDirBtn = document.getElementById("mt-pick-save-dir-btn");
  const openSavedBtn = document.getElementById("mt-open-saved-btn");
  const filterModeSel = document.getElementById("mt-filter-mode-sel");
  const skipSighsChk = document.getElementById("mt-skip-sighs-chk");
  const skipSfxChk = document.getElementById("mt-skip-sfx-chk");
  const freeTextChk = document.getElementById("mt-free-text-chk");
  const uppercaseChk = document.getElementById("mt-uppercase-chk");
  const statusText = document.getElementById("mt-status-text");

  /* Перетаскивание панели за шапку */
  let isDraggingPanel = false;
  let dragOffsetX = 0;
  let dragOffsetY = 0;

  function applySavedPanelPosition() {
    if (settings.panelPos && typeof settings.panelPos.left === "number" && typeof settings.panelPos.top === "number") {
      const maxLeft = Math.max(10, window.innerWidth - 280);
      const maxTop = Math.max(10, window.innerHeight - 80);
      const clampedLeft = Math.min(Math.max(8, settings.panelPos.left), maxLeft);
      const clampedTop = Math.min(Math.max(8, settings.panelPos.top), maxTop);
      panel.style.left = `${clampedLeft}px`;
      panel.style.top = `${clampedTop}px`;
      panel.style.right = "auto";
    }
  }

  headerDrag.addEventListener("mousedown", (e) => {
    if (e.target.closest("button")) return;
    isDraggingPanel = true;
    const rect = panel.getBoundingClientRect();
    dragOffsetX = e.clientX - rect.left;
    dragOffsetY = e.clientY - rect.top;
    e.preventDefault();
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDraggingPanel) return;
    const maxLeft = Math.max(8, window.innerWidth - panel.offsetWidth - 8);
    const maxTop = Math.max(8, window.innerHeight - 54);
    const newLeft = Math.min(Math.max(8, e.clientX - dragOffsetX), maxLeft);
    const newTop = Math.min(Math.max(8, e.clientY - dragOffsetY), maxTop);
    panel.style.left = `${Math.round(newLeft)}px`;
    panel.style.top = `${Math.round(newTop)}px`;
    panel.style.right = "auto";
    settings.panelPos = { left: Math.round(newLeft), top: Math.round(newTop) };
  });

  window.addEventListener("mouseup", () => {
    if (isDraggingPanel) {
      isDraggingPanel = false;
      saveLocalSettings();
    }
  });

  minimizeBtn.addEventListener("click", () => {
    settings.minimized = !settings.minimized;
    panel.classList.toggle("mt-minimized", settings.minimized);
    minimizeBtn.textContent = settings.minimized ? "▢" : "—";
    saveLocalSettings();
  });

  function formatModelDisplayName(modelId) {
    const raw = String(modelId || "auto").trim();
    if (!raw || raw === "auto") return "Автовыбор лучшей модели";
    const found = availableModelsCache.find((m) => m && m.id === raw);
    if (found && found.name) {
      return found.name.replace(/^\[.*?\]\s*/, "").replace(/^[🚀🦙☁️📥✨]+\s*/, "").trim();
    }
    return raw.replace(/^(?:unsloth|ollama)::/, "");
  }

  // Сервер с v1.0.9 отдаёт installed как boolean, старые версии — как строки "true"/"false".
  // Строгое === false / !== false на строках ломало оба списка, поэтому сравниваем через эти хелперы.
  function isModelInstalled(m) {
    return !!m && (m.installed === true || m.installed === "true");
  }

  function isModelDownloadable(m) {
    return !!m && (m.installed === false || m.installed === "false");
  }

  function getBestInstalledLocalModel() {
    const installedLocal = availableModelsCache.filter(
      (m) => m && m.provider === "ollama" && isModelInstalled(m) && String(m.id || "").startsWith("ollama::")
    );
    if (settings.lastLocalModel && installedLocal.some((m) => m.id === settings.lastLocalModel)) {
      return settings.lastLocalModel;
    }
    if (installedLocal.length > 0) {
      return installedLocal[0].id;
    }
    return null;
  }

  function updateProviderPillsUI() {
    const sel = String(settings.model || "auto").trim();
    const eff = sel === "auto" ? String(resolvedServerModel || "auto") : sel;

    const isAuto = !sel || sel === "auto";
    const isFlash =
      !isAuto &&
      (sel === "unsloth::deepseek-v4-flash" ||
        sel.endsWith("/deepseek-v4-flash") ||
        sel.endsWith("::deepseek-chat"));
    const isPro =
      !isAuto &&
      (sel === "unsloth::deepseek-v4-pro" ||
        sel.endsWith("/deepseek-v4-pro") ||
        sel.endsWith("::deepseek-reasoner"));
    const isLocal = !isAuto && sel.startsWith("ollama::");

    pillAuto.classList.toggle("active", isAuto);
    pillFlash.classList.toggle("active", isFlash);
    pillPro.classList.toggle("active", isPro);
    pillLocal.classList.toggle("active", isLocal);

    [pillAuto, pillFlash, pillPro, pillLocal].forEach((b) => {
      b.disabled = isModelSwitching;
    });

    const shortResolved = (eff || "qwen3.5").replace(/^(?:unsloth|ollama)::/, "");
    if (isAuto) {
      providerBadge.className = "mt-provider-badge auto";
      providerBadge.textContent = "✨ АВТО";
      activeModelTitle.textContent = `Авто (${shortResolved})`;
      activeModelSub.textContent = eff.startsWith("unsloth::")
        ? "Активен облачный API + локальный резерв Ollama"
        : "Активна локальная модель Ollama (на GPU)";
    } else if (sel.startsWith("unsloth::")) {
      providerBadge.className = "mt-provider-badge cloud";
      providerBadge.textContent = "☁️ CLOUD";
      activeModelTitle.textContent = formatModelDisplayName(sel);
      activeModelSub.textContent = `ID: ${sel.replace(/^unsloth::/, "")}`;
    } else if (sel.startsWith("ollama::")) {
      providerBadge.className = "mt-provider-badge ollama";
      providerBadge.textContent = "🦙 OLLAMA";
      activeModelTitle.textContent = formatModelDisplayName(sel);
      activeModelSub.textContent = `Локально на GPU (${sel.replace(/^ollama::/, "")})`;
    } else {
      providerBadge.className = "mt-provider-badge auto";
      providerBadge.textContent = "⚙️ СВОЯ";
      activeModelTitle.textContent = sel;
      activeModelSub.textContent = "Пользовательская модель";
    }
  }

  function syncUiFromSettings() {
    sourceLangSelect.value = settings.lang;
    targetLangSelect.value = settings.targetLang;
    unslothUrlInput.value = settings.unslothUrl;
    if (settings.unslothApiKey) {
      unslothKeyInput.value = settings.unslothApiKey;
      unslothKeyInput.placeholder = "Ключ сохранен";
    } else if (serverHasApiKey) {
      unslothKeyInput.value = "";
      unslothKeyInput.placeholder = "•••••••• (ключ сохранен на сервере)";
    } else {
      unslothKeyInput.value = "";
      unslothKeyInput.placeholder = "Пусто, если без пароля";
    }
    manualModelInput.value = settings.model !== "auto" ? settings.model : "";
    minFontSlider.value = String(settings.minFontSize);
    minFontVal.textContent = String(settings.minFontSize);
    fontSlider.value = String(settings.maxFontSize);
    fontVal.textContent = String(settings.maxFontSize);
    overflowModeSel.value = settings.overflowMode || "shrink";
    autoSaveChk.checked = settings.autoSaveWeb !== false;
    webSaveDirInput.value = settings.webSaveDir || "";
    filterModeSel.value = settings.textFilterMode || "dialogue_only";
    skipSighsChk.checked = settings.skipSighs !== false;
    skipSfxChk.checked = settings.skipSfx !== false;
    freeTextChk.checked = Boolean(settings.detectFreeText);
    uppercaseChk.checked = settings.uppercase;
    panel.classList.toggle("mt-minimized", Boolean(settings.minimized));
    minimizeBtn.textContent = settings.minimized ? "▢" : "—";
    applySavedPanelPosition();
    updateProviderPillsUI();
  }

  migrateAndLoadSettings(syncUiFromSettings);

  document.getElementById("mt-close-panel-btn").addEventListener("click", () => {
    panel.classList.remove("mt-open");
  });

  document.getElementById("mt-web-ui-btn").addEventListener("click", () => {
    window.open("http://127.0.0.1:5000/", "_blank");
  });

  sourceLangSelect.addEventListener("change", () => {
    settings.lang = sourceLangSelect.value;
    saveLocalSettings();
  });

  targetLangSelect.addEventListener("change", () => {
    settings.targetLang = targetLangSelect.value;
    saveLocalSettings();
  });

  function setModelPickerOpen(openState) {
    customModelPicker.style.display = openState ? "flex" : "none";
    activeModelBar.classList.toggle("mt-open", openState);
    modelChevron.textContent = openState ? "📋 ▴" : "📋 ▾";
    if (openState) {
      renderCustomModelList(availableModelsCache, resolvedServerModel);
    }
  }

  activeModelBar.addEventListener("click", () => {
    const isHidden =
      customModelPicker.style.display === "none" || !customModelPicker.style.display;
    setModelPickerOpen(isHidden);
    if (isHidden) {
      checkServerAndLoadModels(true, true);
    }
  });

  modelSearchInput.addEventListener("input", () => {
    renderCustomModelList(availableModelsCache, resolvedServerModel);
  });

  async function startPullingLocalModel(modelId, displayName) {
    const cleanTag = String(modelId || "").replace(/^ollama::/, "").trim();
    if (!cleanTag) return;

    pullProgressWrap.style.display = "flex";
    pullProgressFill.style.width = "2%";
    pullProgressText.textContent = `📥 Запуск скачивания ${cleanTag} в Ollama...`;
    statusText.textContent = `📥 Скачивание модели ${cleanTag}...`;

    const res = await safeSendMessage({
      action: "PULL_MODEL",
      model: cleanTag,
      setAsDefault: true,
    });

    if (!res || !res.ok) {
      pullProgressText.textContent = `❌ Ошибка: ${(res && res.error) || "Не удалось начать скачивание"}`;
      return;
    }

    if (pullPollTimer) clearInterval(pullPollTimer);
    pullPollTimer = setInterval(async () => {
      const stRes = await safeSendMessage({ action: "GET_PULL_STATUS" });
      if (!stRes || !stRes.ok || !stRes.data) return;
      const st = stRes.data;
      const pct = Number(st.progress_pct) || 0;
      pullProgressFill.style.width = `${Math.min(100, Math.max(2, pct))}%`;
      if (st.status === "downloading") {
        const gbInfo =
          st.total_gb > 0 ? ` (${st.completed_gb}/${st.total_gb} ГБ, ${pct.toFixed(0)}%)` : "";
        pullProgressText.textContent = `📥 ${st.message || `Скачивание ${cleanTag}`}${gbInfo}`;
      } else if (st.status === "completed") {
        clearInterval(pullPollTimer);
        pullPollTimer = null;
        pullProgressFill.style.width = "100%";
        pullProgressText.textContent = `✅ Модель ${cleanTag} скачана и готова!`;
        const fullId = `ollama::${cleanTag}`;
        settings.lastLocalModel = fullId;
        await applySelectedModel(fullId);
        await checkServerAndLoadModels(true, true);
        setTimeout(() => {
          pullProgressWrap.style.display = "none";
        }, 3500);
      } else if (st.status === "error") {
        clearInterval(pullPollTimer);
        pullPollTimer = null;
        pullProgressText.textContent = `❌ Ошибка скачивания: ${st.error || st.message}`;
      }
    }, 1200);
  }

  async function applySelectedModel(newModelId) {
    const cleanTarget = String(newModelId || "auto").trim() || "auto";
    const reqId = ++modelSwitchSeq;

    settings.model = cleanTarget;
    if (cleanTarget.startsWith("ollama::")) {
      settings.lastLocalModel = cleanTarget;
    }
    manualModelInput.value = settings.model !== "auto" ? settings.model : "";
    isModelSwitching = true;
    saveLocalSettings();
    updateProviderPillsUI();
    renderCustomModelList(availableModelsCache, resolvedServerModel);

    statusText.textContent = "⏳ Переключение ИИ-модели...";
    const res = await safeSendMessage({
      action: "SET_ACTIVE_MODEL",
      selectedModel: settings.model,
    });

    if (reqId !== modelSwitchSeq) {
      return;
    }
    isModelSwitching = false;

    if (res && res.ok && res.data) {
      availableModelsCache = res.data.models || availableModelsCache;
      resolvedServerModel = res.data.active_model || settings.model;
      serverHasApiKey = Boolean(res.data.has_unsloth_key);
      updateProviderPillsUI();
      renderCustomModelList(availableModelsCache, resolvedServerModel);
      const shortLabel = resolvedServerModel.replace(/^.*?::/, "");
      const prefix = resolvedServerModel.startsWith("unsloth::") ? "☁️ Cloud" : "🦙 Ollama";
      statusText.textContent = `✅ Активна: ${prefix} (${shortLabel})`;
    } else {
      updateProviderPillsUI();
      statusText.textContent = `⚠️ Выбрана локально: ${settings.model} (сервер офлайн)`;
    }
  }

  pillAuto.addEventListener("click", () => {
    applySelectedModel("auto");
  });

  pillFlash.addEventListener("click", () => {
    applySelectedModel("unsloth::deepseek-v4-flash");
  });

  pillPro.addEventListener("click", () => {
    applySelectedModel("unsloth::deepseek-v4-pro");
  });

  pillLocal.addEventListener("click", () => {
    const bestLocal = getBestInstalledLocalModel();
    if (bestLocal) {
      applySelectedModel(bestLocal);
    } else {
      setModelPickerOpen(true);
      checkServerAndLoadModels(true, true);
      statusText.textContent = "🦙 Выберите или скачайте локальную модель Ollama из списка ↓";
    }
  });

  minFontSlider.addEventListener("input", () => {
    settings.minFontSize = parseInt(minFontSlider.value, 10);
    if (settings.minFontSize > settings.maxFontSize) {
      settings.maxFontSize = settings.minFontSize;
      fontSlider.value = String(settings.maxFontSize);
      fontVal.textContent = String(settings.maxFontSize);
    }
    minFontVal.textContent = String(settings.minFontSize);
    saveLocalSettings();
  });

  fontSlider.addEventListener("input", () => {
    settings.maxFontSize = parseInt(fontSlider.value, 10);
    if (settings.maxFontSize < settings.minFontSize) {
      settings.minFontSize = settings.maxFontSize;
      minFontSlider.value = String(settings.minFontSize);
      minFontVal.textContent = String(settings.minFontSize);
    }
    fontVal.textContent = String(settings.maxFontSize);
    saveLocalSettings();
  });

  overflowModeSel.addEventListener("change", () => {
    settings.overflowMode = overflowModeSel.value || "shrink";
    saveLocalSettings();
    safeSendMessage({
      action: "SAVE_SERVER_SETTINGS",
      overflowMode: settings.overflowMode,
    });
  });

  autoSaveChk.addEventListener("change", () => {
    settings.autoSaveWeb = autoSaveChk.checked;
    saveLocalSettings();
    safeSendMessage({
      action: "SAVE_SERVER_SETTINGS",
      autoSaveWeb: settings.autoSaveWeb,
      webSaveDir: settings.webSaveDir,
    });
  });

  webSaveDirInput.addEventListener("change", () => {
    settings.webSaveDir = webSaveDirInput.value.trim();
    saveLocalSettings();
  });

  pickSaveDirBtn.addEventListener("click", async () => {
    statusText.textContent = "Выберите папку сохранения в окне Windows...";
    const res = await safeSendMessage({ action: "PICK_WEB_SAVE_FOLDER" });
    if (res && res.ok && res.data && res.data.status === "ok") {
      settings.webSaveDir = res.data.web_save_dir || "";
      webSaveDirInput.value = settings.webSaveDir;
      saveLocalSettings();
      statusText.textContent = `💾 Папка сохранений: ${settings.webSaveDir}`;
    } else {
      statusText.textContent = "Выбор папки отменен.";
    }
  });

  openSavedBtn.addEventListener("click", async () => {
    const res = await safeSendMessage({
      action: "OPEN_WEB_SAVE_FOLDER",
      path: settings.webSaveDir || null,
    });
    if (res && res.ok) {
      statusText.textContent = "📂 Папка сохраненной манги открыта в Проводнике!";
    } else {
      statusText.textContent = `❌ ${(res && res.error) || "Не удалось открыть папку"}`;
    }
  });

  filterModeSel.addEventListener("change", () => {
    const m = filterModeSel.value;
    settings.textFilterMode = m;
    if (m === "dialogue_only") {
      settings.skipSighs = true;
      settings.skipSfx = true;
      settings.detectFreeText = false;
      settings.minTextLength = 2;
    } else if (m === "bubbles_no_sighs") {
      settings.skipSighs = true;
      settings.skipSfx = true;
      settings.detectFreeText = true;
      settings.minTextLength = 2;
    } else if (m === "long_only") {
      settings.skipSighs = true;
      settings.skipSfx = true;
      settings.detectFreeText = false;
      settings.minTextLength = 4;
    } else if (m === "bubbles_only") {
      settings.skipSighs = false;
      settings.skipSfx = true;
      settings.detectFreeText = false;
      settings.minTextLength = 1;
    } else if (m === "all") {
      settings.skipSighs = false;
      settings.skipSfx = false;
      settings.detectFreeText = true;
      settings.minTextLength = 1;
    }
    syncUiFromSettings();
    saveLocalSettings();
    safeSendMessage({
      action: "SAVE_SERVER_SETTINGS",
      textFilterMode: settings.textFilterMode,
      skipSighs: settings.skipSighs,
      skipSfx: settings.skipSfx,
      detectFreeText: settings.detectFreeText,
      minTextLength: settings.minTextLength,
    });
  });

  skipSighsChk.addEventListener("change", () => {
    settings.skipSighs = skipSighsChk.checked;
    saveLocalSettings();
    safeSendMessage({ action: "SAVE_SERVER_SETTINGS", skipSighs: settings.skipSighs });
  });

  skipSfxChk.addEventListener("change", () => {
    settings.skipSfx = skipSfxChk.checked;
    saveLocalSettings();
    safeSendMessage({ action: "SAVE_SERVER_SETTINGS", skipSfx: settings.skipSfx });
  });

  freeTextChk.addEventListener("change", () => {
    settings.detectFreeText = freeTextChk.checked;
    saveLocalSettings();
    safeSendMessage({ action: "SAVE_SERVER_SETTINGS", detectFreeText: settings.detectFreeText });
  });

  uppercaseChk.addEventListener("change", () => {
    settings.uppercase = uppercaseChk.checked;
    saveLocalSettings();
    safeSendMessage({ action: "SAVE_SERVER_SETTINGS", uppercase: settings.uppercase });
  });

  document.getElementById("mt-settings-toggle").addEventListener("click", () => {
    const isHidden = settingsDrawer.style.display === "none" || !settingsDrawer.style.display;
    settingsDrawer.style.display = isHidden ? "flex" : "none";
  });

  document.getElementById("mt-refresh-btn").addEventListener("click", () => {
    checkServerAndLoadModels(false, true);
  });

  document.getElementById("mt-save-settings-btn").addEventListener("click", async () => {
    settings.unslothUrl = unslothUrlInput.value.trim() || "https://moduleapi.pro/v1";
    const enteredKey = unslothKeyInput.value.trim();
    if (enteredKey) {
      settings.unslothApiKey = enteredKey;
    }
    settings.autoSaveWeb = autoSaveChk.checked;
    settings.webSaveDir = webSaveDirInput.value.trim();
    settings.overflowMode = overflowModeSel.value || "shrink";
    const manualMod = manualModelInput.value.trim();
    if (manualMod) {
      settings.model = manualMod;
    } else {
      settings.model = "auto";
    }
    saveLocalSettings();
    updateProviderPillsUI();
    statusText.textContent = "💾 Сохранение настроек на сервере...";

    const savePayload = {
      action: "SAVE_SERVER_SETTINGS",
      unslothUrl: settings.unslothUrl,
      selectedModel: settings.model,
      autoSaveWeb: settings.autoSaveWeb,
      webSaveDir: settings.webSaveDir,
      textFilterMode: settings.textFilterMode,
      skipSighs: settings.skipSighs,
      skipSfx: settings.skipSfx,
      detectFreeText: settings.detectFreeText,
      minTextLength: settings.minTextLength,
      minFontSize: settings.minFontSize,
      maxFontSize: settings.maxFontSize,
      uppercase: settings.uppercase,
      overflowMode: settings.overflowMode,
    };
    if (enteredKey) {
      savePayload.unslothApiKey = enteredKey;
    }

    const res = await safeSendMessage(savePayload);
    if (res && res.ok && res.data) {
      availableModelsCache = res.data.models || [];
      resolvedServerModel = res.data.active_model || settings.model;
      serverHasApiKey = Boolean(res.data.has_unsloth_key);
      syncUiFromSettings();
      renderCustomModelList(availableModelsCache, resolvedServerModel);
      const shortActive = resolvedServerModel.replace(/^.*?::/, "");
      statusText.textContent = `✅ Сохранено! Активна: ${shortActive}`;
    } else {
      statusText.textContent = `❌ ${(res && res.error) || "Сервер недоступен"}`;
    }
  });

  startServerBtn.addEventListener("click", async () => {
    statusText.textContent = "Запуск сервера (~6–10 сек)...";
    triggerCustomProtocol("manga-translator://start");

    for (let attempt = 1; attempt <= 12; attempt++) {
      await new Promise((r) => setTimeout(r, 2000));
      const ok = await checkServerAndLoadModels(true, false);
      if (ok) return;
      statusText.textContent = `Ожидание готовности сервера (${attempt}/12)...`;
    }
    statusText.textContent = "Не удалось дождаться ответа сервера. Нажмите 🔄";
  });

  document.getElementById("mt-stop-server-btn").addEventListener("click", async () => {
    await safeSendMessage({ action: "STOP_SERVER" });
    setTimeout(() => checkServerAndLoadModels(false, false), 800);
  });

  function renderCustomModelList(models, activeModel) {
    modelListItems.innerHTML = "";
    const query = (modelSearchInput.value || "").trim().toLowerCase();
    const list = Array.isArray(models) ? models : [];
    const shortActive = (activeModel || "qwen3.5").replace(/^.*?::/, "");

    const addGroupHeader = (title) => {
      const hdr = document.createElement("div");
      hdr.className = "mt-model-group-title";
      hdr.textContent = title;
      modelListItems.appendChild(hdr);
    };

    const createItemButton = (item, badgeText, isDownloadable = false) => {
      const btn = document.createElement("button");
      btn.type = "button";
      const isSel =
        !isDownloadable &&
        (settings.model === item.id || (settings.model === "auto" && item.id === "auto"));
      btn.className =
        "mt-model-item" +
        (isSel ? " selected" : "") +
        (isDownloadable ? " downloadable" : "");

      const nameSpan = document.createElement("span");
      nameSpan.className = "mt-model-item-name";
      nameSpan.textContent = item.name || item.id;
      btn.appendChild(nameSpan);

      const rightSpan = document.createElement("span");
      rightSpan.className = "mt-model-item-tag";
      rightSpan.textContent = isSel ? "✅ Активна" : badgeText;
      btn.appendChild(rightSpan);

      btn.addEventListener("click", () => {
        if (isDownloadable) {
          startPullingLocalModel(item.id, item.name);
        } else {
          applySelectedModel(item.id);
          setModelPickerOpen(false);
        }
      });
      modelListItems.appendChild(btn);
    };

    if (!query || "авто auto".includes(query) || shortActive.toLowerCase().includes(query)) {
      addGroupHeader("✨ Автоматический выбор");
      createItemButton(
        { id: "auto", name: `✨ Авто (${shortActive})` },
        "Рекомендуется",
        false
      );
    }

    const cloudModels = list.filter(
      (m) =>
        m &&
        (m.provider === "unsloth" || String(m.id || "").startsWith("unsloth::")) &&
        (!query ||
          String(m.name || "").toLowerCase().includes(query) ||
          String(m.id || "").toLowerCase().includes(query))
    );
    if (cloudModels.length > 0) {
      addGroupHeader("☁️ Облачные ИИ-модели (DeepSeek / OpenAI API)");
      cloudModels.forEach((m) => {
        const cleanName = String(m.name || m.id).replace(/^\[Cloud API\]\s*/, "☁️ ");
        createItemButton({ ...m, name: cleanName }, "Cloud API", false);
      });
    }

    const installedOllama = list.filter(
      (m) =>
        m &&
        (m.provider === "ollama" || String(m.id || "").startsWith("ollama::")) &&
        isModelInstalled(m) &&
        (!query ||
          String(m.name || "").toLowerCase().includes(query) ||
          String(m.id || "").toLowerCase().includes(query))
    );
    if (installedOllama.length > 0) {
      addGroupHeader("🦙 Установленные локальные модели (Ollama GPU)");
      installedOllama.forEach((m) => {
        const cleanName = String(m.name || m.id).replace(/^\[Ollama\]\s*/, "🦙 ");
        createItemButton({ ...m, name: cleanName }, m.size_gb ? `${m.size_gb} ГБ` : "Локально", false);
      });
    }

    const downloadableOllama = list.filter(
      (m) =>
        m &&
        isModelDownloadable(m) &&
        (!query ||
          String(m.name || "").toLowerCase().includes(query) ||
          String(m.id || "").toLowerCase().includes(query))
    );
    if (downloadableOllama.length > 0) {
      addGroupHeader("📥 Скачать в Ollama в 1 клик (Без цензуры)");
      downloadableOllama.forEach((m) => {
        const cleanName = String(m.name || m.id).replace(/^\[📥 Скачать в Ollama\]\s*/, "📥 ");
        createItemButton(
          { ...m, name: cleanName },
          m.size_gb ? `Скачать ~${m.size_gb} ГБ` : "Скачать",
          true
        );
      });
    }

    if (modelListItems.children.length === 0) {
      const emptyMsg = document.createElement("div");
      emptyMsg.style.cssText = "color:#9ca3af; font-size:11.5px; text-align:center; padding:8px;";
      emptyMsg.textContent = "Модели по запросу не найдены";
      modelListItems.appendChild(emptyMsg);
    }
  }

  async function checkServerAndLoadModels(silent = false, forceRefresh = false) {
    if (isCheckingServer) {
      if (forceRefresh) pendingForceRefresh = true;
      return false;
    }
    isCheckingServer = true;

    try {
      if (!silent) {
        statusText.textContent = "Проверка связи с сервером...";
      }

      const healthRes = await safeSendMessage({ action: "CHECK_HEALTH" });
      if (!healthRes || !healthRes.ok) {
        serverDot.classList.remove("online");
        startServerBtn.style.display = "block";
        if (!silent) {
          statusText.textContent = "Сервер выключен. Нажмите кнопку запуска ↑";
        }
        return false;
      }

      serverDot.classList.add("online");
      startServerBtn.style.display = "none";

      const hd = healthRes.data || {};
      serverHasApiKey = Boolean(hd.has_unsloth_key);
      if (hd.unsloth_url) {
        settings.unslothUrl = hd.unsloth_url;
      }
      if (hd.unsloth_api_key && !settings.unslothApiKey) {
        settings.unslothApiKey = hd.unsloth_api_key;
      }
      if (hd.selected_model) {
        settings.model = hd.selected_model;
        if (hd.selected_model.startsWith("ollama::")) {
          settings.lastLocalModel = hd.selected_model;
        }
      }
      if (hd.web_save_dir && !settings.webSaveDir) {
        settings.webSaveDir = hd.web_save_dir;
      }
      if (hd.overflow_mode) {
        settings.overflowMode = hd.overflow_mode;
      }
      saveLocalSettings();

      let activeMod = hd.active_model || hd.ollama_model || "qwen3.5";
      const easyOcrOk = hd.easyocr_installed;
      let modelsList = Array.isArray(hd.models) ? hd.models : null;

      if (forceRefresh || pendingForceRefresh || !modelsList) {
        pendingForceRefresh = false;
        const modelsRes = await safeSendMessage({
          action: "GET_MODELS",
          refresh: true,
        });
        if (modelsRes && modelsRes.ok && modelsRes.data) {
          modelsList = modelsRes.data.models || [];
          activeMod = modelsRes.data.active_model || activeMod;
          if (modelsRes.data.selected_model) {
            settings.model = modelsRes.data.selected_model;
          }
        }
      }

      availableModelsCache = modelsList || [];
      resolvedServerModel = activeMod;
      syncUiFromSettings();
      renderCustomModelList(availableModelsCache, activeMod);

      const shortMod = activeMod.replace(/^.*?::/, "");
      const providerIcon = activeMod.startsWith("unsloth::") ? "☁️ Cloud:" : "🦙 Ollama:";
      statusText.textContent = easyOcrOk
        ? `${providerIcon} ${shortMod} (JA/KO/ZH)`
        : `${providerIcon} ${shortMod} (JA)`;
      return true;
    } finally {
      isCheckingServer = false;
      if (pendingForceRefresh) {
        pendingForceRefresh = false;
        checkServerAndLoadModels(true, true);
      }
    }
  }

  function isMangaCandidate(img) {
    if (!img || !img.isConnected) return false;
    if (img.closest(".gallery, .thumb-container, #thumbnail-container, #mt-floating-panel")) {
      return false;
    }
    const rect = img.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      return rect.width >= MIN_WIDTH && rect.height >= MIN_HEIGHT;
    }
    const w = img.naturalWidth || 0;
    const h = img.naturalHeight || 0;
    return w >= MIN_WIDTH && h >= MIN_HEIGHT;
  }

  function isImageInViewport(img) {
    if (!isMangaCandidate(img)) return false;
    const rect = img.getBoundingClientRect();
    const vh = window.innerHeight || document.documentElement.clientHeight;
    const visibleTop = Math.max(0, rect.top);
    const visibleBottom = Math.min(vh, rect.bottom);
    return visibleBottom - visibleTop >= Math.min(180, rect.height * 0.25);
  }

  function resolveImageUrl(img) {
    return (
      img.dataset.mtOriginalSrc ||
      img.currentSrc ||
      img.src ||
      img.dataset.src ||
      img.dataset.lazySrc ||
      ""
    );
  }

  function clearPictureSources(img) {
    const pic = img.parentElement;
    if (pic && pic.tagName === "PICTURE") {
      pic.querySelectorAll("source").forEach((sourceEl) => {
        if (sourceEl.srcset && !sourceEl.dataset.mtOriginalSrcset) {
          sourceEl.dataset.mtOriginalSrcset = sourceEl.srcset;
        }
        sourceEl.removeAttribute("srcset");
      });
    }
  }

  function restorePictureSources(img) {
    const pic = img.parentElement;
    if (pic && pic.tagName === "PICTURE") {
      pic.querySelectorAll("source").forEach((sourceEl) => {
        if (sourceEl.dataset.mtOriginalSrcset) {
          sourceEl.setAttribute("srcset", sourceEl.dataset.mtOriginalSrcset);
        }
      });
    }
  }

  function tryExtractCanvasBase64(img) {
    try {
      if (!img.complete || !img.naturalWidth || !img.naturalHeight) return null;
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      return canvas.toDataURL("image/webp", 0.92);
    } catch (_) {
      return null;
    }
  }

  async function tryFetchInPageContext(url) {
    for (const cred of ["omit", "include"]) {
      try {
        const res = await fetch(url, { credentials: cred });
        if (!res.ok) continue;
        const blob = await res.blob();
        if (!blob || blob.size < 100) continue;
        const dataUrl = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(typeof reader.result === "string" ? reader.result : null);
          reader.onerror = () => resolve(null);
          reader.readAsDataURL(blob);
        });
        if (dataUrl) return dataUrl;
      } catch (_) {}
    }
    return null;
  }

  function resolveCandidatePageIndex(targetImg) {
    if (targetImg.dataset.mtPageIndex) {
      return parseInt(targetImg.dataset.mtPageIndex, 10) || 1;
    }
    const allCandidates = Array.from(document.querySelectorAll("img")).filter((el) => isMangaCandidate(el));
    const idx = allCandidates.indexOf(targetImg);
    const pageNum = idx >= 0 ? idx + 1 : 1;
    targetImg.dataset.mtPageIndex = String(pageNum);
    return pageNum;
  }

  async function translateSingleImage(img, badge, explicitPageIndex = null) {
    if (img.dataset.mtBusy === "true") return { ok: false, skipped: true };
    const srcUrl = resolveImageUrl(img);
    if (!srcUrl) return { ok: false, error: "Пустой URL картинки" };

    const pageIndex = explicitPageIndex || resolveCandidatePageIndex(img);

    img.dataset.mtBusy = "true";
    if (badge) {
      badge.classList.remove("mt-error");
      badge.classList.add("mt-loading");
      badge.textContent = "⏳ Перевод...";
      updateBadgePosition(img, badge);
    }

    try {
      const isRetranslate = img.dataset.mtTranslated === "true";
      let imageB64 = isRetranslate ? (img.dataset.mtOriginalBase64 || null) : tryExtractCanvasBase64(img);
      if (imageB64 && !img.dataset.mtOriginalBase64 && !isRetranslate) {
        img.dataset.mtOriginalBase64 = imageB64;
      }
      if (!imageB64 && srcUrl.startsWith("blob:")) {
        imageB64 = await tryFetchInPageContext(srcUrl);
        if (imageB64 && !img.dataset.mtOriginalBase64) {
          img.dataset.mtOriginalBase64 = imageB64;
        }
      }

      const response = await safeSendMessage({
        action: "TRANSLATE_IMAGE",
        imageUrl: srcUrl,
        imageBase64: imageB64,
        lang: settings.lang,
        targetLang: settings.targetLang,
        model: settings.model,
        uppercase: settings.uppercase,
        minFontSize: settings.minFontSize,
        maxFontSize: settings.maxFontSize,
        overflowMode: settings.overflowMode,
        detectFreeText: settings.detectFreeText,
        textFilterMode: settings.textFilterMode,
        skipSighs: settings.skipSighs,
        skipSfx: settings.skipSfx,
        minTextLength: settings.minTextLength,
        forceRetranslate: isRetranslate,
        autoSave: settings.autoSaveWeb !== false,
        saveDir: settings.webSaveDir || null,
        chapterTitle: document.title || location.hostname || "Web_Manga",
        pageUrl: location.href,
        pageIndex: pageIndex,
      });

      if (!response || !response.ok) {
        throw new Error((response && response.error) || "Ошибка сервера");
      }

      const result = response.data;
      if (result.blocks_count === 0 || !result.translated_image) {
        if (badge) badge.textContent = "Нет диалогов";
        return { ok: true, empty: true };
      }

      if (!img.dataset.mtOriginalSrc) {
        img.dataset.mtOriginalSrc = srcUrl;
      }
      if (img.srcset) {
        img.dataset.mtOriginalSrcset = img.srcset;
        img.removeAttribute("srcset");
      }
      clearPictureSources(img);

      const finalImgSrc = result.translated_image.startsWith("data:")
        ? result.translated_image
        : `data:image/webp;base64,${result.translated_image}`;

      img.src = finalImgSrc;
      img.dataset.mtTranslatedSrc = finalImgSrc;
      img.dataset.mtTranslated = "true";
      showingOriginal = false;
      toggleOrigBtn.textContent = "👁 Показать оригинал";

      const langBadge = (result.detected_lang || settings.lang).toUpperCase();
      const savedMark = result.saved_output_path ? " 💾" : "";
      if (badge) {
        badge.textContent = `✅ ${langBadge}→${settings.targetLang.toUpperCase()} (${result.blocks_count})${savedMark} 🔁`;
        badge.title = result.saved_output_path
          ? `Сохранено на ПК: ${result.saved_output_path}\nНажмите, чтобы перевести страницу заново`
          : "Нажмите, чтобы перевести эту страницу заново с текущими настройками";
      }
      toggleOrigBtn.style.display = "block";
      return { ok: true, blocks: result.blocks_count };
    } catch (err) {
      console.error("[MangaTranslator] Ошибка:", err);
      if (badge) {
        badge.textContent = "❌ Ошибка (повторить)";
        badge.classList.add("mt-error");
      }
      statusText.textContent = `❌ ${err.message}`;
      return { ok: false, error: err.message };
    } finally {
      img.dataset.mtBusy = "false";
      if (badge) {
        badge.classList.remove("mt-loading");
        updateBadgePosition(img, badge);
      }
    }
  }

  let badgesLayer = document.getElementById("mt-badges-layer");
  if (!badgesLayer) {
    badgesLayer = document.createElement("div");
    badgesLayer.id = "mt-badges-layer";
    document.body.appendChild(badgesLayer);
  }

  const trackedImages = new Set();

  function updateBadgePosition(img, badge) {
    if (!img || !img.isConnected || !badge) return;
    const rect = img.getBoundingClientRect();
    const isVisibleInViewport =
      rect.width >= MIN_WIDTH &&
      rect.height >= MIN_HEIGHT &&
      rect.bottom > 40 &&
      rect.top < window.innerHeight - 40 &&
      rect.right > 40 &&
      rect.left < window.innerWidth - 40;

    const isActiveState =
      badge.classList.contains("mt-visible") ||
      badge.classList.contains("mt-loading") ||
      badge.classList.contains("mt-error") ||
      badge.matches(":hover");

    if (!isVisibleInViewport || !isActiveState) {
      badge.style.top = "-9999px";
      return;
    }

    const clampedTop = Math.max(10, Math.min(rect.top + 10, rect.bottom - 38));
    const rightOffset = Math.max(10, document.documentElement.clientWidth - rect.right + 10);
    badge.style.top = `${Math.round(clampedTop)}px`;
    badge.style.right = `${Math.round(rightOffset)}px`;
  }

  function refreshActiveBadgePositions() {
    trackedImages.forEach((img) => {
      if (!img.isConnected) {
        if (img.__mtBadge) img.__mtBadge.remove();
        trackedImages.delete(img);
        return;
      }
      if (img.__mtBadge) {
        updateBadgePosition(img, img.__mtBadge);
      }
    });
  }

  window.addEventListener("scroll", refreshActiveBadgePositions, { passive: true });
  window.addEventListener("resize", () => {
    refreshActiveBadgePositions();
    applySavedPanelPosition();
  }, { passive: true });

  function attachBadgeToImage(img) {
    if (img.dataset.mtAttached === "true") return;
    if (!isMangaCandidate(img)) return;

    img.dataset.mtAttached = "true";

    const badge = document.createElement("button");
    badge.type = "button";
    badge.className = "mt-translate-badge";
    badge.textContent = "🉐 Перевести";

    const showBadge = () => {
      if (!isMangaCandidate(img)) return;
      badge.classList.add("mt-visible");
      updateBadgePosition(img, badge);
    };

    const hideBadge = () => {
      badge.classList.remove("mt-visible");
      setTimeout(() => {
        if (!badge.matches(":hover") && !badge.classList.contains("mt-visible")) {
          updateBadgePosition(img, badge);
        }
      }, 60);
    };

    img.addEventListener("mouseenter", showBadge);
    img.addEventListener("mousemove", showBadge);
    img.addEventListener("mouseleave", hideBadge);
    badge.addEventListener("mouseenter", () => {
      badge.classList.add("mt-visible");
      updateBadgePosition(img, badge);
    });
    badge.addEventListener("mouseleave", hideBadge);

    badge.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      translateSingleImage(img, badge);
    });

    badgesLayer.appendChild(badge);
    img.__mtBadge = badge;
    trackedImages.add(img);
  }

  function scanPageImages() {
    document.querySelectorAll("img").forEach((img) => {
      if (img.complete) {
        attachBadgeToImage(img);
      } else {
        img.addEventListener("load", () => attachBadgeToImage(img), { once: true });
      }
    });
  }

  async function runImagesBatchTranslation(candidates, modeLabel = "страницы") {
    if (pageBatchRunning) return;
    if (!candidates || candidates.length === 0) {
      statusText.textContent = "Страницы для перевода не найдены.";
      return;
    }

    pageBatchRunning = true;
    pageStopRequested = false;
    translateAllBtn.disabled = true;
    translateViewportBtn.disabled = true;
    stopPageBtn.style.display = "inline-block";
    pageProgressWrap.style.display = "flex";
    pageProgressFill.style.width = "0%";

    let okCount = 0;
    let failCount = 0;
    let lastError = "";

    for (let i = 0; i < candidates.length; i++) {
      if (pageStopRequested) {
        break;
      }
      const pct = Math.round((i / candidates.length) * 100);
      pageProgressFill.style.width = `${pct}%`;
      pageProgressText.textContent = `⏳ Перевод ${i + 1} из ${candidates.length} (готово: ${okCount})...`;
      statusText.textContent = `⏳ Перевод ${modeLabel}: ${i + 1} из ${candidates.length}...`;

      const res = await translateSingleImage(candidates[i], candidates[i].__mtBadge);
      if (res && res.ok) {
        okCount++;
      } else if (!res?.skipped) {
        failCount++;
        lastError = res?.error || "Ошибка";
      }
      pageProgressFill.style.width = `${Math.round(((i + 1) / candidates.length) * 100)}%`;
    }

    pageBatchRunning = false;
    translateAllBtn.disabled = false;
    translateViewportBtn.disabled = false;
    stopPageBtn.style.display = "none";
    setTimeout(() => {
      if (!pageBatchRunning) pageProgressWrap.style.display = "none";
    }, 3000);

    if (pageStopRequested) {
      statusText.textContent = `⏸ Остановлено: переведено ${okCount} из ${candidates.length} стр.`;
    } else if (failCount === 0) {
      statusText.textContent = `✅ Готово! Переведено: ${okCount} из ${candidates.length} стр.`;
    } else if (okCount > 0) {
      statusText.textContent = `⚠️ Переведено: ${okCount}, ошибок: ${failCount} (${lastError})`;
    } else {
      statusText.textContent = `❌ Ошибка (${failCount} стр.): ${lastError}`;
    }
  }

  async function translateAllVisibleImages() {
    scanPageImages();
    const allMangaImgs = Array.from(document.querySelectorAll("img")).filter((img) =>
      isMangaCandidate(img)
    );
    let candidates = allMangaImgs.filter((img) => img.dataset.mtTranslated !== "true");
    if (candidates.length === 0 && allMangaImgs.length > 0) {
      candidates = allMangaImgs;
    }
    await runImagesBatchTranslation(candidates, "главы");
  }

  async function translateViewportImagesOnly() {
    scanPageImages();
    const viewportImgs = Array.from(document.querySelectorAll("img")).filter((img) =>
      isImageInViewport(img)
    );
    if (viewportImgs.length === 0) {
      statusText.textContent = "На текущем экране не найдено кадров манги.";
      return;
    }
    await runImagesBatchTranslation(viewportImgs, "текущего экрана");
  }

  stopPageBtn.addEventListener("click", () => {
    pageStopRequested = true;
    statusText.textContent = "⏸ Остановка после текущей страницы...";
  });

  function toggleOriginalAll() {
    const translatedImgs = Array.from(
      document.querySelectorAll('img[data-mt-translated="true"]')
    );
    if (translatedImgs.length === 0) return;

    showingOriginal = !showingOriginal;
    translatedImgs.forEach((img) => {
      if (showingOriginal) {
        img.src = img.dataset.mtOriginalSrc;
        if (img.dataset.mtOriginalSrcset) {
          img.setAttribute("srcset", img.dataset.mtOriginalSrcset);
        }
        restorePictureSources(img);
      } else {
        if (img.srcset) img.removeAttribute("srcset");
        clearPictureSources(img);
        img.src = img.dataset.mtTranslatedSrc;
      }
    });
    toggleOrigBtn.textContent = showingOriginal ? "👁 Вернуть перевод" : "👁 Показать оригинал";
  }

  function pollFolderStatus() {
    if (folderPollTimer) clearInterval(folderPollTimer);
    folderProgressWrap.style.display = "flex";

    folderPollTimer = setInterval(async () => {
      const res = await safeSendMessage({ action: "GET_FOLDER_STATUS" });
      if (!res || !res.ok || !res.data) return;

      const st = res.data;
      if (st.status === "running") {
        const pct = st.total > 0 ? Math.round((st.current / st.total) * 100) : 0;
        progressBarFill.style.width = `${pct}%`;
        const detLang = (st.detected_lang || st.lang || "auto").toUpperCase();
        folderProgressText.textContent = `[${st.current}/${st.total} | ${detLang}] ${st.current_file} (${st.elapsed_sec}с)`;
        folderBtn.disabled = true;
        folderStopBtn.style.display = "inline-block";
      } else if (st.status === "completed") {
        clearInterval(folderPollTimer);
        folderPollTimer = null;
        progressBarFill.style.width = "100%";
        folderProgressText.textContent = `✅ Готово! Все ${st.total} стр. за ${st.elapsed_sec}с`;
        statusText.textContent = "Папка с переводом открыта в Проводнике!";
        folderBtn.disabled = false;
        folderStopBtn.style.display = "none";
      } else if (st.status === "stopped") {
        clearInterval(folderPollTimer);
        folderPollTimer = null;
        folderProgressText.textContent = `⏸ Остановлено на ${st.current}/${st.total} стр. (прогресс сохранен)`;
        statusText.textContent = "Перевод папки поставлен на паузу.";
        folderBtn.disabled = false;
        folderStopBtn.style.display = "none";
      } else if (st.status === "error") {
        clearInterval(folderPollTimer);
        folderPollTimer = null;
        folderProgressText.textContent = `❌ Ошибка: ${st.error}`;
        folderBtn.disabled = false;
        folderStopBtn.style.display = "none";
      }
    }, 1000);
  }

  folderStopBtn.addEventListener("click", async () => {
    folderProgressText.textContent = "⏸ Остановка перевода папки...";
    await safeSendMessage({ action: "STOP_FOLDER_TRANSLATION" });
  });

  folderBtn.addEventListener("click", async () => {
    statusText.textContent = "Выберите папку в открывшемся окне Windows...";
    folderBtn.disabled = true;

    const res = await safeSendMessage({
      action: "START_FOLDER_TRANSLATION",
      lang: settings.lang,
      targetLang: settings.targetLang,
      model: settings.model,
      uppercase: settings.uppercase,
      minFontSize: settings.minFontSize,
      maxFontSize: settings.maxFontSize,
      overflowMode: settings.overflowMode,
      detectFreeText: settings.detectFreeText,
      textFilterMode: settings.textFilterMode,
      skipSighs: settings.skipSighs,
      skipSfx: settings.skipSfx,
      minTextLength: settings.minTextLength,
    });

    if (!res || !res.ok) {
      // Сервер мог успеть запустить перевод, даже если ответ потерялся
      // (например, истёк таймаут ожидания при выборе папки). Проверяем
      // фактический статус, прежде чем показывать пользователю ошибку.
      const stRes = await safeSendMessage({ action: "GET_FOLDER_STATUS" });
      if (stRes && stRes.ok && stRes.data && stRes.data.status === "running") {
        statusText.textContent = "Перевод папки уже идёт...";
        pollFolderStatus();
        return;
      }
      statusText.textContent = `Ошибка: ${(res && res.error) || "Сервер недоступен"}`;
      folderBtn.disabled = false;
      return;
    }

    if (res.data.status === "cancelled") {
      statusText.textContent = "Выбор папки отменен.";
      folderBtn.disabled = false;
      return;
    }

    statusText.textContent = `Перевод папки (${res.data.total_images} стр., пропущено готовых: ${res.data.already_translated || 0})...`;
    pollFolderStatus();
  });

  translateAllBtn.addEventListener("click", translateAllVisibleImages);
  translateViewportBtn.addEventListener("click", translateViewportImagesOnly);
  toggleOrigBtn.addEventListener("click", toggleOriginalAll);

  const onRuntimeMessage = (msg) => {
    if (msg.action === "TOGGLE_PANEL") {
      const willOpen = !panel.classList.contains("mt-open");
      panel.classList.toggle("mt-open", willOpen);
      if (willOpen) {
        applySavedPanelPosition();
        checkServerAndLoadModels(false, false);
      }
    } else if (msg.action === "TRIGGER_TRANSLATE_PAGE") {
      if (!panel.classList.contains("mt-open")) {
        panel.classList.add("mt-open");
        applySavedPanelPosition();
      }
      translateAllVisibleImages();
    }
  };
  chrome.runtime.onMessage.addListener(onRuntimeMessage);

  const onKeyDown = (e) => {
    if (e.altKey && !e.ctrlKey && !e.metaKey && (e.code === "KeyT" || e.key === "t" || e.key === "T")) {
      e.preventDefault();
      translateAllVisibleImages();
    }
  };
  window.addEventListener("keydown", onKeyDown);

  scanPageImages();
  const observer = new MutationObserver(() => scanPageImages());
  observer.observe(document.body, { childList: true, subtree: true });

  window.__mtCleanup = () => {
    observer.disconnect();
    window.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("scroll", refreshActiveBadgePositions);
    if (folderPollTimer) clearInterval(folderPollTimer);
    if (pullPollTimer) clearInterval(pullPollTimer);
    if (badgesLayer) badgesLayer.remove();
    try {
      chrome.runtime.onMessage.removeListener(onRuntimeMessage);
    } catch (_) {}
  };
})();
