/**
 * Content Script (v1.0.0) — Локальный переводчик Манги / Манхвы / Маньхуа
 *
 * Изменения v1.0.0:
 * 1. Добавлен целевой язык «🔤 Romaji (Ромадзи)» наряду с English и Русским.
 * 2. Добавлен выбор минимального (`minFontSize`) и максимального (`maxFontSize`) размера шрифта.
 * 3. Добавлен переключатель «Переводить текст вне бабблов (надписи, фон)» (`detectFreeText`).
 * 4. Добавлена возможность повторно перевести любую отдельную страницу по клику на бейдж и остановка перевода папки.
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
    uppercase: true,
    minFontSize: 12,
    maxFontSize: 32,
    textFilterMode: "dialogue_only",
    skipSighs: true,
    skipSfx: true,
    detectFreeText: false,
    minTextLength: 2,
    autoSaveWeb: true,
    webSaveDir: "",
    unslothUrl: "https://moduleapi.pro/v1",
    unslothApiKey: "",
  };

  let settings = { ...DEFAULT_SETTINGS };
  let availableModelsCache = [];

  function migrateAndLoadSettings(onLoaded) {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      chrome.storage.local.get(
        [
          "mt_source_lang",
          "mt_target_lang",
          "mt_model_v8",
          "mt_uppercase",
          "mt_min_font_size",
          "mt_max_font_size",
          "mt_text_filter_mode",
          "mt_skip_sighs",
          "mt_skip_sfx",
          "mt_detect_free_text",
          "mt_min_text_length",
          "mt_auto_save_web",
          "mt_web_save_dir",
          "mt_unsloth_url",
          "mt_unsloth_key",
        ],
        (stored) => {
          const s = stored || {};
          settings.lang = s.mt_source_lang || DEFAULT_SETTINGS.lang;
          settings.targetLang = s.mt_target_lang || DEFAULT_SETTINGS.targetLang;
          settings.model = s.mt_model_v8 || DEFAULT_SETTINGS.model;
          settings.uppercase =
            s.mt_uppercase !== undefined ? Boolean(s.mt_uppercase) : DEFAULT_SETTINGS.uppercase;
          settings.minFontSize =
            parseInt(String(s.mt_min_font_size || DEFAULT_SETTINGS.minFontSize), 10) || 12;
          settings.maxFontSize =
            parseInt(String(s.mt_max_font_size || DEFAULT_SETTINGS.maxFontSize), 10) || 32;
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
          mt_uppercase: Boolean(settings.uppercase),
          mt_min_font_size: Number(settings.minFontSize) || 12,
          mt_max_font_size: Number(settings.maxFontSize) || 32,
          mt_text_filter_mode: settings.textFilterMode || "dialogue_only",
          mt_skip_sighs: Boolean(settings.skipSighs),
          mt_skip_sfx: Boolean(settings.skipSfx),
          mt_detect_free_text: Boolean(settings.detectFreeText),
          mt_min_text_length: Number(settings.minTextLength) || 2,
          mt_auto_save_web: Boolean(settings.autoSaveWeb),
          mt_web_save_dir: settings.webSaveDir || "",
          mt_unsloth_url: settings.unslothUrl,
          mt_unsloth_key: settings.unslothApiKey,
        });
      } catch (_) {}
    }
  }

  let showingOriginal = false;
  let folderPollTimer = null;
  let isCheckingServer = false;

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

  if (!document.getElementById("mt-floating-styles")) {
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
        gap: 8px;
        background: rgba(16, 16, 22, 0.98);
        color: #f3f4f6;
        padding: 13px 15px;
        border-radius: 14px;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 13px;
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(14px);
        width: 335px;
      }
      #mt-floating-panel.mt-open {
        display: flex !important;
      }
      .mt-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 7px;
        margin-bottom: 2px;
      }
      .mt-title {
        font-weight: 700;
        font-size: 13px;
        color: #a5b4fc;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .mt-server-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #ef4444;
        display: inline-block;
        box-shadow: 0 0 6px #ef4444;
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
      }
      .mt-icon-btn:hover {
        background: rgba(255, 255, 255, 0.18);
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
        background: #1e1e2a;
        color: #fff;
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 6px;
        padding: 5px 7px;
        font-size: 12px;
        outline: none;
        flex: 1;
        min-width: 0;
        cursor: pointer;
        pointer-events: auto !important;
      }
      .mt-input {
        cursor: text;
      }
      .mt-provider-pills {
        display: flex;
        gap: 5px;
        width: 100%;
      }
      .mt-pill-btn {
        flex: 1;
        background: rgba(255, 255, 255, 0.07);
        color: #d1d5db;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 7px;
        padding: 5px 6px;
        font-size: 11.5px;
        font-weight: 600;
        cursor: pointer;
        text-align: center;
        transition: all 0.15s;
      }
      .mt-pill-btn:hover {
        background: rgba(255, 255, 255, 0.15);
      }
      .mt-pill-btn.active {
        background: rgba(79, 70, 229, 0.38);
        border-color: #818cf8;
        color: #fff;
      }
      .mt-pill-btn.unsloth.active {
        background: rgba(5, 150, 105, 0.42);
        border-color: #34d399;
        color: #fff;
      }
      #mt-custom-model-picker {
        display: none;
        flex-direction: column;
        gap: 4px;
        max-height: 170px;
        overflow-y: auto;
        background: #14141d;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 6px;
      }
      .mt-model-item {
        background: rgba(255, 255, 255, 0.05);
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
      }
      .mt-model-item:hover {
        background: rgba(99, 102, 241, 0.25);
        border-color: rgba(129, 140, 248, 0.5);
      }
      .mt-model-item.selected {
        background: rgba(79, 70, 229, 0.35);
        border-color: #818cf8;
        font-weight: 600;
      }
      .mt-btn {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 8px 11px;
        font-weight: 600;
        font-size: 12.5px;
        cursor: pointer;
        transition: opacity 0.15s, transform 0.1s;
        text-align: center;
      }
      .mt-btn:hover { opacity: 0.93; }
      .mt-btn:active { transform: scale(0.98); }
      .mt-btn.secondary {
        background: rgba(255, 255, 255, 0.1);
        font-weight: 500;
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
      }
      #mt-settings-drawer {
        display: none;
        flex-direction: column;
        gap: 7px;
        background: rgba(0, 0, 0, 0.35);
        padding: 9px;
        border-radius: 9px;
        border: 1px solid rgba(255, 255, 255, 0.09);
        margin-top: 2px;
      }
      #mt-folder-progress-wrap {
        display: none;
        flex-direction: column;
        gap: 4px;
        background: rgba(255, 255, 255, 0.06);
        padding: 7px 9px;
        border-radius: 8px;
        font-size: 11px;
      }
      .mt-progress-bar-bg {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 4px;
        overflow: hidden;
      }
      #mt-progress-bar-fill {
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
        background: rgba(17, 24, 39, 0.88) !important;
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
  }

  const panel = document.createElement("div");
  panel.id = "mt-floating-panel";
  panel.innerHTML = `
    <div class="mt-header-row">
      <div class="mt-title">
        <span id="mt-server-dot" class="mt-server-dot" title="Статус локального сервера"></span>
        <span>Manga Translator v1.0.0</span>
      </div>
      <div style="display:flex; gap:4px;">
        <button id="mt-web-ui-btn" class="mt-icon-btn" title="Открыть локальную Веб-Студию (http://127.0.0.1:5000/)">🌐</button>
        <button id="mt-refresh-btn" class="mt-icon-btn" title="Обновить статус и список моделей">🔄</button>
        <button id="mt-settings-toggle" class="mt-icon-btn" title="Настройки Unsloth, мин/макс шрифта и текста вне бабблов">⚙️</button>
        <button id="mt-close-panel-btn" class="mt-icon-btn" title="Скрыть окно">✕</button>
      </div>
    </div>

    <div class="mt-row">
      <span class="mt-label">Оригинал:</span>
      <select id="mt-source-lang" class="mt-select">
        <option value="auto">✨ Авто (JA / KO / ZH)</option>
        <option value="ja">🇯🇵 Японский (Манга)</option>
        <option value="ko">🇰🇷 Корейский (Манхва)</option>
        <option value="zh">🇨🇳 Китайский (Маньхуа)</option>
      </select>
    </div>

    <div class="mt-row">
      <span class="mt-label">Перевод:</span>
      <select id="mt-target-lang" class="mt-select">
        <option value="en">🇬🇧 English</option>
        <option value="ru">🇷🇺 Русский</option>
        <option value="romaji">🔤 Romaji (Ромадзи)</option>
      </select>
    </div>

    <div class="mt-provider-pills">
      <button id="mt-pill-auto" type="button" class="mt-pill-btn" title="Использовать DeepSeek Flash (быстрая и экономная)">⚡ Flash</button>
      <button id="mt-pill-unsloth" type="button" class="mt-pill-btn unsloth" title="Использовать DeepSeek Pro (максимальное качество)">🧠 Pro</button>
      <button id="mt-pill-list" type="button" class="mt-pill-btn" title="Показать список всех моделей">📋 Модели</button>
    </div>

    <div id="mt-custom-model-picker"></div>

    <button id="mt-start-server-btn" class="mt-btn server-start" style="display:none;">
      🚀 Запустить локальный сервер
    </button>

    <button id="mt-translate-all-btn" class="mt-btn">⚡ Перевести страницу (Alt+T)</button>
    <div style="display:flex; gap:6px;">
      <button id="mt-folder-btn" class="mt-btn folder" style="flex:1;">📁 Перевести папку</button>
      <button id="mt-open-saved-btn" class="mt-btn secondary" title="Открыть папку на ПК, куда автоматически сохраняется переведенная манга из браузера">📂 Сохранения</button>
    </div>

    <div id="mt-folder-progress-wrap">
      <div id="mt-folder-progress-text">Подготовка...</div>
      <div class="mt-progress-bar-bg"><div id="mt-progress-bar-fill"></div></div>
    </div>

    <button id="mt-toggle-orig-btn" class="mt-btn secondary" style="display:none;">
      👁 Показать оригинал
    </button>

    <div id="mt-settings-drawer">
      <div class="mt-row">
        <label class="mt-label" style="cursor:pointer; display:flex; align-items:center; gap:6px; color:#34d399; font-weight:600;">
          <input id="mt-autosave-chk" type="checkbox" checked />
          💾 Сохранять перевод в папку на ПК
        </label>
      </div>
      <div class="mt-row" id="mt-save-dir-row">
        <span class="mt-label" title="Папка автосохранения манги из браузера">Папка:</span>
        <input id="mt-web-save-dir" class="mt-input" type="text" placeholder="По умолчанию: Saved_Manga" />
        <button id="mt-pick-save-dir-btn" type="button" class="mt-icon-btn" title="Выбрать папку на компьютере">📁</button>
      </div>
      <div class="mt-row">
        <span class="mt-label">Unsloth URL:</span>
        <input id="mt-unsloth-url" class="mt-input" type="text" placeholder="http://127.0.0.1:8888" />
      </div>
      <div class="mt-row">
        <span class="mt-label">API Key:</span>
        <input id="mt-unsloth-key" class="mt-input" type="password" placeholder="Пусто, если без пароля" />
      </div>
      <div class="mt-row">
        <span class="mt-label">Модель вручную:</span>
        <input id="mt-manual-model" class="mt-input" type="text" placeholder="напр. unsloth::default" />
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
          🚫 Не переводить спецэффекты (SFX)
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
        <button id="mt-save-settings-btn" class="mt-btn secondary" style="flex:1; padding:5px;">💾 Сохранить</button>
        <button id="mt-stop-server-btn" class="mt-btn secondary" style="color:#fca5a5; padding:5px 8px;" title="Выключить локальный сервер">⏹ Стоп</button>
      </div>
    </div>

    <div id="mt-status-text" class="mt-status">Нажмите 🔄 для проверки сервера</div>
  `;
  document.body.appendChild(panel);

  ["mousedown", "mouseup", "click", "dblclick", "keydown"].forEach((evtName) => {
    panel.addEventListener(evtName, (e) => {
      e.stopPropagation();
    });
  });

  const serverDot = document.getElementById("mt-server-dot");
  const sourceLangSelect = document.getElementById("mt-source-lang");
  const targetLangSelect = document.getElementById("mt-target-lang");
  const pillAuto = document.getElementById("mt-pill-auto");
  const pillUnsloth = document.getElementById("mt-pill-unsloth");
  const pillList = document.getElementById("mt-pill-list");
  const customModelPicker = document.getElementById("mt-custom-model-picker");
  const startServerBtn = document.getElementById("mt-start-server-btn");
  const translateAllBtn = document.getElementById("mt-translate-all-btn");
  const folderBtn = document.getElementById("mt-folder-btn");
  const folderProgressWrap = document.getElementById("mt-folder-progress-wrap");
  const folderProgressText = document.getElementById("mt-folder-progress-text");
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

  function updateProviderPillsUI() {
    const isUnsloth = settings.model && settings.model.startsWith("unsloth::");
    const isAuto = !settings.model || settings.model === "auto";
    pillAuto.classList.toggle("active", isAuto);
    pillUnsloth.classList.toggle("active", Boolean(isUnsloth));
    pillList.classList.toggle("active", !isAuto && !isUnsloth);
  }

  function syncUiFromSettings() {
    sourceLangSelect.value = settings.lang;
    targetLangSelect.value = settings.targetLang;
    unslothUrlInput.value = settings.unslothUrl;
    unslothKeyInput.value = settings.unslothApiKey;
    manualModelInput.value = settings.model !== "auto" ? settings.model : "";
    minFontSlider.value = String(settings.minFontSize);
    minFontVal.textContent = String(settings.minFontSize);
    fontSlider.value = String(settings.maxFontSize);
    fontVal.textContent = String(settings.maxFontSize);
    autoSaveChk.checked = settings.autoSaveWeb !== false;
    webSaveDirInput.value = settings.webSaveDir || "";
    filterModeSel.value = settings.textFilterMode || "dialogue_only";
    skipSighsChk.checked = settings.skipSighs !== false;
    skipSfxChk.checked = settings.skipSfx !== false;
    freeTextChk.checked = Boolean(settings.detectFreeText);
    uppercaseChk.checked = settings.uppercase;
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

  async function applySelectedModel(newModelId) {
    settings.model = newModelId || "auto";
    manualModelInput.value = settings.model !== "auto" ? settings.model : "";
    saveLocalSettings();
    updateProviderPillsUI();

    statusText.textContent = "Переключение модели...";
    const res = await safeSendMessage({
      action: "SAVE_SERVER_SETTINGS",
      unslothUrl: settings.unslothUrl,
      unslothApiKey: settings.unslothApiKey,
      selectedModel: settings.model,
    });

    if (res && res.ok && res.data) {
      availableModelsCache = res.data.models || availableModelsCache;
      const activeMod = res.data.active_model || settings.model;
      renderCustomModelList(availableModelsCache, activeMod);
      const shortLabel = activeMod.replace(/^.*?::/, "");
      const prefix = activeMod.startsWith("unsloth::") ? "🚀 Unsloth" : "🦙 Ollama";
      statusText.textContent = `✅ Активна: ${prefix} (${shortLabel})`;
    } else {
      statusText.textContent = `Выбрана: ${settings.model}`;
    }
  }

  pillAuto.addEventListener("click", () => {
    applySelectedModel("unsloth::deepseek-v4-flash");
  });

  pillUnsloth.addEventListener("click", () => {
    applySelectedModel("unsloth::deepseek-v4-pro");
  });

  pillList.addEventListener("click", () => {
    const isHidden =
      customModelPicker.style.display === "none" || !customModelPicker.style.display;
    customModelPicker.style.display = isHidden ? "flex" : "none";
    if (isHidden) {
      checkServerAndLoadModels(true, true);
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

  autoSaveChk.addEventListener("change", () => {
    settings.autoSaveWeb = autoSaveChk.checked;
    saveLocalSettings();
    safeSendMessage({
      action: "SAVE_SERVER_SETTINGS",
      unslothUrl: settings.unslothUrl,
      unslothApiKey: settings.unslothApiKey,
      selectedModel: settings.model,
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
  });

  skipSighsChk.addEventListener("change", () => {
    settings.skipSighs = skipSighsChk.checked;
    saveLocalSettings();
  });

  skipSfxChk.addEventListener("change", () => {
    settings.skipSfx = skipSfxChk.checked;
    saveLocalSettings();
  });

  freeTextChk.addEventListener("change", () => {
    settings.detectFreeText = freeTextChk.checked;
    saveLocalSettings();
  });

  uppercaseChk.addEventListener("change", () => {
    settings.uppercase = uppercaseChk.checked;
    saveLocalSettings();
  });

  document.getElementById("mt-settings-toggle").addEventListener("click", () => {
    const isHidden = settingsDrawer.style.display === "none" || !settingsDrawer.style.display;
    settingsDrawer.style.display = isHidden ? "flex" : "none";
  });

  document.getElementById("mt-refresh-btn").addEventListener("click", () => {
    checkServerAndLoadModels(false, true);
  });

  document.getElementById("mt-save-settings-btn").addEventListener("click", async () => {
    settings.unslothUrl = unslothUrlInput.value.trim() || "http://127.0.0.1:8888";
    settings.unslothApiKey = unslothKeyInput.value.trim();
    settings.autoSaveWeb = autoSaveChk.checked;
    settings.webSaveDir = webSaveDirInput.value.trim();
    const manualMod = manualModelInput.value.trim();
    if (manualMod) {
      settings.model = manualMod;
    }
    saveLocalSettings();
    updateProviderPillsUI();
    statusText.textContent = "Сохранение настроек...";
    const res = await safeSendMessage({
      action: "SAVE_SERVER_SETTINGS",
      unslothUrl: settings.unslothUrl,
      unslothApiKey: settings.unslothApiKey,
      selectedModel: settings.model,
      autoSaveWeb: settings.autoSaveWeb,
      webSaveDir: settings.webSaveDir,
    });
    if (res && res.ok) {
      availableModelsCache = res.data.models || [];
      renderCustomModelList(availableModelsCache, res.data.active_model);
      const shortActive = (res.data.active_model || settings.model).replace(/^.*?::/, "");
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
    customModelPicker.innerHTML = "";

    const allItems = [
      { id: "auto", name: `✨ Авто (${(activeModel || "qwen3.5").replace(/^.*?::/, "")})` },
      ...(Array.isArray(models) ? models : []),
    ];

    allItems.forEach((m) => {
      const btn = document.createElement("button");
      btn.type = "button";
      const isSel =
        settings.model === m.id || (settings.model === "auto" && m.id === "auto");
      btn.className = "mt-model-item" + (isSel ? " selected" : "");
      btn.innerHTML = `<span>${m.name}</span><span>${isSel ? "✅" : ""}</span>`;
      btn.addEventListener("click", () => {
        applySelectedModel(m.id);
        customModelPicker.style.display = "none";
      });
      customModelPicker.appendChild(btn);
    });
  }

  async function checkServerAndLoadModels(silent = false, forceRefresh = false) {
    if (isCheckingServer) return false;
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

      const serverSelected = healthRes.data?.selected_model;
      if (serverSelected && serverSelected !== "auto" && settings.model === "auto") {
        settings.model = serverSelected;
        saveLocalSettings();
      }
      if (healthRes.data?.web_save_dir && !settings.webSaveDir) {
        settings.webSaveDir = healthRes.data.web_save_dir;
        webSaveDirInput.value = settings.webSaveDir;
      }

      let activeMod =
        healthRes.data?.active_model || healthRes.data?.ollama_model || "qwen3.5";
      const easyOcrOk = healthRes.data?.easyocr_installed;
      let modelsList = Array.isArray(healthRes.data?.models) ? healthRes.data.models : null;

      if (forceRefresh || !modelsList) {
        const modelsRes = await safeSendMessage({
          action: "GET_MODELS",
          refresh: Boolean(forceRefresh),
        });
        if (modelsRes && modelsRes.ok && modelsRes.data) {
          modelsList = modelsRes.data.models || [];
          activeMod = modelsRes.data.active_model || activeMod;
        }
      }

      availableModelsCache = modelsList || [];
      renderCustomModelList(availableModelsCache, activeMod);
      updateProviderPillsUI();

      const shortMod = activeMod.replace(/^.*?::/, "");
      const providerIcon = activeMod.startsWith("unsloth::") ? "🚀 Unsloth:" : "🟢";
      statusText.textContent = easyOcrOk
        ? `${providerIcon} ${shortMod} (JA/KO/ZH)`
        : `${providerIcon} ${shortMod} (JA)`;
      return true;
    } finally {
      isCheckingServer = false;
    }
  }

  function isMangaCandidate(img) {
    if (!img || !img.isConnected) return false;
    if (img.closest(".gallery, .thumb-container, #thumbnail-container")) {
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
      let imageB64 = img.dataset.mtTranslated === "true" ? null : tryExtractCanvasBase64(img);
      if (!imageB64 && srcUrl.startsWith("blob:")) {
        imageB64 = await tryFetchInPageContext(srcUrl);
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
        detectFreeText: settings.detectFreeText,
        textFilterMode: settings.textFilterMode,
        skipSighs: settings.skipSighs,
        skipSfx: settings.skipSfx,
        minTextLength: settings.minTextLength,
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
        if (badge) badge.textContent = "Нет текста";
        return { ok: true, empty: true };
      }

      if (!img.dataset.mtOriginalSrc) {
        img.dataset.mtOriginalSrc = srcUrl;
      }
      if (img.srcset) {
        img.dataset.mtOriginalSrcset = img.srcset;
        img.removeAttribute("srcset");
      }

      const finalImgSrc = result.translated_image.startsWith("data:")
        ? result.translated_image
        : `data:image/webp;base64,${result.translated_image}`;

      img.src = finalImgSrc;
      img.dataset.mtTranslatedSrc = finalImgSrc;
      img.dataset.mtTranslated = "true";

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
  window.addEventListener("resize", refreshActiveBadgePositions, { passive: true });

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

  async function translateAllVisibleImages() {
    scanPageImages();
    const candidates = Array.from(document.querySelectorAll("img")).filter(
      (img) => isMangaCandidate(img) && img.dataset.mtTranslated !== "true"
    );

    if (candidates.length === 0) {
      statusText.textContent = "Страницы для перевода не найдены.";
      return;
    }

    translateAllBtn.disabled = true;
    let okCount = 0;
    let failCount = 0;
    let lastError = "";

    for (let i = 0; i < candidates.length; i++) {
      statusText.textContent = `⏳ Перевод стр. ${i + 1} из ${candidates.length} (успешно: ${okCount})...`;
      const res = await translateSingleImage(candidates[i], candidates[i].__mtBadge);
      if (res && res.ok) {
        okCount++;
      } else if (!res?.skipped) {
        failCount++;
        lastError = res?.error || "Ошибка";
      }
    }

    if (failCount === 0) {
      statusText.textContent = `✅ Готово! Переведено: ${okCount} из ${candidates.length}`;
    } else if (okCount > 0) {
      statusText.textContent = `⚠️ Переведено: ${okCount}, ошибок: ${failCount} (${lastError})`;
    } else {
      statusText.textContent = `❌ Ошибка (${failCount} стр.): ${lastError}`;
    }
    translateAllBtn.disabled = false;
  }

  function toggleOriginalAll() {
    const translatedImgs = Array.from(
      document.querySelectorAll('img[data-mt-translated="true"]')
    );
    if (translatedImgs.length === 0) return;

    showingOriginal = !showingOriginal;
    translatedImgs.forEach((img) => {
      img.src = showingOriginal ? img.dataset.mtOriginalSrc : img.dataset.mtTranslatedSrc;
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
      } else if (st.status === "completed") {
        clearInterval(folderPollTimer);
        folderPollTimer = null;
        progressBarFill.style.width = "100%";
        folderProgressText.textContent = `✅ Готово! Все ${st.total} стр. за ${st.elapsed_sec}с`;
        statusText.textContent = "Папка с переводом открыта в Проводнике!";
        folderBtn.disabled = false;
      } else if (st.status === "stopped") {
        clearInterval(folderPollTimer);
        folderPollTimer = null;
        folderProgressText.textContent = `⏸ Остановлено на ${st.current}/${st.total} стр. (прогресс сохранен)`;
        folderBtn.disabled = false;
      } else if (st.status === "error") {
        clearInterval(folderPollTimer);
        folderPollTimer = null;
        folderProgressText.textContent = `❌ Ошибка: ${st.error}`;
        folderBtn.disabled = false;
      }
    }, 1000);
  }

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
      detectFreeText: settings.detectFreeText,
    });

    if (!res || !res.ok) {
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
  toggleOrigBtn.addEventListener("click", toggleOriginalAll);

  const onRuntimeMessage = (msg) => {
    if (msg.action === "TOGGLE_PANEL") {
      const willOpen = !panel.classList.contains("mt-open");
      panel.classList.toggle("mt-open", willOpen);
      if (willOpen) {
        checkServerAndLoadModels(false, false);
      }
    } else if (msg.action === "TRIGGER_TRANSLATE_PAGE") {
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
    window.removeEventListener("resize", refreshActiveBadgePositions);
    if (folderPollTimer) clearInterval(folderPollTimer);
    if (badgesLayer) badgesLayer.remove();
    try {
      chrome.runtime.onMessage.removeListener(onRuntimeMessage);
    } catch (_) {}
  };
})();
