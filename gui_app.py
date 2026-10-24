"""
Локальное графическое приложение (GUI) «Miko-chan Manga Studio» (v1.0.0).

Изменения v1.0.0:
1. В окно «⚙️ Настройки» добавлена секция скачивания в 1 клик лучших моделей манги
   (Shisa V2.1 14B, Shisa V2.1 8B, Shisa V2 12B Uncensored, Gemma 3 12B, Qwen 2.5 14B)
   напрямую через локальный сервер Ollama с прогресс-баром и автоматической установкой по умолчанию!
2. Добавлены поля для подключения внешних облачных API-ключей (FunPay PositionAPI, ListAPI, DeepSeek, OpenRouter).
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import httpx
from PIL import Image, ImageDraw, ImageTk

BASE_DIR = Path(__file__).resolve().parent
SERVER_URL = "http://127.0.0.1:5000"
GUI_SETTINGS_FILE = BASE_DIR / "gui_settings.json"
MASCOT_PATH = BASE_DIR / "mascot.png"

BG_DEEP = "#110f1a"
BG_BENTO = "#1b182a"
BG_INNER = "#26223a"
FG_PRIMARY = "#fdf2f8"
FG_SECONDARY = "#cbc3e3"
FG_MUTED = "#9389b8"
ACCENT_SAKURA = "#ec4899"
ACCENT_INDIGO = "#6366f1"
ACCENT_MINT = "#10b981"
ACCENT_ROSE = "#e11d48"

SOURCE_LANG_MAP = {
    "✨ Авто (JA/KO/ZH)": "auto",
    "🇯🇵 Японский": "ja",
    "🇰🇷 Корейский": "ko",
    "🇨🇳 Китайский": "zh",
}

RECOMMENDED_DOWNLOAD_MAP = {
    "🌸 Shisa V2.1 14B (Phi-4, Топ-1 JA➔EN/RU, ~9.0 ГБ)": "hf.co/mradermacher/shisa-v2.1-unphi4-14b-GGUF:Q4_K_M",
    "🌸 Shisa V2.1 8B (Qwen3, Быстрая JA➔EN/RU, ~5.0 ГБ)": "hf.co/mradermacher/shisa-v2.1-qwen3-8b-GGUF:Q4_K_M",
    "🔥 Shisa V2 12B Uncensored (Без цензуры 18+, ~7.5 ГБ)": "hf.co/yamatazen/Shisa-v2-Mistral-Nemo-12B-Abliterated-Q4_K_M-GGUF",
    "💎 Gemma 3 12B (Литературный слог RU/EN, ~8.1 ГБ)": "gemma3:12b",
    "🐉 Qwen 2.5 14B (Универсал JA/KO/ZH + Ромадзи, ~9.0 ГБ)": "qwen2.5:14b",
}

CLOUD_PRESET_MAP = {
    "⚡ DeepSeek Flash (deepseek-v4-flash — быстрая и экономная)": ("https://moduleapi.pro/v1", "unsloth::deepseek-v4-flash"),
    "🧠 DeepSeek Pro (deepseek-v4-pro — максимальное качество)": ("https://moduleapi.pro/v1", "unsloth::deepseek-v4-pro"),
}

TEXT_FILTER_MODE_MAP = {
    "💬 Только диалоги (без спецэффектов и вздохов «а, о, ах, ох»)": "dialogue_only",
    "💭 Диалоги + фон (без вздохов «а, о, ах, ох» и без SFX)": "bubbles_no_sighs",
    "🛡 Минимум замазки — только длинные фразы (от 4+ симв.)": "long_only",
    "🗨 Все облачка диалогов (без текста на фоне, со вздохами)": "bubbles_only",
    "🔊 Переводить вообще всё (диалоги, фон, спецэффекты и вздохи)": "all",
}

OVERFLOW_MODE_MAP = {
    "💬 Авто-расширение баббла (Вариант В)": "expand_bubble",
    "📐 Уменьшать шрифт в рамку (Стандарт)": "shrink",
    "🔓 Разрешить выход за рамки (Вариант Б)": "overflow",
}
OVERFLOW_LABEL_BY_CODE = {v: k for k, v in OVERFLOW_MODE_MAP.items()}


class MikoMangaStudioGUI:
    """Минималистичный Bento-интерфейс переводчика манги с кавайным маскотом Мико-тян (v1.0.0)."""

    def __init__(self, root: tk.Tk) -> None:
        print("[GUI] Инициализация окна Miko-chan Manga Studio v1.0.5...", flush=True)
        self.root = root
        self.root.title("🌸 Miko-chan Manga Studio v1.0.5")
        self.root.geometry("1460x880")
        self.root.minsize(920, 600)
        self.root.configure(bg=BG_DEEP)
        self.is_fullscreen: bool = False
        self.left_sidebar_collapsed: bool = False
        self._resize_job: str | None = None
        self.root.bind("<F11>", lambda _: self._toggle_fullscreen())

        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(600, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

        self.server_process: subprocess.Popen[Any] | None = None
        self.server_online: bool = False
        self.available_models: list[dict[str, str]] = []
        self.active_model_short: str = "Авто"

        self.current_folder: Path | None = None
        self.current_output_dir: Path | None = None
        self.folder_pages: list[dict[str, Any]] = []
        self.selected_page: dict[str, Any] | None = None

        self.is_translating_single: bool = False
        self.is_folder_job_running: bool = False
        self.is_pulling_model: bool = False

        self.mascot_photo: ImageTk.PhotoImage | None = None
        self.preview_orig_pil: Image.Image | None = None
        self.preview_trans_pil: Image.Image | None = None
        self.preview_photo_ref: ImageTk.PhotoImage | None = None
        self.last_blocks: list[dict[str, Any]] = []
        self.is_draw_region_mode: bool = False
        self._draw_start_xy: tuple[int, int] | None = None
        self._draw_rect_id: int | None = None
        self._last_canvas_render_info: dict[str, Any] = {}
        self.show_canvas_pins: tk.BooleanVar = tk.BooleanVar(value=True)
        self._bbox_drag_state: dict[str, Any] | None = None
        self._idle_autosave_job: str | None = None

        self.var_source_lang = tk.StringVar(value="✨ Авто (JA/KO/ZH)")
        self.target_lang: str = "en"
        self.var_model_display = tk.StringVar(value="✨ Автовыбор лучшей модели (auto)")
        self.var_recommended_pull = tk.StringVar(value=list(RECOMMENDED_DOWNLOAD_MAP.keys())[0])
        self.var_cloud_preset = tk.StringVar(value=list(CLOUD_PRESET_MAP.keys())[0])
        self.var_custom_model = tk.StringVar(value="")
        self.var_unsloth_url = tk.StringVar(value="https://moduleapi.pro/v1")
        self.var_api_key = tk.StringVar(value="")
        self.var_min_font = tk.IntVar(value=12)
        self.var_max_font = tk.IntVar(value=32)
        self.var_uppercase = tk.BooleanVar(value=True)
        self.var_overflow_mode = tk.StringVar(value="shrink")
        self.var_overflow_label = tk.StringVar(value="📐 В рамку (Стандарт)")
        self.var_text_filter_label = tk.StringVar(value=list(TEXT_FILTER_MODE_MAP.keys())[0])
        self.var_skip_sighs = tk.BooleanVar(value=True)
        self.var_skip_sfx = tk.BooleanVar(value=True)
        self.var_detect_free_text = tk.BooleanVar(value=False)
        self.var_min_text_len = tk.IntVar(value=2)
        self.var_skip_existing = tk.BooleanVar(value=True)
        self.var_auto_save_boxes = tk.BooleanVar(value=True)
        self.var_auto_save_sec = tk.DoubleVar(value=2.5)
        self.var_auto_save_web = tk.BooleanVar(value=True)
        self.var_web_save_dir = tk.StringVar(value=str(BASE_DIR / "Saved_Manga"))
        self.view_mode: str = "split"

        self._load_gui_settings()
        self._configure_styles()
        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._initial_server_check)
        self.root.after(1200, self._poll_folder_status_loop)
        print("[GUI] Окно интерфейса открыто! Проверка связи с локальным сервером...", flush=True)

    def _load_gui_settings(self) -> None:
        if GUI_SETTINGS_FILE.exists():
            try:
                data = json.loads(GUI_SETTINGS_FILE.read_text(encoding="utf-8"))
                s_code = data.get("lang", "auto")
                for k, v in SOURCE_LANG_MAP.items():
                    if v == s_code:
                        self.var_source_lang.set(k)
                        break
                self.target_lang = data.get("target_lang", "en")
                if self.target_lang not in ("en", "ru", "romaji"):
                    self.target_lang = "en"
                self.var_min_font.set(int(data.get("min_font_size", 12)))
                self.var_max_font.set(int(data.get("max_font_size", 32)))
                self.var_uppercase.set(bool(data.get("uppercase", True)))
                ovf_m = str(data.get("overflow_mode", "shrink")).strip().lower()
                if ovf_m not in ("shrink", "overflow", "expand_bubble"):
                    ovf_m = "shrink"
                self.var_overflow_mode.set(ovf_m)
                for olbl, ocode in OVERFLOW_MODE_MAP.items():
                    if ocode == ovf_m:
                        self.var_overflow_label.set(olbl)
                        break
                self.var_detect_free_text.set(bool(data.get("detect_free_text", False)))
                self.var_skip_sighs.set(bool(data.get("skip_sighs", True)))
                self.var_skip_sfx.set(bool(data.get("skip_sfx", True)))
                self.var_min_text_len.set(int(data.get("min_text_length", 2)))
                saved_mode = data.get("text_filter_mode", "dialogue_only")
                for lbl, m_code in TEXT_FILTER_MODE_MAP.items():
                    if m_code == saved_mode:
                        self.var_text_filter_label.set(lbl)
                        break
                self.var_skip_existing.set(bool(data.get("skip_existing", True)))
                self.var_auto_save_boxes.set(bool(data.get("auto_save_boxes", True)))
                try:
                    self.var_auto_save_sec.set(max(0.5, min(30.0, float(data.get("auto_save_sec", 2.5)))))
                except Exception:
                    self.var_auto_save_sec.set(2.5)
                if data.get("unsloth_url"):
                    self.var_unsloth_url.set(str(data["unsloth_url"]))
                if data.get("unsloth_api_key"):
                    self.var_api_key.set(str(data["unsloth_api_key"]))
                if data.get("last_folder") and Path(data["last_folder"]).is_dir():
                    self.current_folder = Path(data["last_folder"])
            except Exception:
                pass

    def _save_gui_settings(self) -> None:
        data = {
            "lang": SOURCE_LANG_MAP.get(self.var_source_lang.get(), "auto"),
            "target_lang": self.target_lang,
            "min_font_size": int(self.var_min_font.get()),
            "max_font_size": int(self.var_max_font.get()),
            "overflow_mode": self.var_overflow_mode.get(),
            "uppercase": bool(self.var_uppercase.get()),
            "overflow_mode": self.var_overflow_mode.get(),
            "text_filter_mode": TEXT_FILTER_MODE_MAP.get(self.var_text_filter_label.get(), "dialogue_only"),
            "skip_sighs": bool(self.var_skip_sighs.get()),
            "skip_sfx": bool(self.var_skip_sfx.get()),
            "detect_free_text": bool(self.var_detect_free_text.get()),
            "min_text_length": int(self.var_min_text_len.get()),
            "skip_existing": bool(self.var_skip_existing.get()),
            "auto_save_boxes": bool(self.var_auto_save_boxes.get()),
            "auto_save_sec": float(self.var_auto_save_sec.get() or 2.5),
            "unsloth_url": self.var_unsloth_url.get().strip(),
            "unsloth_api_key": self.var_api_key.get().strip(),
            "last_folder": str(self.current_folder) if self.current_folder else None,
        }
        try:
            GUI_SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=BG_DEEP, foreground=FG_PRIMARY, font=("Segoe UI", 10))
        style.configure(
            "Treeview",
            background="#141122",
            foreground=FG_PRIMARY,
            fieldbackground="#141122",
            rowheight=30,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Treeview.Heading",
            background=BG_INNER,
            foreground=FG_SECONDARY,
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
        )
        style.map("Treeview", background=[("selected", ACCENT_SAKURA)], foreground=[("selected", "#ffffff")])

        style.configure(
            "Sakura.Horizontal.TProgressbar",
            troughcolor="#141122",
            background=ACCENT_SAKURA,
            bordercolor="#141122",
            lightcolor=ACCENT_SAKURA,
            darkcolor=ACCENT_INDIGO,
            thickness=8,
        )

    def _load_mascot_avatar(self, size: int = 92) -> ImageTk.PhotoImage | None:
        if not MASCOT_PATH.exists():
            return None
        try:
            img = Image.open(MASCOT_PATH).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
            mask = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(mask)
            draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=18, fill=255)
            out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            out.paste(img, (0, 0), mask=mask)
            return ImageTk.PhotoImage(out)
        except Exception:
            return None

    def _build_ui(self) -> None:
        topbar = tk.Frame(self.root, bg=BG_BENTO, height=58, padx=18, pady=8)
        topbar.pack(fill=tk.X, side=tk.TOP)

        brand_frame = tk.Frame(topbar, bg=BG_BENTO)
        brand_frame.pack(side=tk.LEFT)
        self.btn_toggle_left = tk.Button(
            brand_frame,
            text="◀ Панель",
            bg=BG_INNER,
            fg=FG_PRIMARY,
            activebackground=ACCENT_INDIGO,
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=4,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._toggle_left_sidebar,
        )
        self.btn_toggle_left.pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(
            brand_frame,
            text="🌸 Miko-chan Manga Studio",
            bg=BG_BENTO,
            fg=FG_PRIMARY,
            font=("Segoe UI", 13, "bold"),
        ).pack(side=tk.LEFT)

        lang_bar = tk.Frame(topbar, bg=BG_INNER, padx=8, pady=4)
        lang_bar.pack(side=tk.LEFT, padx=28)

        cb_src = ttk.Combobox(
            lang_bar,
            textvariable=self.var_source_lang,
            values=list(SOURCE_LANG_MAP.keys()),
            state="readonly",
            width=18,
        )
        cb_src.pack(side=tk.LEFT, padx=(4, 6))
        cb_src.bind("<<ComboboxSelected>>", lambda _: self._save_gui_settings())

        tk.Label(lang_bar, text="➔", bg=BG_INNER, fg=FG_MUTED, font=("Segoe UI", 10, "bold")).pack(
            side=tk.LEFT, padx=4
        )

        self.lang_pill_btns: dict[str, tk.Button] = {}
        for code, label in (("en", "🇬🇧 English"), ("ru", "🇷🇺 Русский"), ("romaji", "🔤 Ромадзи")):
            btn = tk.Button(
                lang_bar,
                text=label,
                bd=0,
                padx=12,
                pady=4,
                font=("Segoe UI", 9, "bold"),
                cursor="hand2",
                command=lambda c=code: self._set_target_lang(c),
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.lang_pill_btns[code] = btn

        right_top = tk.Frame(topbar, bg=BG_BENTO)
        right_top.pack(side=tk.RIGHT)

        self.lbl_summary_chip = tk.Label(
            right_top,
            text="",
            bg=BG_INNER,
            fg=FG_SECONDARY,
            font=("Segoe UI", 9),
            padx=10,
            pady=5,
            cursor="hand2",
        )
        self.lbl_summary_chip.pack(side=tk.LEFT, padx=6)
        self.lbl_summary_chip.bind("<Button-1>", lambda _: self._open_settings_modal())

        btn_diag = tk.Button(
            right_top,
            text="🩺 Диагностика / Логи",
            bg="#065f46",
            fg="#a7f3d0",
            activebackground=ACCENT_MINT,
            activeforeground="#ffffff",
            bd=0,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._open_diagnostics_modal,
        )
        btn_diag.pack(side=tk.LEFT, padx=4)

        btn_settings = tk.Button(
            right_top,
            text="⚙️ Настройки / Модели",
            bg=BG_INNER,
            fg=FG_PRIMARY,
            activebackground=ACCENT_SAKURA,
            activeforeground="#ffffff",
            bd=0,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._open_settings_modal,
        )
        btn_settings.pack(side=tk.LEFT, padx=4)

        btn_app_mode = tk.Button(
            right_top,
            text="✨ Веб-Аниме Студия",
            bg="#312e81",
            fg="#e0e7ff",
            activebackground=ACCENT_INDIGO,
            activeforeground="#ffffff",
            bd=0,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._launch_standalone_web_app,
        )
        btn_app_mode.pack(side=tk.LEFT, padx=(4, 0))

        self.btn_fullscreen = tk.Button(
            right_top,
            text="⛶ Полный экран",
            bg=BG_INNER,
            fg=FG_SECONDARY,
            activebackground=ACCENT_SAKURA,
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._toggle_fullscreen,
        )
        self.btn_fullscreen.pack(side=tk.LEFT, padx=(4, 0))

        workspace = tk.Frame(self.root, bg=BG_DEEP, padx=14, pady=12)
        self.workspace_frame = workspace
        workspace.pack(fill=tk.BOTH, expand=True)

        left_col = tk.Frame(workspace, bg=BG_DEEP, width=375)
        left_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left_col.pack_propagate(False)
        self.left_col_frame = left_col

        mascot_card = tk.Frame(
            left_col,
            bg=BG_BENTO,
            padx=12,
            pady=12,
            highlightbackground="#4c1d95",
            highlightthickness=1,
        )
        mascot_card.pack(fill=tk.X, pady=(0, 10))

        self.mascot_photo = self._load_mascot_avatar(88)
        if self.mascot_photo is not None:
            lbl_avatar = tk.Label(mascot_card, image=self.mascot_photo, bg=BG_BENTO)
            lbl_avatar.pack(side=tk.LEFT, padx=(0, 10))

        speech_wrap = tk.Frame(mascot_card, bg=BG_BENTO)
        speech_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        name_row = tk.Frame(speech_wrap, bg=BG_BENTO)
        name_row.pack(fill=tk.X)
        tk.Label(
            name_row,
            text="✨ Мико-тян",
            bg=BG_BENTO,
            fg="#f472b6",
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT)
        self.lbl_server_dot = tk.Label(
            name_row,
            text="⏳ Запуск...",
            bg=BG_BENTO,
            fg=FG_MUTED,
            font=("Segoe UI", 8, "bold"),
        )
        self.lbl_server_dot.pack(side=tk.RIGHT)

        self.lbl_miko_speech = tk.Label(
            speech_wrap,
            text="Привет! Версия v1.0.0 готова: в редакторе есть кнопка «🗑 Удалить перевод», а в настройках — фильтр вздохов/SFX и пресеты DeepSeek Flash/Pro~ 🌸",
            bg="#141122",
            fg=FG_PRIMARY,
            font=("Segoe UI", 9),
            wraplength=235,
            justify=tk.LEFT,
            anchor="w",
            padx=10,
            pady=8,
        )
        self.lbl_miko_speech.pack(fill=tk.X, pady=(4, 0))

        chapter_card = tk.Frame(
            left_col,
            bg=BG_BENTO,
            padx=14,
            pady=14,
            highlightbackground="#2e294e",
            highlightthickness=1,
        )
        chapter_card.pack(fill=tk.BOTH, expand=True)

        pick_row = tk.Frame(chapter_card, bg=BG_BENTO)
        pick_row.pack(fill=tk.X, pady=(0, 10))

        tk.Button(
            pick_row,
            text="📁 Выбрать папку",
            bg=BG_INNER,
            fg=FG_PRIMARY,
            activebackground=ACCENT_INDIGO,
            activeforeground="#ffffff",
            bd=0,
            pady=8,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._choose_folder_dialog,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        tk.Button(
            pick_row,
            text="📄 Одна страница",
            bg=BG_INNER,
            fg=FG_PRIMARY,
            activebackground=ACCENT_INDIGO,
            activeforeground="#ffffff",
            bd=0,
            pady=8,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._choose_single_file_dialog,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        prog_header = tk.Frame(chapter_card, bg=BG_BENTO)
        prog_header.pack(fill=tk.X)
        self.lbl_folder_name = tk.Label(
            prog_header,
            text="Глава не выбрана",
            bg=BG_BENTO,
            fg=FG_SECONDARY,
            font=("Segoe UI", 9, "bold"),
        )
        self.lbl_folder_name.pack(side=tk.LEFT)
        self.lbl_folder_counts = tk.Label(
            prog_header,
            text="0 / 0 стр.",
            bg=BG_BENTO,
            fg=FG_MUTED,
            font=("Segoe UI", 9),
        )
        self.lbl_folder_counts.pack(side=tk.RIGHT)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            chapter_card,
            variable=self.progress_var,
            maximum=100.0,
            style="Sakura.Horizontal.TProgressbar",
        )
        self.progress_bar.pack(fill=tk.X, pady=(5, 10))

        tree_wrap = tk.Frame(chapter_card, bg="#141122")
        tree_wrap.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.tree_pages = ttk.Treeview(
            tree_wrap,
            columns=("page", "status"),
            show="headings",
            selectmode="browse",
        )
        self.tree_pages.heading("page", text="Страница")
        self.tree_pages.heading("status", text="Статус")
        self.tree_pages.column("page", width=215, anchor=tk.W, stretch=True)
        self.tree_pages.column("status", width=110, anchor=tk.CENTER, stretch=False)

        vsb = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree_pages.yview)
        self.tree_pages.configure(yscrollcommand=vsb.set)
        self.tree_pages.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_pages.bind("<<TreeviewSelect>>", self._on_tree_select)

        self.btn_primary_folder = tk.Button(
            chapter_card,
            text="▶ Перевести папку",
            bg=ACCENT_SAKURA,
            fg="#ffffff",
            activebackground="#db2777",
            activeforeground="#ffffff",
            bd=0,
            pady=11,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            command=self._on_primary_folder_click,
        )
        self.btn_primary_folder.pack(fill=tk.X, pady=(0, 6))

        self.btn_context_page = tk.Button(
            chapter_card,
            text="🎯 Перевести выбранную страницу",
            bg=BG_INNER,
            fg=FG_PRIMARY,
            activebackground=ACCENT_INDIGO,
            activeforeground="#ffffff",
            bd=0,
            pady=8,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._translate_selected_page,
        )
        self.btn_context_page.pack(fill=tk.X, pady=(0, 5))

        self.btn_edit_page = tk.Button(
            chapter_card,
            text="✏️ Редактировать перевод и Ромадзи",
            bg="#1e293b",
            fg="#a7f3d0",
            activebackground=ACCENT_MINT,
            activeforeground="#ffffff",
            bd=0,
            pady=8,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=lambda: self._show_phrases_popup(open_romaji=True),
        )
        self.btn_edit_page.pack(fill=tk.X)

        right_card = tk.Frame(
            workspace,
            bg=BG_BENTO,
            highlightbackground="#2e294e",
            highlightthickness=1,
        )
        right_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        viewer_bar = tk.Frame(right_card, bg=BG_BENTO, padx=14, pady=10)
        viewer_bar.pack(fill=tk.X)

        tk.Button(
            viewer_bar,
            text="◀ Пред.",
            bg=BG_INNER,
            fg=FG_PRIMARY,
            bd=0,
            padx=9,
            pady=5,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=lambda: self._step_page(-1),
        ).pack(side=tk.LEFT, padx=(0, 2))

        tk.Button(
            viewer_bar,
            text="След. ▶",
            bg=BG_INNER,
            fg=FG_PRIMARY,
            bd=0,
            padx=9,
            pady=5,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=lambda: self._step_page(1),
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.view_mode_btns: dict[str, tk.Button] = {}
        for v_code, v_label in (
            ("split", "🔀 Сравнить До и После"),
            ("translated", "🈯 Перевод"),
            ("original", "🇯🇵 Оригинал"),
        ):
            b = tk.Button(
                viewer_bar,
                text=v_label,
                bd=0,
                padx=12,
                pady=5,
                font=("Segoe UI", 9, "bold"),
                cursor="hand2",
                command=lambda m=v_code: self._set_view_mode(m),
            )
            b.pack(side=tk.LEFT, padx=3)
            self.view_mode_btns[v_code] = b

        self.overflow_mode_btns: dict[str, tk.Button] = {}
        for o_code, o_label in (
            ("shrink", "📐 В рамку"),
            ("overflow", "🔓 За рамки (Б)"),
            ("expand_bubble", "💬 Авто-баббл (В)"),
        ):
            ob = tk.Button(
                viewer_bar,
                text=o_label,
                bd=0,
                padx=8,
                pady=5,
                font=("Segoe UI", 8, "bold"),
                cursor="hand2",
                command=lambda m=o_code: self._set_overflow_mode(m),
            )
            ob.pack(side=tk.LEFT, padx=2)
            self.overflow_mode_btns[o_code] = ob

        tk.Button(
            viewer_bar,
            text="📂 Открыть папку перевода",
            bg=BG_INNER,
            fg=FG_SECONDARY,
            bd=0,
            padx=10,
            pady=5,
            font=("Segoe UI", 9),
            cursor="hand2",
            command=self._open_output_folder_in_explorer,
        ).pack(side=tk.RIGHT, padx=3)

        tk.Button(
            viewer_bar,
            text="🔤 Ромадзи",
            bg="#312e81",
            fg="#e0e7ff",
            activebackground=ACCENT_INDIGO,
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=5,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=lambda: self._show_phrases_popup(open_romaji=True),
        ).pack(side=tk.RIGHT, padx=3)

        self.btn_show_phrases = tk.Button(
            viewer_bar,
            text="✏️ Редактор текста (0)",
            bg=BG_INNER,
            fg=FG_PRIMARY,
            activebackground=ACCENT_SAKURA,
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=5,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._show_phrases_popup,
        )
        self.btn_show_phrases.pack(side=tk.RIGHT, padx=3)

        self.btn_draw_region = tk.Button(
            viewer_bar,
            text="➕ Обвести текст мышкой",
            bg="#065f46",
            fg="#a7f3d0",
            activebackground=ACCENT_MINT,
            activeforeground="#ffffff",
            bd=0,
            padx=10,
            pady=5,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=self._toggle_draw_region_mode,
        )
        self.btn_draw_region.pack(side=tk.RIGHT, padx=3)

        tk.Checkbutton(
            viewer_bar,
            text="📌 Рамки #N",
            variable=self.show_canvas_pins,
            bg=BG_BENTO,
            fg=FG_PRIMARY,
            selectcolor=BG_INNER,
            activebackground=BG_BENTO,
            activeforeground=FG_PRIMARY,
            font=("Segoe UI", 9, "bold"),
            command=self._redraw_preview_canvas,
        ).pack(side=tk.RIGHT, padx=4)

        self.preview_canvas = tk.Canvas(right_card, bg="#0d0b14", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.preview_canvas.bind("<Configure>", self._on_canvas_configure_debounced)
        self.preview_canvas.bind("<ButtonPress-1>", self._on_canvas_mouse_down)
        self.preview_canvas.bind("<B1-Motion>", self._on_canvas_mouse_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_canvas_mouse_up)
        self.preview_canvas.bind("<MouseWheel>", self._on_canvas_mouse_wheel)

        self._set_target_lang(self.target_lang, refresh=False)
        self._set_view_mode("split")
        self._set_overflow_mode(self.var_overflow_mode.get(), rerender=False)
        self._update_summary_chip()

    def _toggle_fullscreen(self) -> None:
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
        self.btn_fullscreen.config(text="🗗 Оконный" if self.is_fullscreen else "⛶ Полный экран")

    def _toggle_left_sidebar(self) -> None:
        self.left_sidebar_collapsed = not self.left_sidebar_collapsed
        if self.left_sidebar_collapsed:
            self.left_col_frame.pack_forget()
            self.btn_toggle_left.config(text="▶ Панель", bg=ACCENT_INDIGO)
        else:
            self.left_col_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12), before=self.preview_canvas.master)
            self.btn_toggle_left.config(text="◀ Панель", bg=BG_INNER)
        self.root.after(50, self._redraw_preview_canvas)

    def _step_page(self, delta: int) -> None:
        if not self.folder_pages or not self.selected_page:
            return
        cur_rel = self.selected_page.get("rel_path")
        idx = next((i for i, p in enumerate(self.folder_pages) if p.get("rel_path") == cur_rel), -1)
        if idx < 0:
            return
        nxt = idx + delta
        if 0 <= nxt < len(self.folder_pages):
            iid = self.folder_pages[nxt]["rel_path"]
            if self.tree_pages.exists(iid):
                self.tree_pages.selection_set(iid)
                self.tree_pages.see(iid)

    def _on_canvas_configure_debounced(self, _event: Any = None) -> None:
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.root.after(35, self._redraw_preview_canvas)

    def _say_miko(self, text: str) -> None:
        self.lbl_miko_speech.config(text=text)

    def _update_summary_chip(self) -> None:
        mn = self.var_min_font.get()
        mx = self.var_max_font.get()
        mode = TEXT_FILTER_MODE_MAP.get(self.var_text_filter_label.get(), "dialogue_only")
        mode_short = {
            "dialogue_only": "Только диалоги (без вздохов/SFX)",
            "bubbles_no_sighs": "Без вздохов и SFX",
            "long_only": "Только длинные фразы",
            "bubbles_only": "Только бабблы",
            "all": "Весь текст",
        }.get(mode, "Только диалоги")
        short_mod = self.active_model_short[:24] + "…" if len(self.active_model_short) > 26 else self.active_model_short
        self.lbl_summary_chip.config(text=f"Шрифт: {mn}–{mx}px · {mode_short} · ИИ: {short_mod}")

    def _set_target_lang(self, lang_code: str, refresh: bool = True) -> None:
        self.target_lang = lang_code
        for code, btn in self.lang_pill_btns.items():
            if code == lang_code:
                btn.config(bg=ACCENT_SAKURA, fg="#ffffff")
            else:
                btn.config(bg=BG_INNER, fg=FG_SECONDARY)
        self._save_gui_settings()
        if refresh and self.current_folder:
            self._refresh_folder_inspection_async()

    def _set_overflow_mode(self, mode: str, rerender: bool = True) -> None:
        if mode not in ("shrink", "overflow", "expand_bubble"):
            mode = "shrink"
        self.var_overflow_mode.set(mode)
        for olbl, ocode in OVERFLOW_MODE_MAP.items():
            if ocode == mode:
                self.var_overflow_label.set(olbl)
                break
        for code, btn in getattr(self, "overflow_mode_btns", {}).items():
            if code == mode:
                btn.config(bg=ACCENT_MINT, fg="#ffffff")
            else:
                btn.config(bg=BG_INNER, fg=FG_SECONDARY)
        self._save_gui_settings()
        self._update_summary_chip()
        if rerender:
            self._apply_model_async()
            if self.selected_page and self.last_blocks:
                self._auto_save_edited_blocks_now()

    def _set_view_mode(self, mode: str) -> None:
        self.view_mode = mode
        for code, btn in self.view_mode_btns.items():
            if code == mode:
                btn.config(bg=ACCENT_INDIGO, fg="#ffffff")
            else:
                btn.config(bg=BG_INNER, fg=FG_SECONDARY)
        self._redraw_preview_canvas()

    def _launch_standalone_web_app(self) -> None:
        url = f"{SERVER_URL}/"
        if os.name == "nt":
            candidates = [
                shutil.which("msedge"),
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                shutil.which("chrome"),
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            ]
            for exe in candidates:
                if exe and os.path.exists(exe):
                    subprocess.Popen([exe, f"--app={url}", "--window-size=1480,900"])
                    return
        webbrowser.open(url)

    def _open_diagnostics_modal(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("🩺 Диагностика, Тайминги и Логи (v1.0.5)")
        win.geometry("880x640")
        win.configure(bg=BG_BENTO)
        win.transient(self.root)

        top_bar = tk.Frame(win, bg=BG_BENTO, padx=16, pady=12)
        top_bar.pack(fill=tk.X)

        lbl_status = tk.Label(
            top_bar,
            text="Лог пишется автоматически в logs/miko_studio.log",
            bg=BG_BENTO,
            fg=FG_SECONDARY,
            font=("Segoe UI", 9, "bold"),
        )
        lbl_status.pack(side=tk.LEFT)

        txt_wrap = tk.Frame(win, bg=BG_DEEP, padx=16, pady=8)
        txt_wrap.pack(fill=tk.BOTH, expand=True)

        txt_report = tk.Text(
            txt_wrap,
            bg="#0b0912",
            fg="#a7f3d0",
            insertbackground="#ffffff",
            font=("Consolas", 9),
            wrap=tk.WORD,
            padx=10,
            pady=10,
        )
        vsb = ttk.Scrollbar(txt_wrap, orient=tk.VERTICAL, command=txt_report.yview)
        txt_report.configure(yscrollcommand=vsb.set)
        txt_report.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        def _refresh_report() -> None:
            def _worker() -> None:
                try:
                    with httpx.Client(trust_env=False, timeout=10.0) as client:
                        resp = client.get(f"{SERVER_URL}/diagnostics/report")
                        resp.raise_for_status()
                        rep_txt = resp.json().get("report_text", "")

                    def _apply() -> None:
                        if win.winfo_exists():
                            txt_report.delete("1.0", tk.END)
                            txt_report.insert("1.0", rep_txt)
                            txt_report.see(tk.END)

                    self.root.after(0, _apply)
                except Exception as exc:
                    if win.winfo_exists():
                        self.root.after(
                            0,
                            lambda: (
                                txt_report.delete("1.0", tk.END),
                                txt_report.insert("1.0", f"Ошибка получения отчета диагностики: {exc}"),
                            ),
                        )

            threading.Thread(target=_worker, daemon=True).start()

        def _copy_to_clipboard() -> None:
            content = txt_report.get("1.0", tk.END).strip()
            if not content:
                return
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.root.update()
            lbl_status.config(text="✅ Отчет скопирован в буфер обмена! Вставьте его в чат (Ctrl+V)", fg=ACCENT_MINT)
            self._say_miko("📋 Диагностический отчет скопирован в буфер обмена! Вставь его в наш чат (Ctrl+V)~ 🩺🌸")

        def _open_logs_dir() -> None:
            def _w() -> None:
                try:
                    with httpx.Client(trust_env=False, timeout=5.0) as client:
                        client.post(f"{SERVER_URL}/diagnostics/open_logs")
                except Exception:
                    pass

            threading.Thread(target=_w, daemon=True).start()

        tk.Button(
            top_bar,
            text="📋 Скопировать отчет для чата (в 1 клик)",
            bg=ACCENT_SAKURA,
            fg="#ffffff",
            bd=0,
            padx=14,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=_copy_to_clipboard,
        ).pack(side=tk.RIGHT, padx=(6, 0))

        tk.Button(
            top_bar,
            text="📂 Открыть папку logs/",
            bg=BG_INNER,
            fg=FG_PRIMARY,
            bd=0,
            padx=10,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=_open_logs_dir,
        ).pack(side=tk.RIGHT, padx=4)

        tk.Button(
            top_bar,
            text="🔄 Обновить",
            bg=BG_INNER,
            fg=FG_PRIMARY,
            bd=0,
            padx=10,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=_refresh_report,
        ).pack(side=tk.RIGHT, padx=4)

        _refresh_report()

    def _open_settings_modal(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("⚙️ Настройки DeepSeek API, фильтрации текста и шрифта")
        win.geometry("560x730")
        win.configure(bg=BG_BENTO)
        win.transient(self.root)
        win.grab_set()

        pad = tk.Frame(win, bg=BG_BENTO, padx=20, pady=16)
        pad.pack(fill=tk.BOTH, expand=True)

        # 1. Автоскачивание Shisa и топ-моделей через Ollama
        tk.Label(
            pad,
            text="🌸 Скачать Shisa V2.1 / Топ-модели в 1 клик (через Ollama, без Unsloth):",
            bg=BG_BENTO,
            fg="#f472b6",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        cb_pull = ttk.Combobox(
            pad,
            textvariable=self.var_recommended_pull,
            values=list(RECOMMENDED_DOWNLOAD_MAP.keys()),
            state="readonly",
        )
        cb_pull.pack(fill=tk.X, pady=(4, 6))

        lbl_pull_status = tk.Label(pad, text="", bg=BG_BENTO, fg=ACCENT_MINT, font=("Segoe UI", 9))

        def _start_pull() -> None:
            tag = RECOMMENDED_DOWNLOAD_MAP.get(self.var_recommended_pull.get(), "hf.co/mradermacher/shisa-v2.1-unphi4-14b-GGUF:Q4_K_M")
            lbl_pull_status.pack(anchor="w", pady=(0, 4))
            lbl_pull_status.config(text=f"⏳ Скачивание {tag} через Ollama...")
            self._say_miko(f"Скачиваю модель «{tag}» через Ollama! Когда скачается — сразу станет дефолтной~ 🌸")

            def _pull_worker() -> None:
                try:
                    with httpx.Client(trust_env=False, timeout=15.0) as client:
                        client.post(f"{SERVER_URL}/pull_model", json={"model": tag, "set_as_default": True})
                    while True:
                        time.sleep(1.3)
                        with httpx.Client(trust_env=False, timeout=8.0) as client:
                            st = client.get(f"{SERVER_URL}/pull_status").json()
                        msg = st.get("message", "Скачивание...")
                        status_code = st.get("status", "idle")
                        self.root.after(0, lambda m=msg: self._say_miko(f"📥 {m}"))
                        if win.winfo_exists():
                            self.root.after(0, lambda m=msg: lbl_pull_status.config(text=f"📥 {m}"))
                        if status_code in ("completed", "error", "idle"):
                            self._ping_server(force_models=True)
                            break
                except Exception as exc:
                    self.root.after(0, lambda: self._say_miko(f"Ошибка скачивания: {exc}"))

            threading.Thread(target=_pull_worker, daemon=True).start()

        tk.Button(
            pad,
            text="📥 Скачать в Ollama и сделать моделью по умолчанию",
            bg="#db2777",
            fg="#ffffff",
            bd=0,
            pady=7,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=_start_pull,
        ).pack(fill=tk.X, pady=(0, 10))

        # 2. Выбор активной модели
        tk.Label(pad, text="🤖 Активная ИИ-модель (Ollama / Cloud API):", bg=BG_BENTO, fg=FG_SECONDARY).pack(anchor="w")
        model_names = ["✨ Автовыбор лучшей модели (auto)"] + [
            str(m.get("name", m.get("id"))) for m in self.available_models
        ]
        cb_mod = ttk.Combobox(pad, textvariable=self.var_model_display, values=model_names, state="readonly")
        cb_mod.pack(fill=tk.X, pady=(3, 10))

        # 3. Пресеты DeepSeek API (только Flash и Pro)
        tk.Label(
            pad,
            text="🐋 Пресеты API DeepSeek (Flash и Pro):",
            bg=BG_BENTO,
            fg="#818cf8",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        cb_preset = ttk.Combobox(
            pad,
            textvariable=self.var_cloud_preset,
            values=list(CLOUD_PRESET_MAP.keys()),
            state="readonly",
        )
        cb_preset.pack(fill=tk.X, pady=(3, 6))

        def _on_preset_selected(_event: Any = None) -> None:
            pair = CLOUD_PRESET_MAP.get(self.var_cloud_preset.get())
            if pair:
                cur_u = self.var_unsloth_url.get().strip()
                if not cur_u or "127.0.0.1" in cur_u or "localhost" in cur_u:
                    self.var_unsloth_url.set(pair[0])
                self.var_custom_model.set(pair[1])

        cb_preset.bind("<<ComboboxSelected>>", _on_preset_selected)

        tk.Label(pad, text="Base URL API DeepSeek:", bg=BG_BENTO, fg=FG_SECONDARY).pack(anchor="w")
        tk.Entry(pad, textvariable=self.var_unsloth_url, bg=BG_INNER, fg="#ffffff", insertbackground="#ffffff", bd=0).pack(
            fill=tk.X, ipady=4, pady=(2, 6)
        )

        tk.Label(pad, text="API Ключ (sk-...):", bg=BG_BENTO, fg=FG_SECONDARY).pack(anchor="w")
        tk.Entry(pad, textvariable=self.var_api_key, show="*", bg=BG_INNER, fg="#ffffff", insertbackground="#ffffff", bd=0).pack(
            fill=tk.X, ipady=4, pady=(2, 8)
        )

        # 4. Объем перевода текста на странице (фильтр вздохов и спецэффектов)
        tk.Label(
            pad,
            text="📊 Объем перевода на странице (защита рисунка от замазки):",
            bg=BG_BENTO,
            fg="#f472b6",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        cb_filter = ttk.Combobox(
            pad,
            textvariable=self.var_text_filter_label,
            values=list(TEXT_FILTER_MODE_MAP.keys()),
            state="readonly",
        )
        cb_filter.pack(fill=tk.X, pady=(3, 4))

        def _on_filter_mode_selected(_event: Any = None) -> None:
            m = TEXT_FILTER_MODE_MAP.get(self.var_text_filter_label.get(), "dialogue_only")
            if m == "dialogue_only":
                self.var_skip_sighs.set(True)
                self.var_skip_sfx.set(True)
                self.var_detect_free_text.set(False)
                self.var_min_text_len.set(2)
            elif m == "bubbles_no_sighs":
                self.var_skip_sighs.set(True)
                self.var_skip_sfx.set(True)
                self.var_detect_free_text.set(True)
                self.var_min_text_len.set(2)
            elif m == "long_only":
                self.var_skip_sighs.set(True)
                self.var_skip_sfx.set(True)
                self.var_detect_free_text.set(False)
                self.var_min_text_len.set(4)
            elif m == "bubbles_only":
                self.var_skip_sighs.set(False)
                self.var_skip_sfx.set(True)
                self.var_detect_free_text.set(False)
                self.var_min_text_len.set(1)
            elif m == "all":
                self.var_skip_sighs.set(False)
                self.var_skip_sfx.set(False)
                self.var_detect_free_text.set(True)
                self.var_min_text_len.set(1)

        cb_filter.bind("<<ComboboxSelected>>", _on_filter_mode_selected)

        tk.Checkbutton(
            pad,
            text="🚫 Не переводить вздохи и охи («а», «о», «ах», «ох», «мм» и т.п.)",
            variable=self.var_skip_sighs,
            bg=BG_BENTO,
            fg=FG_PRIMARY,
            selectcolor=BG_INNER,
            activebackground=BG_BENTO,
            activeforeground=FG_PRIMARY,
        ).pack(anchor="w", pady=1)

        tk.Checkbutton(
            pad,
            text="🚫 Не переводить звуковые спецэффекты (SFX / ономатопею)",
            variable=self.var_skip_sfx,
            bg=BG_BENTO,
            fg=FG_PRIMARY,
            selectcolor=BG_INNER,
            activebackground=BG_BENTO,
            activeforeground=FG_PRIMARY,
        ).pack(anchor="w", pady=1)

        tk.Checkbutton(
            pad,
            text="🔍 Переводить текст ВНЕ бабблов (надписи на фоне поверх рисунка)",
            variable=self.var_detect_free_text,
            bg=BG_BENTO,
            fg=FG_PRIMARY,
            selectcolor=BG_INNER,
            activebackground=BG_BENTO,
            activeforeground=FG_PRIMARY,
        ).pack(anchor="w", pady=(1, 4))

        f_row_len = tk.Frame(pad, bg=BG_BENTO)
        f_row_len.pack(fill=tk.X, pady=2)
        tk.Label(f_row_len, text="Мин. длина фразы (символов):", bg=BG_BENTO, fg=FG_SECONDARY).pack(side=tk.LEFT)
        ttk.Spinbox(f_row_len, from_=1, to=15, textvariable=self.var_min_text_len, width=8).pack(side=tk.RIGHT)

        # 5. Настройки шрифта
        f_row = tk.Frame(pad, bg=BG_BENTO)
        f_row.pack(fill=tk.X, pady=2)
        tk.Label(f_row, text="Минимальный шрифт (px):", bg=BG_BENTO, fg=FG_SECONDARY).pack(side=tk.LEFT)
        ttk.Spinbox(f_row, from_=8, to=42, textvariable=self.var_min_font, width=8).pack(side=tk.RIGHT)

        f_row2 = tk.Frame(pad, bg=BG_BENTO)
        f_row2.pack(fill=tk.X, pady=2)
        tk.Label(f_row2, text="Максимальный шрифт (px):", bg=BG_BENTO, fg=FG_SECONDARY).pack(side=tk.LEFT)
        ttk.Spinbox(f_row2, from_=12, to=72, textvariable=self.var_max_font, width=8).pack(side=tk.RIGHT)

        f_row_ovf = tk.Frame(pad, bg=BG_BENTO)
        f_row_ovf.pack(fill=tk.X, pady=2)
        tk.Label(f_row_ovf, text="Режим узких бабблов:", bg=BG_BENTO, fg=FG_SECONDARY).pack(side=tk.LEFT)
        cb_ovf = ttk.Combobox(
            f_row_ovf,
            textvariable=self.var_overflow_label,
            values=list(OVERFLOW_MODE_MAP.keys()),
            state="readonly",
            width=28,
        )
        cb_ovf.pack(side=tk.RIGHT)

        def _on_ovf_selected(_event: Any = None) -> None:
            m = OVERFLOW_MODE_MAP.get(self.var_overflow_label.get(), "shrink")
            self._set_overflow_mode(m, rerender=False)

        cb_ovf.bind("<<ComboboxSelected>>", _on_ovf_selected)

        tk.Checkbutton(
            pad,
            text="⏩ Продолжать перевод папки с места остановки",
            variable=self.var_skip_existing,
            bg=BG_BENTO,
            fg=FG_PRIMARY,
            selectcolor=BG_INNER,
            activebackground=BG_BENTO,
            activeforeground=FG_PRIMARY,
        ).pack(anchor="w", pady=1)

        tk.Checkbutton(
            pad,
            text="🔠 Заглавные буквы (КОМИКС-СТИЛЬ)",
            variable=self.var_uppercase,
            bg=BG_BENTO,
            fg=FG_PRIMARY,
            selectcolor=BG_INNER,
            activebackground=BG_BENTO,
            activeforeground=FG_PRIMARY,
        ).pack(anchor="w", pady=1)

        tk.Checkbutton(
            pad,
            text="⏱ Автосохранение при перетаскивании рамки или прокрутке колесиком",
            variable=self.var_auto_save_boxes,
            bg=BG_BENTO,
            fg=FG_PRIMARY,
            selectcolor=BG_INNER,
            activebackground=BG_BENTO,
            activeforeground=FG_PRIMARY,
        ).pack(anchor="w", pady=1)

        f_row_as = tk.Frame(pad, bg=BG_BENTO)
        f_row_as.pack(fill=tk.X, pady=2)
        tk.Label(
            f_row_as,
            text="⏳ Интервал автосохранения рамок (сек):",
            bg=BG_BENTO,
            fg=FG_SECONDARY,
        ).pack(side=tk.LEFT)
        ttk.Spinbox(
            f_row_as,
            from_=0.5,
            to=30.0,
            increment=0.5,
            textvariable=self.var_auto_save_sec,
            width=8,
        ).pack(side=tk.RIGHT)

        tk.Checkbutton(
            pad,
            text="💾 Автосохранение манги из расширения Chrome в папку на ПК",
            variable=self.var_auto_save_web,
            bg=BG_BENTO,
            fg=ACCENT_MINT,
            selectcolor=BG_INNER,
            activebackground=BG_BENTO,
            activeforeground=FG_PRIMARY,
        ).pack(anchor="w", pady=(1, 8))

        def _save_and_close() -> None:
            mn = int(self.var_min_font.get())
            mx = int(self.var_max_font.get())
            if mn > mx:
                self.var_max_font.set(mn)
            self._save_gui_settings()
            self._apply_model_async()
            self._update_summary_chip()
            win.destroy()

        tk.Button(
            pad,
            text="✅ Сохранить настройки",
            bg=ACCENT_SAKURA,
            fg="#ffffff",
            bd=0,
            pady=9,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            command=_save_and_close,
        ).pack(fill=tk.X)

    def _show_phrases_popup(self, open_romaji: bool = True) -> None:
        if not self.selected_page:
            messagebox.showinfo("Редактор перевода", "Сначала выберите папку с мангой или откройте страницу слева!")
            return

        win = tk.Toplevel(self.root)
        page_label = self.selected_page.get("rel_path") or self.selected_page.get("filename") or "Страница"
        win.title(f"✏️ Ручное редактирование перевода и Ромадзи — {page_label}")
        win.geometry("880x660")
        win.minsize(720, 480)
        win.configure(bg=BG_DEEP)
        win.transient(self.root)

        show_romaji_var = tk.BooleanVar(value=bool(open_romaji or True))
        status_var = tk.StringVar(
            value="Измените перевод любого блока ниже и нажмите «💾 Сохранить и перерисовать страницу»."
        )
        block_widgets: list[dict[str, Any]] = []

        header = tk.Frame(win, bg=BG_BENTO, padx=16, pady=10)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text=f"✏️ Редактор текста: {page_label}",
            bg=BG_BENTO,
            fg=FG_PRIMARY,
            font=("Segoe UI", 11, "bold"),
        ).pack(side=tk.LEFT)

        btn_bar = tk.Frame(header, bg=BG_BENTO)
        btn_bar.pack(side=tk.RIGHT)

        list_outer = tk.Frame(win, bg=BG_DEEP, padx=14, pady=8)
        list_outer.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(list_outer, bg=BG_DEEP, highlightthickness=0)
        vsb = ttk.Scrollbar(list_outer, orient=tk.VERTICAL, command=canvas.yview)
        cards_frame = tk.Frame(canvas, bg=BG_DEEP)

        cards_frame.bind(
            "<Configure>",
            lambda _: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas_win = canvas.create_window((0, 0), window=cards_frame, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfig(canvas_win, width=e.width),
        )
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        footer = tk.Frame(win, bg=BG_BENTO, padx=16, pady=8)
        footer.pack(fill=tk.X)
        tk.Label(footer, textvariable=status_var, bg=BG_BENTO, fg=FG_SECONDARY, font=("Segoe UI", 9)).pack(
            side=tk.LEFT
        )

        def _collect_edited_blocks() -> list[dict[str, Any]]:
            updated: list[dict[str, Any]] = []
            for item in block_widgets:
                b_copy = dict(item["block"])
                if not b_copy.get("clean_bbox") and b_copy.get("bbox"):
                    b_copy["clean_bbox"] = list(b_copy["bbox"])
                txt_w: tk.Text = item["text_widget"]
                fs_var: tk.StringVar = item["font_var"]
                val_t = txt_w.get("1.0", tk.END).strip()
                b_copy["translated_text"] = val_t
                b_copy["is_deleted"] = not bool(val_t)
                fs_str = fs_var.get().strip()
                if fs_str.isdigit() and int(fs_str) >= 6:
                    b_copy["font_size"] = int(fs_str)
                else:
                    b_copy["font_size"] = None
                updated.append(b_copy)
            return updated

        def _rebuild_cards() -> None:
            for child in cards_frame.winfo_children():
                child.destroy()
            block_widgets.clear()

            if not self.last_blocks:
                tk.Label(
                    cards_frame,
                    text="На этой странице пока нет загруженных текстовых блоков. Нажмите «🔍 OCR заново» вверху окна.",
                    bg=BG_DEEP,
                    fg=FG_MUTED,
                    font=("Segoe UI", 10),
                    pady=30,
                ).pack(fill=tk.X)
                return

            show_rom = show_romaji_var.get()
            font_choices = ["Авто"] + [str(x) for x in (10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 42)]

            for b in self.last_blocks:
                card = tk.Frame(
                    cards_frame,
                    bg=BG_BENTO,
                    padx=12,
                    pady=10,
                    highlightbackground="#2e294e",
                    highlightthickness=1,
                )
                card.pack(fill=tk.X, pady=5)

                top_row = tk.Frame(card, bg=BG_BENTO)
                top_row.pack(fill=tk.X, pady=(0, 4))

                b_id = b.get("id", 1)
                is_free = bool(b.get("is_free_text", False))
                tag_lbl = "📝 Надпись вне баббла" if is_free else "💬 Баббл"
                tk.Label(
                    top_row,
                    text=f"#{b_id} · {tag_lbl}",
                    bg=BG_INNER,
                    fg="#f472b6",
                    font=("Segoe UI", 9, "bold"),
                    padx=8,
                    pady=2,
                ).pack(side=tk.LEFT)

                fs_init = str(b.get("font_size")) if b.get("font_size") else "Авто"
                fs_var = tk.StringVar(value=fs_init)
                cb_fs = ttk.Combobox(top_row, textvariable=fs_var, values=font_choices, state="readonly", width=8)
                cb_fs.pack(side=tk.RIGHT, padx=(4, 0))
                tk.Label(top_row, text="Шрифт (px):", bg=BG_BENTO, fg=FG_MUTED, font=("Segoe UI", 8)).pack(
                    side=tk.RIGHT
                )

                orig_txt = str(b.get("original_text") or "").strip()
                rom_txt = str(b.get("romaji_text") or "").strip()
                trans_txt = str(b.get("translated_text") or "").strip()

                orig_row = tk.Frame(card, bg="#141122", padx=8, pady=5)
                orig_row.pack(fill=tk.X, pady=2)
                tk.Label(
                    orig_row,
                    text=f"🇯🇵 Оригинал:  {orig_txt}",
                    bg="#141122",
                    fg=FG_PRIMARY,
                    font=("Segoe UI", 9),
                    anchor="w",
                    justify=tk.LEFT,
                    wraplength=780,
                ).pack(fill=tk.X)

                txt_edit = tk.Text(
                    card,
                    height=2,
                    bg="#110e1c",
                    fg="#ffffff",
                    insertbackground="#ffffff",
                    font=("Segoe UI", 10, "bold"),
                    bd=1,
                    relief=tk.FLAT,
                    padx=8,
                    pady=5,
                    wrap=tk.WORD,
                )
                txt_edit.insert("1.0", trans_txt)

                if show_rom:
                    rom_row = tk.Frame(card, bg="#1e1b4b", padx=8, pady=5)
                    rom_row.pack(fill=tk.X, pady=2)
                    tk.Label(
                        rom_row,
                        text=f"🔤 Ромадзи:  {rom_txt or '—'}",
                        bg="#1e1b4b",
                        fg="#c7d2fe",
                        font=("Segoe UI", 9, "bold"),
                        anchor="w",
                        justify=tk.LEFT,
                        wraplength=670,
                    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

                    def _copy_rom_to_trans(t_widget: tk.Text = txt_edit, r_val: str = rom_txt) -> None:
                        if not r_val:
                            return
                        t_widget.delete("1.0", tk.END)
                        t_widget.insert("1.0", r_val)

                    tk.Button(
                        rom_row,
                        text="➔ В перевод",
                        bg=BG_INNER,
                        fg=FG_PRIMARY,
                        bd=0,
                        padx=8,
                        pady=2,
                        font=("Segoe UI", 8),
                        cursor="hand2",
                        command=_copy_rom_to_trans,
                    ).pack(side=tk.RIGHT)

                trans_lbl_row = tk.Frame(card, bg=BG_BENTO)
                trans_lbl_row.pack(fill=tk.X, pady=(4, 2))
                tk.Label(
                    trans_lbl_row,
                    text="✏️ Перевод (оставьте пустым или нажмите «Удалить перевод», чтобы сохранить оригинал на картинке):",
                    bg=BG_BENTO,
                    fg=FG_SECONDARY,
                    font=("Segoe UI", 8, "bold"),
                    anchor="w",
                ).pack(side=tk.LEFT)

                def _remove_frame_completely(blk_ref: dict[str, Any] = b) -> None:
                    if block_widgets:
                        self.last_blocks = _collect_edited_blocks()
                    target_id = blk_ref.get("id")
                    self.last_blocks = [x for x in self.last_blocks if x.get("id") != target_id]
                    _rebuild_cards()
                    _save_edited_page()

                def _clear_single_block_text(t_w: tk.Text = txt_edit, blk_ref: dict[str, Any] = b) -> None:
                    cur_v = t_w.get("1.0", tk.END).strip()
                    if cur_v:
                        blk_ref["_backup_translation"] = cur_v
                    t_w.delete("1.0", tk.END)
                    blk_ref["translated_text"] = ""
                    blk_ref["is_deleted"] = True
                    _save_edited_page()

                tk.Button(
                    trans_lbl_row,
                    text="🗑 Удалить рамку",
                    bg=ACCENT_ROSE,
                    fg="#ffffff",
                    bd=0,
                    padx=9,
                    pady=2,
                    font=("Segoe UI", 8, "bold"),
                    cursor="hand2",
                    command=_remove_frame_completely,
                ).pack(side=tk.RIGHT, padx=(4, 0))

                tk.Button(
                    trans_lbl_row,
                    text="🧹 Очистить текст",
                    bg=BG_INNER,
                    fg=FG_PRIMARY,
                    bd=0,
                    padx=8,
                    pady=2,
                    font=("Segoe UI", 8, "bold"),
                    cursor="hand2",
                    command=_clear_single_block_text,
                ).pack(side=tk.RIGHT)

                txt_edit.pack(fill=tk.X)

                block_widgets.append({"block": b, "text_widget": txt_edit, "font_var": fs_var})

        def _toggle_romaji() -> None:
            if block_widgets:
                self.last_blocks = _collect_edited_blocks()
            show_romaji_var.set(not show_romaji_var.get())
            btn_romaji_toggle.config(
                text="🔤 Ромадзи: ВКЛ" if show_romaji_var.get() else "🔤 Ромадзи: ВЫКЛ",
                bg=ACCENT_INDIGO if show_romaji_var.get() else BG_INNER,
            )
            _rebuild_cards()

        def _fetch_blocks_worker(force_rescan: bool = False) -> None:
            if not self.selected_page:
                return
            status_var.set("🔍 Загрузка текстовых блоков страницы и генерация Ромадзи...")
            payload = {
                "source_path": self.selected_page["source_path"],
                "source_dir": str(self.current_folder) if self.current_folder else None,
                "dest_path": self.selected_page.get("dest_path") or None,
                "lang": SOURCE_LANG_MAP.get(self.var_source_lang.get(), "auto"),
                "target_lang": self.target_lang,
                "detect_free_text": bool(self.var_detect_free_text.get()),
                "force_rescan": force_rescan,
                "use_llm_romaji": False,
                "ollama_model": self._get_selected_model_id(),
            }

            def _work() -> None:
                try:
                    with httpx.Client(trust_env=False, timeout=180.0) as client:
                        resp = client.post(f"{SERVER_URL}/page_blocks", json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                    blocks = data.get("blocks", [])

                    def _apply() -> None:
                        self.last_blocks = blocks
                        if self.selected_page is not None:
                            self.selected_page["blocks"] = blocks
                            if data.get("dest_path") and not self.selected_page.get("dest_path"):
                                self.selected_page["dest_path"] = data["dest_path"]
                        self.btn_show_phrases.config(text=f"✏️ Редактор текста ({len(blocks)})")
                        if win.winfo_exists():
                            status_var.set(f"✅ Загружено блоков: {len(blocks)}. Подсказки Ромадзи готовы!")
                            _rebuild_cards()

                    self.root.after(0, _apply)
                except Exception as exc:
                    if win.winfo_exists():
                        self.root.after(0, lambda: status_var.set(f"Ошибка чтения блоков: {exc}"))

            threading.Thread(target=_work, daemon=True).start()

        def _refine_romaji_via_ai() -> None:
            if block_widgets:
                self.last_blocks = _collect_edited_blocks()
            if not self.last_blocks:
                return
            status_var.set("⏳ Запрашиваю у ИИ точное контекстное чтение всех кандзи в Ромадзи...")
            payload = {
                "source_path": self.selected_page["source_path"] if self.selected_page else None,
                "source_dir": str(self.current_folder) if self.current_folder else None,
                "target_lang": self.target_lang,
                "use_llm": True,
                "ollama_model": self._get_selected_model_id(),
                "blocks": self.last_blocks,
            }

            def _work() -> None:
                try:
                    with httpx.Client(trust_env=False, timeout=120.0) as client:
                        resp = client.post(f"{SERVER_URL}/page_romaji", json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                    new_blocks = data.get("blocks", [])
                    rom_map = {b.get("id"): b.get("romaji_text") for b in new_blocks}

                    def _apply() -> None:
                        for b in self.last_blocks:
                            if rom_map.get(b.get("id")):
                                b["romaji_text"] = rom_map[b.get("id")]
                        show_romaji_var.set(True)
                        if win.winfo_exists():
                            btn_romaji_toggle.config(text="🔤 Ромадзи: ВКЛ", bg=ACCENT_INDIGO)
                            status_var.set("✨ Чтение кандзи в Ромадзи успешно уточнено через ИИ!")
                            _rebuild_cards()

                    self.root.after(0, _apply)
                except Exception as exc:
                    if win.winfo_exists():
                        self.root.after(0, lambda: status_var.set(f"Не удалось уточнить через ИИ ({exc}), показан локальный Ромадзи."))

            threading.Thread(target=_work, daemon=True).start()

        def _save_edited_page() -> None:
            if not self.selected_page:
                return
            edited_blocks = _collect_edited_blocks()
            self.last_blocks = edited_blocks
            if edited_blocks is None:
                return

            status_var.set("⏳ Перерисовываю текст на странице и сохраняю файл...")
            btn_save.config(state=tk.DISABLED)
            payload = {
                "source_path": self.selected_page["source_path"],
                "dest_path": self.selected_page.get("dest_path") or None,
                "source_dir": str(self.current_folder) if self.current_folder else None,
                "target_lang": self.target_lang,
                "uppercase": bool(self.var_uppercase.get()),
            "overflow_mode": self.var_overflow_mode.get(),
                "min_font_size": int(self.var_min_font.get()),
                "max_font_size": int(self.var_max_font.get()),
                "overflow_mode": self.var_overflow_mode.get(),
                "return_image": True,
                "image_format": "webp",
                "blocks": edited_blocks,
            }

            def _work() -> None:
                try:
                    with httpx.Client(trust_env=False, timeout=120.0) as client:
                        resp = client.post(f"{SERVER_URL}/save_edited_page", json=payload)
                        resp.raise_for_status()
                        data = resp.json()

                    b64_img = data.get("translated_image")
                    trans_pil: Image.Image | None = None
                    if b64_img:
                        raw = b64_img.split(",", 1)[1] if "," in b64_img else b64_img
                        trans_pil = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")

                    def _done() -> None:
                        if trans_pil is not None:
                            self.preview_trans_pil = trans_pil
                        self.last_blocks = data.get("blocks", edited_blocks)
                        if self.selected_page is not None:
                            self.selected_page["blocks"] = self.last_blocks
                            self.selected_page["is_translated"] = True
                            if data.get("saved_output_path"):
                                self.selected_page["dest_path"] = data["saved_output_path"]
                        self.btn_show_phrases.config(text=f"✏️ Редактор текста ({len(self.last_blocks)})")
                        self._redraw_preview_canvas()
                        self._say_miko(
                            f"Сохранила твой отредактированный перевод на странице «{page_label}» за {data.get('processing_time_sec')}с! 💾🌸"
                        )
                        if self.current_folder:
                            self._refresh_folder_inspection_async()
                        if win.winfo_exists():
                            status_var.set(f"✅ Сохранено в файл: {data.get('saved_output_path')}")
                            btn_save.config(state=tk.NORMAL)

                    self.root.after(0, _done)
                except Exception as exc:
                    if win.winfo_exists():
                        self.root.after(0, lambda: status_var.set(f"Ошибка сохранения: {exc}"))
                        self.root.after(0, lambda: btn_save.config(state=tk.NORMAL))

            threading.Thread(target=_work, daemon=True).start()

        btn_save = tk.Button(
            btn_bar,
            text="💾 Сохранить и перерисовать страницу",
            bg=ACCENT_MINT,
            fg="#ffffff",
            bd=0,
            padx=14,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=_save_edited_page,
        )
        btn_save.pack(side=tk.RIGHT, padx=(6, 0))

        tk.Button(
            btn_bar,
            text="🔍 OCR заново",
            bg=BG_INNER,
            fg=FG_SECONDARY,
            bd=0,
            padx=10,
            pady=6,
            font=("Segoe UI", 9),
            cursor="hand2",
            command=lambda: _fetch_blocks_worker(force_rescan=True),
        ).pack(side=tk.RIGHT, padx=4)

        tk.Button(
            btn_bar,
            text="✨ Уточнить Ромадзи (ИИ)",
            bg=BG_INNER,
            fg="#c7d2fe",
            bd=0,
            padx=10,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=_refine_romaji_via_ai,
        ).pack(side=tk.RIGHT, padx=4)

        btn_romaji_toggle = tk.Button(
            btn_bar,
            text="🔤 Ромадзи: ВКЛ",
            bg=ACCENT_INDIGO,
            fg="#ffffff",
            bd=0,
            padx=10,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            command=_toggle_romaji,
        )
        btn_romaji_toggle.pack(side=tk.RIGHT, padx=4)

        _rebuild_cards()
        if not self.last_blocks:
            _fetch_blocks_worker(force_rescan=False)

    def _initial_server_check(self) -> None:
        def _worker() -> None:
            if not self._ping_server(force_models=False):
                print("[GUI] Сервер на 127.0.0.1:5000 еще не запущен — запускаю server.py...", flush=True)
                self.root.after(0, lambda: self._say_miko("Запускаю локальный ИИ-сервер... Подожди 5–10 секунд~ 🚀"))
                self._spawn_server_process()
                for attempt in range(1, 25):
                    time.sleep(1.5)
                    if self._ping_server(force_models=True):
                        print("[GUI] Сервер успешно запущен и отвечает!", flush=True)
                        break
                    print(f"[GUI] Ожидание загрузки моделей сервера ({attempt}/24)...", flush=True)
            else:
                print("[GUI] Сервер уже активен на 127.0.0.1:5000!", flush=True)
                if self.current_folder:
                    self._refresh_folder_inspection_async()

        threading.Thread(target=_worker, daemon=True).start()

    def _spawn_server_process(self) -> None:
        if self.server_process is not None and self.server_process.poll() is None:
            return
        server_script = BASE_DIR / "server.py"
        if not server_script.exists():
            return
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            self.server_process = subprocess.Popen(
                [sys.executable, str(server_script)],
                cwd=str(BASE_DIR),
                env=env,
            )
        except Exception as exc:
            self.root.after(0, lambda: self._say_miko(f"Не удалось запустить server.py: {exc}"))

    def _ping_server(self, force_models: bool = False) -> bool:
        try:
            with httpx.Client(trust_env=False, timeout=8.0) as client:
                resp = client.get(f"{SERVER_URL}/health")
                resp.raise_for_status()
                data = resp.json()
                running_ver = str(data.get("version") or "")
                if running_ver != "1.0.5":
                    print(f"[GUI] На порту 5000 запущена старая версия сервера (v{running_ver}). Перезапускаю на v1.0.5...", flush=True)
                    try:
                        client.post(f"{SERVER_URL}/shutdown", timeout=2.0)
                    except Exception:
                        pass
                    time.sleep(1.5)
                    return False
                self.server_online = True
                self.root.after(0, lambda: self._on_server_healthy(data))
                return True
        except Exception:
            self.server_online = False
            self.root.after(0, lambda: self.lbl_server_dot.config(text="🔴 Оффлайн", fg=ACCENT_ROSE))
            return False

    def _on_server_healthy(self, data: dict[str, Any]) -> None:
        gpu_name = data.get("gpu", {}).get("device_name", "CPU")
        act = str(data.get("active_model") or "auto").split("::")[-1]
        self.active_model_short = act
        self.lbl_server_dot.config(text=f"🟢 {gpu_name}", fg=ACCENT_MINT)
        self.available_models = data.get("models", [])
        if data.get("unsloth_url"):
            self.var_unsloth_url.set(str(data["unsloth_url"]))
        if data.get("unsloth_api_key"):
            self.var_api_key.set(str(data["unsloth_api_key"]))
        self._update_summary_chip()
        if self.current_folder and not self.folder_pages:
            self._refresh_folder_inspection_async()

    def _get_selected_model_id(self) -> str:
        custom = self.var_custom_model.get().strip()
        if custom:
            if "::" not in custom:
                u_url = self.var_unsloth_url.get().strip()
                is_rem = not ("127.0.0.1" in u_url or "localhost" in u_url)
                return f"unsloth::{custom}" if (is_rem or self.var_api_key.get().strip()) else f"ollama::{custom}"
            return custom
        disp = self.var_model_display.get()
        if disp.startswith("✨"):
            return "auto"
        for m in self.available_models:
            if m.get("name") == disp or m.get("id") == disp:
                return str(m["id"])
        return "auto"

    def _apply_model_async(self) -> None:
        mod_id = self._get_selected_model_id()
        u_url = self.var_unsloth_url.get().strip()
        api_key = self.var_api_key.get().strip()

        def _worker() -> None:
            try:
                with httpx.Client(trust_env=False, timeout=12.0) as client:
                    client.post(
                        f"{SERVER_URL}/settings",
                        json={
                            "selected_model": mod_id,
                            "unsloth_url": u_url,
                            "unsloth_api_key": api_key,
                            "auto_save_web_pages": bool(self.var_auto_save_web.get()),
                            "web_save_dir": self.var_web_save_dir.get().strip(),
                            "text_filter_mode": TEXT_FILTER_MODE_MAP.get(self.var_text_filter_label.get(), "dialogue_only"),
                            "skip_sighs": bool(self.var_skip_sighs.get()),
                            "skip_sfx": bool(self.var_skip_sfx.get()),
                            "detect_free_text": bool(self.var_detect_free_text.get()),
                            "min_text_length": int(self.var_min_text_len.get()),
                        },
                    )
                self._ping_server(force_models=True)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _choose_folder_dialog(self) -> None:
        initial = str(self.current_folder) if self.current_folder else str(BASE_DIR)
        selected = filedialog.askdirectory(title="Выберите папку с мангой", initialdir=initial)
        if not selected:
            return
        self.current_folder = Path(selected).resolve()
        self._save_gui_settings()
        self._refresh_folder_inspection_async()

    def _refresh_folder_inspection_async(self) -> None:
        if not self.current_folder:
            return
        folder_str = str(self.current_folder)
        t_lang = self.target_lang

        def _worker() -> None:
            try:
                with httpx.Client(trust_env=False, timeout=15.0) as client:
                    resp = client.post(
                        f"{SERVER_URL}/folder_inspect",
                        json={"folder_path": folder_str, "target_lang": t_lang},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    self.root.after(0, lambda: self._populate_folder_ui(data))
            except Exception as exc:
                self.root.after(0, lambda: self._say_miko(f"Ошибка чтения папки: {exc}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _populate_folder_ui(self, data: dict[str, Any]) -> None:
        self.current_output_dir = Path(data["output_dir"]) if data.get("output_dir") else None
        self.folder_pages = data.get("pages", [])
        total = data.get("total_pages", 0)
        done = data.get("translated_pages", 0)
        rem = data.get("remaining_pages", 0)
        folder_name = Path(data.get("source_dir", "")).name

        self.lbl_folder_name.config(text=f"📁 {folder_name} ➔ _{self.target_lang.upper()}")
        self.lbl_folder_counts.config(text=f"{done} / {total} готово")
        pct = (done / total * 100.0) if total > 0 else 0.0
        self.progress_var.set(pct)

        if not self.is_folder_job_running:
            if done > 0 and rem > 0:
                self.btn_primary_folder.config(
                    text=f"▶ Продолжить перевод папки (осталось {rem} стр.)",
                    bg=ACCENT_SAKURA,
                )
                self._say_miko(
                    f"В папке «{folder_name}» уже переведено {done} из {total} стр.! "
                    f"Нажми кнопку ниже, чтобы продолжить с места остановки~ 🌸"
                )
            elif total > 0 and rem == 0:
                self.btn_primary_folder.config(text=f"✅ Вся папка переведена ({total} стр.)", bg=ACCENT_MINT)
                self._say_miko(
                    f"Ура! Все {total} страниц в этой папке уже переведены! "
                    f"Выбери любую страницу в списке, если хочешь перевести её заново ✨"
                )
            else:
                self.btn_primary_folder.config(text=f"▶ Перевести всю папку ({total} стр.)", bg=ACCENT_SAKURA)
                self._say_miko(f"Открыта папка «{folder_name}» ({total} стр.). Готова к переводу!")

        prev_sel = self.tree_pages.selection()
        prev_iid = prev_sel[0] if prev_sel else None
        self.tree_pages.delete(*self.tree_pages.get_children())

        for p in self.folder_pages:
            iid = p["rel_path"]
            st_txt = "✅ Готово" if p.get("is_translated") else "⏳ Ждет"
            self.tree_pages.insert("", tk.END, iid=iid, values=(f"#{p['index']}  {p['rel_path']}", st_txt))

        if prev_iid and self.tree_pages.exists(prev_iid):
            self.tree_pages.selection_set(prev_iid)
        elif self.folder_pages:
            self.tree_pages.selection_set(self.folder_pages[0]["rel_path"])

    def _on_tree_select(self, _event: Any = None) -> None:
        sel = self.tree_pages.selection()
        if not sel:
            return
        rel = sel[0]
        page = next((p for p in self.folder_pages if p["rel_path"] == rel), None)
        if not page:
            return
        self.selected_page = page
        saved_b = page.get("blocks")
        self.last_blocks = [dict(b) for b in saved_b] if isinstance(saved_b, list) else []
        self.btn_show_phrases.config(text=f"✏️ Редактор текста ({len(self.last_blocks)})")
        self.btn_edit_page.config(text=f"✏️ Редактировать стр. #{page['index']} (+ Ромадзи)")
        if page.get("is_translated"):
            self.btn_context_page.config(text=f"🔁 Перевести стр. #{page['index']} заново (ИИ)")
        else:
            self.btn_context_page.config(text=f"🎯 Перевести только стр. #{page['index']} ({page['filename']})")
        self._load_preview_from_disk(page)

    def _load_preview_from_disk(self, page: dict[str, Any]) -> None:
        src_p = Path(page["source_path"])
        dst_p = Path(page["dest_path"]) if page.get("dest_path") else None

        try:
            self.preview_orig_pil = Image.open(src_p).convert("RGB").copy() if src_p.exists() else None
        except Exception:
            self.preview_orig_pil = None

        try:
            self.preview_trans_pil = (
                Image.open(dst_p).convert("RGB").copy()
                if (dst_p and dst_p.exists() and dst_p.stat().st_size > 128)
                else None
            )
        except Exception:
            self.preview_trans_pil = None

        self._redraw_preview_canvas()

    def _redraw_preview_canvas(self) -> None:
        self.preview_canvas.delete("all")
        cw = max(100, self.preview_canvas.winfo_width())
        ch = max(100, self.preview_canvas.winfo_height())

        orig = self.preview_orig_pil
        trans = self.preview_trans_pil

        if orig is None and trans is None:
            self.preview_canvas.create_text(
                cw // 2,
                ch // 2,
                text="📖 Выберите папку манги или отдельную страницу,\nчтобы увидеть сравнение До и После",
                fill=FG_MUTED,
                font=("Segoe UI", 11),
                justify=tk.CENTER,
            )
            return

        if self.view_mode == "original" or (self.view_mode == "split" and trans is None):
            img_show = orig
        elif self.view_mode == "translated":
            img_show = trans if trans is not None else orig
        else:
            if orig is not None and trans is not None:
                gap = 12
                w1, h1 = orig.size
                w2, h2 = trans.size
                comb = Image.new("RGB", (w1 + w2 + gap, max(h1, h2)), (13, 11, 20))
                comb.paste(orig, (0, 0))
                comb.paste(trans, (w1 + gap, 0))
                img_show = comb
            else:
                img_show = trans or orig

        if img_show is None:
            return

        iw, ih = img_show.size
        scale = min((cw - 20) / max(1, iw), (ch - 20) / max(1, ih))
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))

        img_x0 = (cw - nw) // 2
        img_y0 = (ch - nh) // 2
        is_split_both = bool(self.view_mode == "split" and orig is not None and trans is not None)
        self._last_canvas_render_info = {
            "img_x0": img_x0,
            "img_y0": img_y0,
            "scale": scale,
            "is_split_both": is_split_both,
            "orig_w": orig.size[0] if orig is not None else (trans.size[0] if trans is not None else 1),
            "orig_h": orig.size[1] if orig is not None else (trans.size[1] if trans is not None else 1),
            "gap": 12 if is_split_both else 0,
        }

        resized = img_show.resize((nw, nh), Image.Resampling.LANCZOS)
        self.preview_photo_ref = ImageTk.PhotoImage(resized)
        self.preview_canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self.preview_photo_ref)
        self._draw_interactive_bbox_pins()

    def _redraw_pins_only(self) -> None:
        """Быстрая перерисовка только контуров рамок на холсте без тяжелого ресайза картинки (<0.3 мс)."""
        self.preview_canvas.delete("bbox_pin_overlay")
        self._draw_interactive_bbox_pins()

    def _draw_interactive_bbox_pins(self) -> None:
        if not self.show_canvas_pins.get() or not self.last_blocks:
            return
        info = self._last_canvas_render_info
        if not info:
            return
        scale = float(info.get("scale") or 1.0)
        img_x0 = int(info.get("img_x0") or 0)
        img_y0 = int(info.get("img_y0") or 0)
        orig_w = int(info.get("orig_w") or 1)
        is_split = bool(info.get("is_split_both"))
        gap = int(info.get("gap") or 0)
        offset_x = (orig_w + gap) if is_split else 0
        auto_off = not bool(self.var_auto_save_boxes.get())

        for b in self.last_blocks:
            bbox = b.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            if not b.get("clean_bbox"):
                b["clean_bbox"] = list(bbox)
            x1, y1, x2, y2 = bbox[:4]
            sx1 = img_x0 + int((x1 + offset_x) * scale)
            sy1 = img_y0 + int(y1 * scale)
            sx2 = img_x0 + int((x2 + offset_x) * scale)
            sy2 = img_y0 + int(y2 * scale)
            is_del = bool(b.get("is_deleted")) or not str(b.get("translated_text") or "").strip()
            color = "#fbbf24" if is_del else ("#818cf8" if b.get("bbox_modified") else "#ec4899")
            self.preview_canvas.create_rectangle(
                sx1, sy1, sx2, sy2,
                outline=color,
                width=2,
                dash=(4, 2) if is_del else (),
                tags=("bbox_pin_overlay",),
            )
            fs_lbl = f"{b['font_size']}px" if b.get("font_size") else "Авто"
            self.preview_canvas.create_text(
                sx1 + 4,
                max(8, sy1 - 9),
                text=f"#{b.get('id', 1)} ({fs_lbl})",
                fill="#ffffff",
                anchor="w",
                font=("Segoe UI", 8, "bold"),
                tags=("bbox_pin_overlay",),
            )
            # Кнопка полного удаления рамки [🗑] в правом верхнем углу
            del_x1, del_y1 = sx2 - 16, max(2, sy1 - 16)
            del_x2, del_y2 = sx2, max(16, sy1)
            self.preview_canvas.create_rectangle(
                del_x1, del_y1, del_x2, del_y2,
                fill="#e11d48", outline="#ffffff", width=1,
                tags=("bbox_pin_overlay",),
            )
            self.preview_canvas.create_text(
                (del_x1 + del_x2) // 2,
                (del_y1 + del_y2) // 2,
                text="×",
                fill="#ffffff",
                font=("Segoe UI", 9, "bold"),
                tags=("bbox_pin_overlay",),
            )
            # Значок сохранения [💾] на рамке, если автосохранение выключено или есть несохраненные изменения
            if auto_off or b.get("_dirty"):
                sav_x1, sav_y1 = sx2 - 35, max(2, sy1 - 16)
                sav_x2, sav_y2 = sx2 - 18, max(16, sy1)
                self.preview_canvas.create_rectangle(
                    sav_x1, sav_y1, sav_x2, sav_y2,
                    fill="#059669", outline="#6ee7b7", width=1,
                    tags=("bbox_pin_overlay",),
                )
                self.preview_canvas.create_text(
                    (sav_x1 + sav_x2) // 2,
                    (sav_y1 + sav_y2) // 2,
                    text="💾",
                    fill="#ffffff",
                    font=("Segoe UI", 7, "bold"),
                    tags=("bbox_pin_overlay",),
                )
            # Угловой маркер ресайза (правый нижний угол)
            self.preview_canvas.create_rectangle(
                sx2 - 5, sy2 - 5, sx2 + 5, sy2 + 5,
                fill="#10b981", outline="#0f0d18", width=1,
                tags=("bbox_pin_overlay",),
            )

    def _hit_test_bbox_pin(self, cx: int, cy: int) -> tuple[dict[str, Any] | None, str]:
        if not self.show_canvas_pins.get() or not self.last_blocks:
            return None, ""
        info = self._last_canvas_render_info
        if not info:
            return None, ""
        scale = float(info.get("scale") or 1.0)
        img_x0 = int(info.get("img_x0") or 0)
        img_y0 = int(info.get("img_y0") or 0)
        orig_w = int(info.get("orig_w") or 1)
        is_split = bool(info.get("is_split_both"))
        gap = int(info.get("gap") or 0)
        offset_x = (orig_w + gap) if is_split else 0
        auto_off = not bool(self.var_auto_save_boxes.get())

        for b in reversed(self.last_blocks):
            bbox = b.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox[:4]
            sx1 = img_x0 + int((x1 + offset_x) * scale)
            sy1 = img_y0 + int(y1 * scale)
            sx2 = img_x0 + int((x2 + offset_x) * scale)
            sy2 = img_y0 + int(y2 * scale)

            del_x1, del_y1 = sx2 - 16, max(2, sy1 - 16)
            del_x2, del_y2 = sx2, max(16, sy1)
            if del_x1 <= cx <= del_x2 and del_y1 <= cy <= del_y2:
                return b, "delete_pin"

            if auto_off or b.get("_dirty"):
                sav_x1, sav_y1 = sx2 - 35, max(2, sy1 - 16)
                sav_x2, sav_y2 = sx2 - 18, max(16, sy1)
                if sav_x1 <= cx <= sav_x2 and sav_y1 <= cy <= sav_y2:
                    return b, "save_pin"

            if abs(cx - sx2) <= 9 and abs(cy - sy2) <= 9:
                return b, "resize_se"
            if abs(cx - sx1) <= 8 and abs(cy - sy1) <= 8:
                return b, "resize_nw"
            if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                return b, "move"
        return None, ""

    def _schedule_idle_autosave(self, delay_ms: int | None = None) -> None:
        if not bool(self.var_auto_save_boxes.get()):
            return
        if self._idle_autosave_job is not None:
            try:
                self.root.after_cancel(self._idle_autosave_job)
            except Exception:
                pass
        if delay_ms is None:
            try:
                sec = max(0.5, min(30.0, float(self.var_auto_save_sec.get())))
            except Exception:
                sec = 2.5
            delay_ms = int(round(sec * 1000))
        self._idle_autosave_job = self.root.after(delay_ms, self._auto_save_edited_blocks_now)

    def _auto_save_edited_blocks_now(self) -> None:
        self._idle_autosave_job = None
        if not self.selected_page or self.last_blocks is None:
            return
        page = self.selected_page
        for b in self.last_blocks:
            b.pop("_dirty", None)
        payload = {
            "source_path": page["source_path"],
            "dest_path": page.get("dest_path") or None,
            "source_dir": str(self.current_folder) if self.current_folder else None,
            "target_lang": self.target_lang,
            "uppercase": bool(self.var_uppercase.get()),
            "overflow_mode": self.var_overflow_mode.get(),
            "min_font_size": int(self.var_min_font.get()),
            "max_font_size": int(self.var_max_font.get()),
            "overflow_mode": self.var_overflow_mode.get(),
            "return_image": True,
            "image_format": "webp",
            "blocks": self.last_blocks,
        }

        def _worker() -> None:
            try:
                with httpx.Client(trust_env=False, timeout=120.0) as client:
                    resp = client.post(f"{SERVER_URL}/save_edited_page", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                b64_img = data.get("translated_image")
                trans_pil: Image.Image | None = None
                if b64_img:
                    raw = b64_img.split(",", 1)[1] if "," in b64_img else b64_img
                    trans_pil = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")

                def _done() -> None:
                    if trans_pil is not None:
                        self.preview_trans_pil = trans_pil
                    self.last_blocks = data.get("blocks", self.last_blocks)
                    page["blocks"] = self.last_blocks
                    page["is_translated"] = True
                    if data.get("saved_output_path"):
                        page["dest_path"] = data["saved_output_path"]
                    self._redraw_preview_canvas()
                    self._say_miko(f"Рамки и текст перерисованы и сохранены ({data.get('processing_time_sec')}с)! 💾🌸")

                self.root.after(0, _done)
            except Exception as exc:
                self.root.after(0, lambda: self._say_miko(f"Ошибка сохранения рамок: {exc}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_canvas_mouse_wheel(self, event: tk.Event[Any]) -> None:
        blk, _ = self._hit_test_bbox_pin(int(event.x), int(event.y))
        if blk is None:
            return
        cur_fs = int(blk.get("font_size") or blk.get("rendered_font_size") or 20)
        step = 2 if event.delta > 0 else -2
        next_fs = max(8, min(140, cur_fs + step))
        blk["font_size"] = next_fs
        blk["is_deleted"] = False
        blk["_dirty"] = True
        self._redraw_pins_only()
        if bool(self.var_auto_save_boxes.get()):
            self._say_miko(f"Шрифт рамки #{blk.get('id')}: {next_fs}px! Авто-обновление через {self.var_auto_save_sec.get()} сек~ 🔤✨")
            self._schedule_idle_autosave()
        else:
            self._say_miko(f"Шрифт рамки #{blk.get('id')}: {next_fs}px! Нажмите 💾 на рамке для сохранения~ 🔤💾")

    def _toggle_draw_region_mode(self) -> None:
        if not self.selected_page or self.preview_orig_pil is None:
            self._say_miko("Сначала открой папку с мангой или выбери страницу слева~ 🌸")
            return
        self.is_draw_region_mode = not self.is_draw_region_mode
        if self.is_draw_region_mode:
            self.btn_draw_region.config(text="🎯 Выделите область на картинке", bg=ACCENT_MINT, fg="#ffffff")
            self.preview_canvas.config(cursor="crosshair")
            self._say_miko("Зажми левую кнопку мыши и обведи рамкой любой кружок или текст прямо на странице! 🎯✨")
        else:
            self.btn_draw_region.config(text="➕ Обвести текст мышкой", bg="#065f46", fg="#a7f3d0")
            self.preview_canvas.config(cursor="")

    def _on_canvas_mouse_down(self, event: tk.Event[Any]) -> None:
        if not self.is_draw_region_mode:
            blk, mode = self._hit_test_bbox_pin(int(event.x), int(event.y))
            if blk is not None:
                if mode == "delete_pin":
                    bid = blk.get("id")
                    self.last_blocks = [x for x in self.last_blocks if x.get("id") != bid]
                    self._redraw_pins_only()
                    self._say_miko(f"Рамка #{bid} полностью удалена с картинки! Перерисовываю страницу... 🗑✨")
                    self._auto_save_edited_blocks_now()
                    return
                if mode == "save_pin":
                    self._say_miko(f"Сохраняю изменения рамки #{blk.get('id')}... 💾✨")
                    self._auto_save_edited_blocks_now()
                    return
                if not blk.get("clean_bbox"):
                    blk["clean_bbox"] = list(blk["bbox"])
                self._bbox_drag_state = {
                    "block": blk,
                    "mode": mode,
                    "start_xy": (int(event.x), int(event.y)),
                    "orig_bbox": list(blk["bbox"]),
                    "moved": False,
                }
            return
        self._draw_start_xy = (int(event.x), int(event.y))
        if self._draw_rect_id is not None:
            self.preview_canvas.delete(self._draw_rect_id)
        self._draw_rect_id = self.preview_canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#10b981", width=2, dash=(5, 3)
        )

    def _on_canvas_mouse_drag(self, event: tk.Event[Any]) -> None:
        if not self.is_draw_region_mode:
            if not self._bbox_drag_state:
                return
            info = self._last_canvas_render_info
            scale = float(info.get("scale") or 1.0)
            orig_w = int(info.get("orig_w") or 1000)
            orig_h = int(info.get("orig_h") or 1000)
            sx, sy = self._bbox_drag_state["start_xy"]
            dx = int(round((int(event.x) - sx) / max(0.001, scale)))
            dy = int(round((int(event.y) - sy) / max(0.001, scale)))
            if abs(int(event.x) - sx) >= 3 or abs(int(event.y) - sy) >= 3:
                self._bbox_drag_state["moved"] = True
            blk = self._bbox_drag_state["block"]
            ox1, oy1, ox2, oy2 = self._bbox_drag_state["orig_bbox"]
            mode = self._bbox_drag_state["mode"]
            if mode == "move":
                bw, bh = ox2 - ox1, oy2 - oy1
                nx1 = max(0, min(orig_w - bw, ox1 + dx))
                ny1 = max(0, min(orig_h - bh, oy1 + dy))
                blk["bbox"] = [nx1, ny1, nx1 + bw, ny1 + bh]
                blk["bbox_modified"] = True
            elif mode == "resize_se":
                nx2 = max(ox1 + 18, min(orig_w, ox2 + dx))
                ny2 = max(oy1 + 18, min(orig_h, oy2 + dy))
                blk["bbox"] = [ox1, oy1, nx2, ny2]
                blk["bbox_modified"] = True
                blk["font_size"] = None
            elif mode == "resize_nw":
                nx1 = max(0, min(ox2 - 18, ox1 + dx))
                ny1 = max(0, min(oy2 - 18, oy1 + dy))
                blk["bbox"] = [nx1, ny1, ox2, oy2]
                blk["bbox_modified"] = True
                blk["font_size"] = None
            blk["_dirty"] = True
            self._redraw_pins_only()
            return
        if self._draw_start_xy is None or self._draw_rect_id is None:
            return
        sx, sy = self._draw_start_xy
        self.preview_canvas.coords(self._draw_rect_id, sx, sy, int(event.x), int(event.y))

    def _on_canvas_mouse_up(self, event: tk.Event[Any]) -> None:
        if not self.is_draw_region_mode:
            if self._bbox_drag_state:
                st = self._bbox_drag_state
                self._bbox_drag_state = None
                if st.get("moved"):
                    self._redraw_pins_only()
                    if bool(self.var_auto_save_boxes.get()):
                        self._say_miko(f"Рамка #{st['block'].get('id')} изменена мышкой! Перерисую страницу через {self.var_auto_save_sec.get()} сек~ ✨")
                        self._schedule_idle_autosave()
                    else:
                        self._say_miko(f"Рамка #{st['block'].get('id')} изменена! Нажмите 💾 на рамке для сохранения~ 💾")
            return
        if self._draw_start_xy is None:
            return
        sx, sy = self._draw_start_xy
        ex, ey = int(event.x), int(event.y)
        self._draw_start_xy = None
        if self._draw_rect_id is not None:
            self.preview_canvas.delete(self._draw_rect_id)
            self._draw_rect_id = None

        info = self._last_canvas_render_info
        if not info or not self.selected_page:
            return
        scale = float(info.get("scale") or 1.0)
        img_x0 = int(info.get("img_x0") or 0)
        img_y0 = int(info.get("img_y0") or 0)
        orig_w = int(info.get("orig_w") or 1)
        orig_h = int(info.get("orig_h") or 1)
        is_split = bool(info.get("is_split_both"))
        gap = int(info.get("gap") or 0)

        rx1 = (min(sx, ex) - img_x0) / max(0.001, scale)
        ry1 = (min(sy, ey) - img_y0) / max(0.001, scale)
        rx2 = (max(sx, ex) - img_x0) / max(0.001, scale)
        ry2 = (max(sy, ey) - img_y0) / max(0.001, scale)

        # Если выделение сделано на правой половине в режиме "До и После", смещаем X к координатам страницы
        if is_split and rx1 >= (orig_w + gap * 0.5):
            rx1 -= (orig_w + gap)
            rx2 -= (orig_w + gap)

        bx1 = max(0, min(orig_w - 4, int(round(rx1))))
        by1 = max(0, min(orig_h - 4, int(round(ry1))))
        bx2 = max(bx1 + 4, min(orig_w, int(round(rx2))))
        by2 = max(by1 + 4, min(orig_h, int(round(ry2))))

        if (bx2 - bx1) < 12 or (by2 - by1) < 12:
            self._say_miko("Рамка слишком маленькая — обведи весь текст внутри кружка~ 🎯")
            return

        self._toggle_draw_region_mode()
        self._translate_manual_bbox([bx1, by1, bx2, by2])

    def _translate_manual_bbox(self, bbox: list[int]) -> None:
        if not self.selected_page:
            return
        page = self.selected_page
        self._say_miko(f"🔍 Распознаю и перевожу выделенную область {bbox}... Секундочку~ ✨")
        payload = {
            "source_path": page["source_path"],
            "dest_path": page.get("dest_path") or None,
            "source_dir": str(self.current_folder) if self.current_folder else None,
            "bbox": bbox,
            "lang": SOURCE_LANG_MAP.get(self.var_source_lang.get(), "auto"),
            "target_lang": self.target_lang,
            "ollama_model": self._get_selected_model_id(),
            "uppercase": bool(self.var_uppercase.get()),
            "overflow_mode": self.var_overflow_mode.get(),
            "min_font_size": int(self.var_min_font.get()),
            "max_font_size": int(self.var_max_font.get()),
            "overflow_mode": self.var_overflow_mode.get(),
            "return_image": True,
            "image_format": "webp",
            "blocks": self.last_blocks,
        }

        def _worker() -> None:
            try:
                with httpx.Client(trust_env=False, timeout=180.0) as client:
                    resp = client.post(f"{SERVER_URL}/translate_manual_region", json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                b64_img = data.get("translated_image")
                trans_pil: Image.Image | None = None
                if b64_img:
                    raw = b64_img.split(",", 1)[1] if "," in b64_img else b64_img
                    trans_pil = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")

                def _done() -> None:
                    if trans_pil is not None:
                        self.preview_trans_pil = trans_pil
                    self.last_blocks = data.get("blocks", self.last_blocks)
                    page["blocks"] = self.last_blocks
                    page["is_translated"] = True
                    if data.get("saved_output_path"):
                        page["dest_path"] = data["saved_output_path"]
                    self.btn_show_phrases.config(text=f"✏️ Редактор текста ({len(self.last_blocks)})")
                    self._redraw_preview_canvas()
                    new_b = data.get("new_block") or {}
                    self._say_miko(
                        f"Готово! Добавлен блок #{new_b.get('id')}: «{new_b.get('original_text', '')}» ➔ «{new_b.get('translated_text', '')}» 🌸"
                    )
                    if self.current_folder:
                        self._refresh_folder_inspection_async()

                self.root.after(0, _done)
            except Exception as exc:
                self.root.after(0, lambda: self._say_miko(f"Ошибка перевода выделенной области: {exc}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_primary_folder_click(self) -> None:
        if self.is_folder_job_running:
            self._say_miko("Ставлю на паузу после текущей страницы... Весь прогресс сохранен! ⏸")
            threading.Thread(
                target=lambda: httpx.post(f"{SERVER_URL}/folder_stop", trust_env=False, timeout=8.0),
                daemon=True,
            ).start()
            return

        if not self.current_folder:
            self._choose_folder_dialog()
            if not self.current_folder:
                return

        payload = {
            "folder_path": str(self.current_folder),
            "lang": SOURCE_LANG_MAP.get(self.var_source_lang.get(), "auto"),
            "target_lang": self.target_lang,
            "ollama_model": self._get_selected_model_id(),
            "uppercase": bool(self.var_uppercase.get()),
            "overflow_mode": self.var_overflow_mode.get(),
            "min_font_size": int(self.var_min_font.get()),
            "max_font_size": int(self.var_max_font.get()),
            "overflow_mode": self.var_overflow_mode.get(),
            "detect_free_text": bool(self.var_detect_free_text.get()),
            "text_filter_mode": TEXT_FILTER_MODE_MAP.get(self.var_text_filter_label.get(), "dialogue_only"),
            "skip_sighs": bool(self.var_skip_sighs.get()),
            "skip_sfx": bool(self.var_skip_sfx.get()),
            "min_text_length": int(self.var_min_text_len.get()),
            "skip_existing": bool(self.var_skip_existing.get()),
            "batch_pages": 12,
            "force_retranslate": not bool(self.var_skip_existing.get()),
            "open_when_done": False,
        }

        def _worker() -> None:
            try:
                with httpx.Client(trust_env=False, timeout=25.0) as client:
                    resp = client.post(f"{SERVER_URL}/translate_folder", json=payload)
                    resp.raise_for_status()
                    self.is_folder_job_running = True
                    self.root.after(
                        0,
                        lambda: self.btn_primary_folder.config(
                            text="⏸ Пауза (остановить и сохранить прогресс)",
                            bg=ACCENT_ROSE,
                        ),
                    )
            except Exception as exc:
                self.root.after(0, lambda: self._say_miko(f"Ошибка запуска перевода папки: {exc}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _poll_folder_status_loop(self) -> None:
        if self.server_online and self.is_folder_job_running:
            threading.Thread(target=self._check_folder_status_once, daemon=True).start()
        self.root.after(1200, self._poll_folder_status_loop)

    def _check_folder_status_once(self) -> None:
        try:
            with httpx.Client(trust_env=False, timeout=5.0) as client:
                st = client.get(f"{SERVER_URL}/folder_status").json()
                self.root.after(0, lambda: self._apply_folder_status(st))
        except Exception:
            pass

    def _apply_folder_status(self, st: dict[str, Any]) -> None:
        status = st.get("status", "idle")
        total = int(st.get("total") or 0)
        cur = int(st.get("current") or 0)
        pct = (cur / total * 100.0) if total > 0 else 0.0
        self.progress_var.set(pct)
        self.lbl_folder_counts.config(text=f"{cur} / {total} готово")

        if status in ("running", "stopping"):
            self._say_miko(f"Перевожу [{cur}/{total}]: {st.get('current_file')} ({st.get('elapsed_sec')}с)... ✍️")
        else:
            self.is_folder_job_running = False
            self._refresh_folder_inspection_async()

    def _choose_single_file_dialog(self) -> None:
        initial = str(self.current_folder) if self.current_folder else str(BASE_DIR)
        selected = filedialog.askopenfilename(
            title="Выберите страницу манги",
            initialdir=initial,
            filetypes=[("Изображения манги", "*.jpg *.jpeg *.png *.webp *.bmp"), ("Все файлы", "*.*")],
        )
        if not selected:
            return
        fpath = Path(selected).resolve()
        self.selected_page = {
            "index": 1,
            "rel_path": fpath.name,
            "filename": fpath.name,
            "source_path": str(fpath),
            "dest_path": "",
            "is_translated": False,
        }
        self._load_preview_from_disk(self.selected_page)
        self._translate_selected_page()

    def _translate_selected_page(self) -> None:
        if not self.selected_page or self.is_translating_single:
            return
        self.is_translating_single = True
        page = self.selected_page
        self.btn_context_page.config(state=tk.DISABLED)
        self._say_miko(f"Старательно перевожу страницу «{page['rel_path']}»... Подожди чуточку~ ✍️✨")

        payload = {
            "file_path": page["source_path"],
            "source_dir": str(self.current_folder) if self.current_folder else None,
            "lang": SOURCE_LANG_MAP.get(self.var_source_lang.get(), "auto"),
            "target_lang": self.target_lang,
            "ollama_model": self._get_selected_model_id(),
            "uppercase": bool(self.var_uppercase.get()),
            "overflow_mode": self.var_overflow_mode.get(),
            "min_font_size": int(self.var_min_font.get()),
            "max_font_size": int(self.var_max_font.get()),
            "overflow_mode": self.var_overflow_mode.get(),
            "detect_free_text": bool(self.var_detect_free_text.get()),
            "text_filter_mode": TEXT_FILTER_MODE_MAP.get(self.var_text_filter_label.get(), "dialogue_only"),
            "skip_sighs": bool(self.var_skip_sighs.get()),
            "skip_sfx": bool(self.var_skip_sfx.get()),
            "min_text_length": int(self.var_min_text_len.get()),
            "force_retranslate": True,
            "return_image": True,
            "image_format": "webp",
        }

        def _worker() -> None:
            try:
                with httpx.Client(trust_env=False, timeout=240.0) as client:
                    resp = client.post(f"{SERVER_URL}/translate_page_file", json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                b64_img = data.get("translated_image")
                trans_pil: Image.Image | None = None
                if b64_img:
                    raw = b64_img.split(",", 1)[1] if "," in b64_img else b64_img
                    trans_pil = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")

                def _done() -> None:
                    self.preview_trans_pil = trans_pil
                    self.last_blocks = data.get("blocks", [])
                    self.btn_show_phrases.config(text=f"✏️ Редактор текста ({len(self.last_blocks)})")
                    self._redraw_preview_canvas()
                    self._say_miko(
                        f"Страница «{page['rel_path']}» готова за {data.get('processing_time_sec')}с! "
                        f"Найдено блоков: {data.get('blocks_count')} (вне бабблов: {data.get('free_text_blocks_count', 0)}) 🌸"
                    )
                    if self.current_folder:
                        self._refresh_folder_inspection_async()

                self.root.after(0, _done)
            except Exception as exc:
                self.root.after(0, lambda: self._say_miko(f"Ой, не получилось перевести страницу: {exc}"))
            finally:
                self.is_translating_single = False
                self.root.after(0, lambda: self.btn_context_page.config(state=tk.NORMAL))

        threading.Thread(target=_worker, daemon=True).start()

    def _open_output_folder_in_explorer(self) -> None:
        out_dir = self.current_output_dir
        if out_dir and out_dir.exists() and os.name == "nt" and hasattr(os, "startfile"):
            os.startfile(str(out_dir))
        else:
            self._say_miko("Сначала выбери папку и переведи хотя бы одну страничку~ 🌸")

    def _on_close(self) -> None:
        self._save_gui_settings()
        if self.server_process is not None and self.server_process.poll() is None:
            try:
                self.server_process.terminate()
            except Exception:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    MikoMangaStudioGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
