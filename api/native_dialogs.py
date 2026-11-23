"""
Нативные системные диалоги выбора папки и файла для Miko-chan Manga Studio.

Запускает Tkinter-диалог в изолированном дочернем процессе (на Windows — через `pythonw.exe`
с `AllowSetForegroundWindow`, `AttachThreadInput` и корректным top-level HWND), чтобы окно
выбора гарантированно открывалось поверх браузерного окна студии (`--app`) и не блокировало
фоновые потоки FastAPI.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

logger = logging.getLogger("manga_server")

__all__ = [
    "_native_picker_lock",
    "NativePickerBusyError",
    "_SUBPROCESS_PICKER_SCRIPT",
    "_run_native_picker_dialog",
    "open_native_folder_picker",
    "open_native_file_picker",
]

_native_picker_lock = threading.Lock()


class NativePickerBusyError(RuntimeError):
    """Исключение, когда системное окно выбора папки/файла уже открыто."""


_SUBPROCESS_PICKER_SCRIPT = r"""
import ctypes, json, os, sys, tkinter as tk
from tkinter import filedialog

mode = sys.argv[1] if len(sys.argv) > 1 else "folder"
owner_hwnd = 0
if len(sys.argv) > 2:
    try:
        owner_hwnd = int(sys.argv[2])
    except Exception:
        owner_hwnd = 0

if os.name == "nt":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

root = tk.Tk()
root.title("Miko-chan Manga Studio")
sw = root.winfo_screenwidth()
sh = root.winfo_screenheight()
root.geometry(f"1x1+{sw // 2}+{sh // 2}")
root.attributes("-topmost", True)
root.attributes("-alpha", 0.01)
root.deiconify()
root.update_idletasks()
root.update()

if os.name == "nt":
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
        fg_hwnd = user32.GetForegroundWindow() or owner_hwnd
        fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
        cur_tid = kernel32.GetCurrentThreadId()
        attached = False
        if fg_tid and fg_tid != cur_tid:
            attached = bool(user32.AttachThreadInput(fg_tid, cur_tid, True))
        try:
            user32.ShowWindow(hwnd, 5)  # SW_SHOW
            user32.SetWindowPos(hwnd, -1, sw // 2, sh // 2, 1, 1, 0x0001 | 0x0002 | 0x0040)  # HWND_TOPMOST
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            user32.SetActiveWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(fg_tid, cur_tid, False)
    except Exception:
        pass

root.lift()
root.focus_force()
root.update()

if mode == "file":
    selected = filedialog.askopenfilename(
        parent=root,
        title="Выберите страницу манги для перевода",
        filetypes=[
            ("Изображения манги", "*.jpg *.jpeg *.png *.webp *.bmp"),
            ("Все файлы", "*.*"),
        ],
    )
else:
    selected = filedialog.askdirectory(
        parent=root,
        mustexist=True,
        title="Выберите папку с мангой / манхвой / маньхуа для перевода",
    )

try:
    root.destroy()
except Exception:
    pass

sys.stdout.write(json.dumps({"selected": selected or ""}, ensure_ascii=False))
sys.stdout.flush()
"""


def _run_native_picker_dialog(mode: str = "folder") -> str | None:
    """
    Открывает системное окно выбора папки/файла Windows в изолированном главном потоке дочернего процесса
    с принудительным выводом поверх окна браузера (--app) через корректный top-level HWND,
    AllowSetForegroundWindow и AttachThreadInput.
    Исключает зависание Tcl/Tk в фоновых потоках ThreadPoolExecutor и открытие диалога позади окна студии!
    """
    if not _native_picker_lock.acquire(blocking=False):
        raise NativePickerBusyError(
            "Окно выбора папки уже открыто — проверьте панель задач Windows"
        )
    try:
        try:
            owner_hwnd = 0
            picker_exe = sys.executable
            if os.name == "nt":
                try:
                    import ctypes

                    user32 = ctypes.windll.user32
                    # ASFW_ANY (-1) разрешает дочернему процессу выводить диалог поверх активного окна Edge/Chrome --app
                    user32.AllowSetForegroundWindow(-1)
                    owner_hwnd = int(user32.GetForegroundWindow() or 0)
                except Exception as _fg_exc:
                    logger.debug("AllowSetForegroundWindow skipped: %s", _fg_exc)
                pythonw_candidate = Path(sys.executable).with_name("pythonw.exe")
                if pythonw_candidate.exists():
                    picker_exe = str(pythonw_candidate)

            env = dict(os.environ)
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.run(
                [picker_exe, "-c", _SUBPROCESS_PICKER_SCRIPT, mode, str(owner_hwnd)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=180,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                raw_out = proc.stdout.strip().splitlines()[-1]
                data = json.loads(raw_out)
                sel = str(data.get("selected") or "").strip()
                return sel if sel else None
        except subprocess.TimeoutExpired:
            logger.warning("Истек таймаут (180 сек) ожидания системного окна выбора (%s).", mode)
            return None
        except Exception as sub_exc:
            logger.warning("Ошибка запуска системного диалога выбора (%s): %s", mode, sub_exc)
            return None
        if proc.returncode != 0:
            err_tail = (proc.stderr or "").strip()[-300:]
            logger.warning(
                "Дочерний процесс системного диалога (%s) завершился с кодом %s%s",
                mode,
                proc.returncode,
                f": {err_tail}" if err_tail else "",
            )
        return None
    finally:
        _native_picker_lock.release()


def open_native_folder_picker() -> str | None:
    return _run_native_picker_dialog("folder")


def open_native_file_picker() -> str | None:
    return _run_native_picker_dialog("file")