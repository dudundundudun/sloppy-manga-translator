"""
Нативные системные диалоги выбора папки и файла для Miko-chan Manga Studio.

Запускает Tkinter-диалог в изолированном дочернем процессе (на Windows — через `pythonw.exe`
с `AllowSetForegroundWindow`, `AttachThreadInput` и корректным 64-bit top-level HWND), чтобы окно
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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger("manga_server")

__all__ = [
    "PickerResult",
    "_native_picker_lock",
    "NativePickerBusyError",
    "_SUBPROCESS_PICKER_SCRIPT",
    "run_native_picker_result",
    "open_native_folder_picker_result",
    "open_native_file_picker_result",
    "_run_native_picker_dialog",
    "open_native_folder_picker",
    "open_native_file_picker",
]

_native_picker_lock = threading.Lock()


class NativePickerBusyError(RuntimeError):
    """Исключение, когда системное окно выбора папки/файла уже открыто."""


@dataclass(frozen=True)
class PickerResult:
    """Типизированный результат системного диалога выбора папки или файла."""

    status: Literal["ok", "cancelled", "timeout", "error"]
    path: str | None = None
    error: str | None = None


_SUBPROCESS_PICKER_SCRIPT = r"""
import ctypes, json, os, sys, tkinter as tk
from ctypes import wintypes
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
    except Exception as exc:
        sys.stderr.write(f"SetProcessDpiAwareness skipped: {exc}\n")

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
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        user32.GetParent.argtypes = [wintypes.HWND]
        user32.GetParent.restype = wintypes.HWND
        user32.GetForegroundWindow.argtypes = []
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        kernel32.GetCurrentThreadId.argtypes = []
        kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
        user32.AttachThreadInput.restype = wintypes.BOOL
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow.restype = wintypes.BOOL
        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL
        user32.BringWindowToTop.argtypes = [wintypes.HWND]
        user32.BringWindowToTop.restype = wintypes.BOOL
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        user32.SetForegroundWindow.restype = wintypes.BOOL
        user32.SetActiveWindow.argtypes = [wintypes.HWND]
        user32.SetActiveWindow.restype = wintypes.HWND

        raw_wid = int(root.winfo_id())
        parent_hwnd = user32.GetParent(wintypes.HWND(raw_wid))
        hwnd = wintypes.HWND(int(parent_hwnd) if parent_hwnd else raw_wid)
        fg_raw = user32.GetForegroundWindow()
        fg_hwnd_val = int(fg_raw) if fg_raw else owner_hwnd
        fg_tid = (
            user32.GetWindowThreadProcessId(wintypes.HWND(fg_hwnd_val), None)
            if fg_hwnd_val
            else 0
        )
        cur_tid = kernel32.GetCurrentThreadId()
        attached = False
        if fg_tid and fg_tid != cur_tid:
            attached = bool(user32.AttachThreadInput(fg_tid, cur_tid, True))
        try:
            user32.ShowWindow(hwnd, 5)  # SW_SHOW
            ctypes.set_last_error(0)
            ok_pos = user32.SetWindowPos(
                hwnd,
                wintypes.HWND(-1),  # HWND_TOPMOST
                sw // 2,
                sh // 2,
                1,
                1,
                0x0001 | 0x0002 | 0x0040,  # SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW
            )
            if not ok_pos:
                err_code = ctypes.get_last_error()
                sys.stderr.write(f"SetWindowPos(HWND_TOPMOST) failed: last_error={err_code}\n")
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            user32.SetActiveWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(fg_tid, cur_tid, False)
    except Exception as win_exc:
        sys.stderr.write(f"Win32 topmost setup error: {win_exc}\n")

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


def _terminate_picker_process_tree(proc: subprocess.Popen[str]) -> None:
    """Завершает дочерний процесс выбора папки вместе со всеми потомками (venv launcher stub)."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as exc:
            logger.debug("taskkill для диалога выбора завершился с ошибкой: %s", exc)
    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass
    try:
        proc.communicate(timeout=3)
    except Exception:
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass


def run_native_picker_result(mode: str = "folder", timeout_sec: float = 180.0) -> PickerResult:
    """
    Открывает системное окно выбора папки/файла Windows в изолированном главном потоке дочернего процесса
    с принудительным выводом поверх окна браузера (--app) через корректный 64-bit top-level HWND,
    AllowSetForegroundWindow и AttachThreadInput.
    Возвращает типизированный PickerResult, различая отмену, таймаут и ошибку запуска/диалога.
    """
    if not _native_picker_lock.acquire(blocking=False):
        raise NativePickerBusyError(
            "Окно выбора папки уже открыто — проверьте панель задач Windows"
        )
    try:
        owner_hwnd = 0
        picker_exe = sys.executable
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes

                user32 = ctypes.WinDLL("user32", use_last_error=True)
                user32.AllowSetForegroundWindow.argtypes = [wintypes.DWORD]
                user32.AllowSetForegroundWindow.restype = wintypes.BOOL
                user32.GetForegroundWindow.argtypes = []
                user32.GetForegroundWindow.restype = wintypes.HWND

                # ASFW_ANY (0xFFFFFFFF) разрешает дочернему процессу выводить диалог поверх активного окна Edge/Chrome --app
                ctypes.set_last_error(0)
                ok_asfw = user32.AllowSetForegroundWindow(wintypes.DWORD(0xFFFFFFFF))
                if not ok_asfw:
                    logger.debug(
                        "AllowSetForegroundWindow returned 0 (last_error=%s)",
                        ctypes.get_last_error(),
                    )
                fg_raw = user32.GetForegroundWindow()
                owner_hwnd = int(fg_raw) if fg_raw else 0
            except Exception as _fg_exc:
                logger.debug("AllowSetForegroundWindow skipped: %s", _fg_exc)
            pythonw_candidate = Path(sys.executable).with_name("pythonw.exe")
            if pythonw_candidate.exists():
                picker_exe = str(pythonw_candidate)

        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            proc = subprocess.Popen(
                [picker_exe, "-c", _SUBPROCESS_PICKER_SCRIPT, mode, str(owner_hwnd)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except Exception as launch_exc:
            logger.warning("Ошибка запуска системного диалога выбора (%s): %s", mode, launch_exc)
            return PickerResult(
                status="error",
                error=f"Не удалось запустить системный диалог выбора: {launch_exc}",
            )

        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Истек таймаут (%.0f сек) ожидания системного окна выбора (%s).",
                timeout_sec,
                mode,
            )
            _terminate_picker_process_tree(proc)
            return PickerResult(
                status="timeout",
                error=f"Истек таймаут ({int(timeout_sec)} сек) ожидания системного окна выбора. Проверьте панель задач Windows.",
            )
        except Exception as comm_exc:
            logger.warning("Ошибка ожидания системного диалога выбора (%s): %s", mode, comm_exc)
            _terminate_picker_process_tree(proc)
            return PickerResult(
                status="error",
                error=f"Ошибка системного диалога выбора: {comm_exc}",
            )

        if proc.returncode != 0:
            err_tail = (stderr_data or "").strip()[-300:]
            logger.warning(
                "Дочерний процесс системного диалога (%s) завершился с кодом %s%s",
                mode,
                proc.returncode,
                f": {err_tail}" if err_tail else "",
            )
            return PickerResult(
                status="error",
                error=(
                    f"Системный диалог завершился с кодом {proc.returncode}"
                    + (f": {err_tail}" if err_tail else "")
                ),
            )

        raw_lines = [ln.strip() for ln in (stdout_data or "").splitlines() if ln.strip()]
        if not raw_lines:
            err_tail = (stderr_data or "").strip()[-300:]
            logger.warning("Дочерний процесс системного диалога (%s) не вернул данных%s", mode, f": {err_tail}" if err_tail else "")
            return PickerResult(
                status="error",
                error="Системный диалог не вернул результат выбора" + (f": {err_tail}" if err_tail else ""),
            )

        try:
            data = json.loads(raw_lines[-1])
        except Exception as json_exc:
            logger.warning("Некорректный JSON от системного диалога (%s): %s", mode, json_exc)
            return PickerResult(
                status="error",
                error=f"Некорректный ответ системного диалога: {json_exc}",
            )

        sel = str(data.get("selected") or "").strip()
        if sel:
            return PickerResult(status="ok", path=sel)
        return PickerResult(status="cancelled")
    finally:
        _native_picker_lock.release()


def open_native_folder_picker_result(timeout_sec: float = 180.0) -> PickerResult:
    return run_native_picker_result("folder", timeout_sec=timeout_sec)


def open_native_file_picker_result(timeout_sec: float = 180.0) -> PickerResult:
    return run_native_picker_result("file", timeout_sec=timeout_sec)


def _run_native_picker_dialog(mode: str = "folder") -> str | None:
    res = run_native_picker_result(mode)
    return res.path if res.status == "ok" else None


def open_native_folder_picker() -> str | None:
    return _run_native_picker_dialog("folder")


def open_native_file_picker() -> str | None:
    return _run_native_picker_dialog("file")
