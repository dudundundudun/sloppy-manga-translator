"""
Консольный клиент для выбора модели, пакетного перевода папок (с продолжением с места остановки)
и перевода отдельных страниц (v6.1.0).
Вызывается из translate_folder.bat и test_page.bat.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def run_select_model(server_url: str = "http://127.0.0.1:5000") -> int:
    _configure_stdout()
    try:
        with httpx.Client(trust_env=False, timeout=20.0) as client:
            resp = client.get(f"{server_url}/models?refresh=true")
            resp.raise_for_status()
            data = resp.json()

            models = [
                m for m in data.get("models", [])
                if str(m.get("installed", "true")).lower() != "false"
                and "glm-ocr" not in str(m.get("id", "")).lower()
            ]
            active = data.get("active_model", "auto")
            unsloth_url = data.get("unsloth_url", "http://127.0.0.1:8888")
            u_stat = data.get("unsloth_status", {})

            print("=" * 68)
            print(f"  CURRENT ACTIVE MODEL : {active}")
            print(f"  UNSLOTH STUDIO URL   : {unsloth_url} ({'ONLINE' if u_stat.get('reachable') else 'NOT DETECTED'})")
            print(f"  WEB STUDIO PANEL     : {server_url}/")
            print("=" * 68)
            print("Available options:")
            print("  [0] Auto-select best model (auto)")
            for idx, m in enumerate(models, start=1):
                mark = "  <-- ACTIVE" if m.get("id") == active else ""
                print(f"  [{idx}] {m.get('id')}  ({m.get('name')}){mark}")
            print("  [U] Set custom Unsloth Studio port/URL (e.g. http://127.0.0.1:8888)")
            print("  [M] Type custom model name manually")
            print("-" * 68)

            choice = input("Select number [0..N], 'U', 'M', or press Enter to cancel: ").strip()
            if not choice:
                return 0

            payload: dict[str, str] = {}
            if choice.upper() == "U":
                new_url = input(f"Enter Unsloth Studio URL (default {unsloth_url}): ").strip()
                if new_url:
                    payload["unsloth_url"] = new_url
                payload["selected_model"] = "auto"
            elif choice.upper() == "M":
                custom_mod = input("Enter model ID (e.g. unsloth::default or ollama::qwen2.5:14b): ").strip()
                if not custom_mod:
                    return 0
                payload["selected_model"] = custom_mod
            elif choice == "0":
                payload["selected_model"] = "auto"
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    payload["selected_model"] = str(models[idx]["id"])
                else:
                    print("[ERROR] Invalid option number.")
                    return 1
            else:
                payload["selected_model"] = choice

            save_resp = client.post(f"{server_url}/settings", json=payload)
            save_resp.raise_for_status()
            saved = save_resp.json()
            print("=" * 68)
            print(f"[OK] Active model switched to: {saved.get('active_model')}")
            print("=" * 68)
            return 0
    except Exception as exc:
        print(f"[ERROR] Failed to connect to local server ({server_url}): {exc}")
        print("Make sure start_server.bat or start_gui.bat is running first!")
        return 1


def run_test_page(
    img_path_str: str,
    target_lang: str = "en",
    min_font_size: int = 12,
    max_font_size: int = 32,
    detect_free_text: bool | None = None,
    server_url: str = "http://127.0.0.1:5000",
) -> int:
    _configure_stdout()
    img_path = Path(img_path_str).resolve()
    if not img_path.exists():
        print(f"[ERROR] File not found: {img_path}")
        return 1

    try:
        with httpx.Client(trust_env=False, timeout=240.0) as client:
            req_payload: dict[str, object] = {
                "file_path": str(img_path),
                "lang": "auto",
                "target_lang": target_lang,
                "min_font_size": min_font_size,
                "max_font_size": max_font_size,
                "return_image": False,
            }
            if detect_free_text is not None:
                req_payload["detect_free_text"] = detect_free_text
            resp = client.post(
                f"{server_url}/translate_page_file",
                json=req_payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        print(f"[ERROR] Failed to translate image via {server_url}: {exc}")
        return 1

    print(
        f"Done in {data.get('processing_time_sec')}s | "
        f"Lang: {data.get('detected_lang', 'auto')} -> {target_lang} | "
        f"Blocks: {data.get('blocks_count')} (free text: {data.get('free_text_blocks_count', 0)}) | "
        f"Model: {data.get('ollama_model')}"
    )
    print("-" * 60)
    for b in data.get("blocks", []):
        tag = "[TXT]" if b.get("is_free_text") else "[BBL]"
        print(f"#{b.get('id')} {tag} {b.get('bbox')}: {b.get('original_text')}  ==>  {b.get('translated_text', '')}")
    print("-" * 60)

    out_img = data.get("saved_output_path")
    print("Saved translated page:", out_img)
    if out_img and os.path.exists(out_img) and hasattr(os, "startfile"):
        try:
            os.startfile(out_img)
        except Exception:
            pass
    return 0


def run_translate_folder(
    target_lang: str = "en",
    folder_path: str | None = None,
    skip_existing: bool = True,
    min_font_size: int = 12,
    max_font_size: int = 32,
    detect_free_text: bool | None = None,
    server_url: str = "http://127.0.0.1:5000",
) -> int:
    _configure_stdout()
    try:
        with httpx.Client(trust_env=False, timeout=240.0) as client:
            payload: dict[str, object] = {
                "lang": "auto",
                "target_lang": target_lang,
                "min_font_size": min_font_size,
                "max_font_size": max_font_size,
                "skip_existing": skip_existing,
                "open_when_done": True,
            }
            if detect_free_text is not None:
                payload["detect_free_text"] = detect_free_text
            if folder_path:
                payload["folder_path"] = folder_path

            resp = client.post(f"{server_url}/translate_folder", json=payload)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "cancelled":
                print("Folder selection cancelled.")
                return 0
            if data.get("status") == "busy":
                print(f"[BUSY] {data.get('message') or 'Folder translation or native dialog is currently busy.'}")
                return 1

            already = data.get("already_translated", 0)
            total = data.get("total_images", 0)
            print(
                f"Folder: {data.get('source_dir')} -> {data.get('output_dir')}\n"
                f"Total pages: {total} | Already translated (skipped): {already} | Remaining: {data.get('remaining_images', total)}"
            )

            idle_ticks = 0
            for _ in range(36000):
                time.sleep(1.0)
                st_resp = client.get(f"{server_url}/folder_status")
                st_resp.raise_for_status()
                st = st_resp.json()
                det_lang = str(st.get("detected_lang") or st.get("lang") or "auto").upper()
                print(
                    f"[{st.get('current', 0)}/{st.get('total', 0)} | {det_lang}->{target_lang.upper()}] "
                    f"{st.get('current_file', '')} ({st.get('elapsed_sec', 0.0)}s)"
                )
                status = st.get("status")
                if status == "completed":
                    print(f"[OK] Completed! Output folder: {st.get('output_dir')}")
                    return 0
                if status == "stopped":
                    print(f"[PAUSED] Stopped at {st.get('current')}/{st.get('total')}. Next run will resume from here!")
                    return 0
                if status == "error":
                    print(f"[ERROR] Batch translation failed: {st.get('error')}")
                    return 1
                if status == "idle":
                    idle_ticks += 1
                    if idle_ticks >= 3:
                        print("[INFO] Server reports folder job is idle.")
                        return 0
                else:
                    idle_ticks = 0
    except KeyboardInterrupt:
        try:
            with httpx.Client(trust_env=False, timeout=5.0) as client:
                client.post(f"{server_url}/folder_stop")
            print("\n[PAUSED] Stop signal sent to server. Progress saved!")
        except Exception:
            pass
        return 0
    except Exception as exc:
        print(f"[ERROR] Failed to communicate with local server ({server_url}): {exc}")
        return 1

    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python cli_client.py models")
        print("  python cli_client.py test <image_path> [en|ru|romaji]")
        print("  python cli_client.py folder [en|ru|romaji] [optional_folder_path]")
        return 1

    mode = sys.argv[1].lower().strip()
    if mode in ("models", "model", "select"):
        return run_select_model()

    if mode == "test":
        img_path = sys.argv[2] if len(sys.argv) >= 3 else "test.jpg"
        t_lang = sys.argv[3] if len(sys.argv) >= 4 else "en"
        return run_test_page(img_path, t_lang)

    if mode == "folder":
        t_lang = sys.argv[2] if len(sys.argv) >= 3 else "en"
        folder = sys.argv[3] if len(sys.argv) >= 4 else None
        return run_translate_folder(t_lang, folder)

    print(f"Unknown mode: {mode}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
