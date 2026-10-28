"""Проверка синхронности версий (шаг 7 плана исправлений).

APP_VERSION в app_version.py — единый источник. Не-Python поверхности
не могут его импортировать, поэтому сверяем их этим скриптом перед
релизом: python tools/check_versions.py  (код выхода 1 при расхождении).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app_version import APP_VERSION


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def main() -> int:
    checks: list[tuple[str, str | None]] = []

    manifest = json.loads(read("extension/manifest.json"))
    checks.append(("extension/manifest.json version", manifest.get("version")))

    m = re.search(r'class="mt-ver-tag">v([\d.]+)<', read("extension/content.js"))
    checks.append(("extension/content.js panel badge", m.group(1) if m else None))

    m = re.search(r"<title>[^<]*v([\d.]+)</title>", read("web_studio.html"))
    checks.append(("web_studio.html title", m.group(1) if m else None))

    m = re.search(r"MikoMangaStudio/([\d.]+)", read("ocr_engine.py"))
    checks.append(("ocr_engine.py User-Agent", m.group(1) if m else None))

    failed = False
    print(f"APP_VERSION = {APP_VERSION}")
    for name, value in checks:
        ok = value == APP_VERSION
        failed |= not ok
        print(f"  {'OK ' if ok else 'FAIL'} {name}: {value}")
    if failed:
        print("Версии разъехались — поправьте поверхности выше под APP_VERSION.")
        return 1
    print("Все версии синхронны.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
