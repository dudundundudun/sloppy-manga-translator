"""
Автотесты безопасности и базовой логики Miko-chan Manga Studio (v6.1.0).
Совместимы с `pytest` и с прямым запуском: `python tests/test_security_and_core.py`.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import HTTPException

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_server_module():
    """Загружает server.py с лёгкими заглушками GPU-модулей, чтобы тесты шли быстро и без VRAM."""
    if "server_tested_mod" in sys.modules:
        return sys.modules["server_tested_mod"]

    for mod_name in ("torch", "uvicorn"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    stub_specs = [
        ("cleaner", ["BubbleCleaner"]),
        ("ocr_engine", ["MangaOCREngine", "sort_page_blocks_reading_order_dag"]),
        ("renderer", ["MangaRenderer"]),
        (
            "translator",
            [
                "clean_and_validate_glossary_name",
                "enforce_glossary_on_translated_text",
                "BUILT_IN_MODEL_STACK_PRESETS",
                "CLOUD_API_PRESETS",
                "DEEPSEEK_ANTI_CENSORSHIP_PROMPT_TEMPLATE",
                "DEFAULT_MODEL_STACK_TEMPLATE",
                "FICTION_NOVEL_CONTEXT_NOTE",
                "PIPELINE_MODES",
                "RECOMMENDED_LOCAL_MODELS",
                "TRANSLATION_PROFILES",
                "EndpointCircuitBreaker",
                "HybridMangaTranslator",
                "OllamaTranslator",
                "export_glossary_to_sakura_text",
                "export_glossary_to_tsv",
                "japanese_to_romaji_local",
                "merge_glossaries",
                "normalize_glossary_entries",
                "is_translated_sigh_text",
                "parse_glossary_text_or_json",
            ],
        ),
    ]
    for mod_name, attrs in stub_specs:
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            for attr in attrs:
                setattr(m, attr, {} if attr.isupper() else type(attr, (), {}))
            sys.modules[mod_name] = m

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("server_tested_mod", REPO_ROOT / "server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["server_tested_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


# ==================== STEP 1: API KEYS & ORIGIN ====================

def test_mask_api_key():
    srv = _load_server_module()
    assert srv._mask_api_key("") == ""
    assert srv._mask_api_key(None) == ""
    assert srv._mask_api_key("short") == "***"
    assert srv._mask_api_key("sk-1234567890abcdef") == "sk-12...cdef"


def test_is_trusted_local_origin():
    srv = _load_server_module()
    assert srv._is_trusted_local_origin(None) is True
    assert srv._is_trusted_local_origin("http://127.0.0.1:5000") is True
    assert srv._is_trusted_local_origin("http://localhost:5000") is True
    assert srv._is_trusted_local_origin("chrome-extension://abcdefghijklmnop") is True

    assert srv._is_trusted_local_origin("null") is False
    assert srv._is_trusted_local_origin("https://evil.example.com") is False
    assert srv._is_trusted_local_origin("http://127.0.0.1.evil.com") is False
    assert srv._is_trusted_local_origin("javascript:alert(1)") is False


def test_privileged_endpoints_and_clear_api_key():
    srv = _load_server_module()
    required = {"/local_image", "/open_folder", "/settings", "/shutdown", "/delete_model", "/pick_folder", "/folder_inspect"}
    assert required.issubset(srv._PRIVILEGED_ENDPOINTS)
    req = srv.ServerSettingsRequest()
    assert req.clear_api_key is False


# ==================== STEP 2: PATHS, OPEN_FOLDER & SSRF ====================

def test_open_folder_blocks_files_and_symlinks():
    srv = _load_server_module()

    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp).resolve()
            sub_dir = tmp_dir / "chapter_01"
            sub_dir.mkdir()
            fake_exe = tmp_dir / "run.exe"
            fake_exe.write_bytes(b"MZ")
            sym = tmp_dir / "sym_exe"
            try:
                sym.symlink_to(fake_exe)
                has_sym = True
            except OSError:
                has_sym = False

            for bad_target in ([fake_exe, sym] if has_sym else [fake_exe]):
                try:
                    await srv.open_folder_in_explorer_endpoint(path=str(bad_target))
                    raise AssertionError(f"Expected 400 for {bad_target}")
                except HTTPException as exc:
                    assert exc.status_code == 400

            try:
                await srv.open_folder_in_explorer_endpoint(path=str(tmp_dir / "nonexistent_dir"))
                raise AssertionError("Expected 404 for missing dir")
            except HTTPException as exc:
                assert exc.status_code == 404

            with patch.object(srv.os, "name", "posix"):
                ok = await srv.open_folder_in_explorer_endpoint(path=str(sub_dir))
                assert ok == {"status": "ok"}

    asyncio.run(_run())


def test_allowed_roots_strict_allowlist():
    srv = _load_server_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp).resolve()
        manga_dir = tmp_dir / "manga"
        manga_dir.mkdir()
        img = manga_dir / "01.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        srv._SESSION_ALLOWED_ROOTS.clear()
        srv.folder_job_state["source_dir"] = None
        srv.folder_job_state["output_dir"] = None
        srv.translator_engine = None

        assert srv._is_path_within_allowed_roots(img) is False
        assert srv._is_path_within_allowed_roots(srv.BASE_DIR / ".." / ".." / "secret.png") is False
        assert srv._is_path_within_allowed_roots(srv.BASE_DIR / "mascot.png") is True
        assert srv._is_path_within_allowed_roots(srv.TEMP_DIR / "preview.png") is True

        srv._register_allowed_root(manga_dir)
        assert srv._is_path_within_allowed_roots(img) is True


def test_ssrf_blocks_private_ips_and_bad_schemes():
    srv = _load_server_module()
    blocked = [
        "http://127.0.0.1/a.png",
        "http://localhost/a.png",
        "http://sub.localhost/a.png",
        "http://10.0.0.1/a.png",
        "http://172.16.0.1/a.png",
        "http://192.168.1.1/a.png",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/a.png",
        "http://[::ffff:127.0.0.1]/a.png",
        "http://0.0.0.0/a.png",
        "http://100.64.0.1/a.png",
        "file:///etc/passwd",
        "ftp://example.com/a.png",
    ]
    for url in blocked:
        try:
            srv._validate_remote_image_url(url)
            raise AssertionError(f"Expected SSRF block for {url}")
        except ValueError:
            pass

    with patch.object(srv.socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("127.0.0.1", 80))]):
        try:
            srv._validate_remote_image_url("https://rebind.example/img.png")
            raise AssertionError("Expected DNS->127.0.0.1 to be blocked")
        except ValueError:
            pass

    with patch.object(srv.socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 443))]):
        assert srv._validate_remote_image_url("https://manga.example/01.png") == "https://manga.example/01.png"


def test_ssrf_blocks_redirects_and_enforces_25mb_cap():
    srv = _load_server_module()
    public_dns = [(2, 1, 6, "", ("93.184.216.34", 443))]

    class FakeResp:
        def __init__(self, status_code, headers=None, chunks=None):
            self.status_code = status_code
            self.headers = headers or {}
            self._chunks = chunks or []
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("err", request=None, response=None)
        async def aiter_bytes(self, chunk_size=65536):
            for c in self._chunks:
                yield c

    def _make_client(resp_obj):
        class _Client:
            def __init__(self, *args, **kwargs):
                assert kwargs.get("trust_env") is False
                assert kwargs.get("follow_redirects") is False
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def stream(self, method, url, headers=None):
                return resp_obj
        return _Client

    async def _run():
        # Redirect to loopback -> blocked
        with patch.object(srv.socket, "getaddrinfo", return_value=public_dns), \
             patch.object(srv.httpx, "AsyncClient", _make_client(FakeResp(302, {"location": "http://127.0.0.1:5000/settings"}))):
            try:
                await srv._download_remote_image_bytes("https://manga.example/redir.png", {})
                raise AssertionError("Expected redirect SSRF block")
            except ValueError:
                pass

        # Content-Length > 25 MB -> HTTP 413
        with patch.object(srv.socket, "getaddrinfo", return_value=public_dns), \
             patch.object(srv.httpx, "AsyncClient", _make_client(FakeResp(200, {"content-length": str(26 * 1024 * 1024)}))):
            try:
                await srv._download_remote_image_bytes("https://manga.example/big.png", {})
                raise AssertionError("Expected 413 for Content-Length > 25MB")
            except HTTPException as exc:
                assert exc.status_code == 413

        # Streamed > 25 MB -> HTTP 413
        with patch.object(srv.socket, "getaddrinfo", return_value=public_dns), \
             patch.object(srv.httpx, "AsyncClient", _make_client(FakeResp(200, {}, [b"a" * (1024 * 1024)] * 26))):
            try:
                await srv._download_remote_image_bytes("https://manga.example/stream.png", {})
                raise AssertionError("Expected 413 for streamed > 25MB")
            except HTTPException as exc:
                assert exc.status_code == 413

        # Small valid image -> OK
        with patch.object(srv.socket, "getaddrinfo", return_value=public_dns), \
             patch.object(srv.httpx, "AsyncClient", _make_client(FakeResp(200, {"content-length": "8"}, [b"\x89PNG\r\n\x1a\n"]))):
            data = await srv._download_remote_image_bytes("https://manga.example/ok.png", {})
            assert data == b"\x89PNG\r\n\x1a\n"

    asyncio.run(_run())


# ==================== STEP 3, 4 & 5: HYGIENE, XSS & CORE ====================

def test_install_shisa_models_bat_ascii_crlf_and_no_stray_en():
    bat = REPO_ROOT / "install_shisa_models.bat"
    raw = bat.read_bytes()
    text = raw.decode("ascii")  # must be strictly ASCII
    assert "\r\n" in text
    assert text.replace("\r\n", "").find("\n") == -1, "All line endings must be CRLF"
    for line in text.split("\r\n"):
        if line.lstrip().lower().startswith("echo "):
            assert ">" not in line and "<" not in line, f"Unescaped redirect in echo: {line}"
    assert not (REPO_ROOT / "EN").exists(), "Stray EN file must not exist"


def test_version_and_manifest_least_privilege():
    srv = _load_server_module()
    manifest = json.loads((REPO_ROOT / "extension" / "manifest.json").read_text(encoding="utf-8"))
    assert srv.APP_VERSION == "6.1.0"
    assert manifest["version"] == srv.APP_VERSION
    assert "tabs" not in manifest["permissions"]
    assert "<all_urls>" not in manifest["host_permissions"]
    assert "<all_urls>" not in manifest["content_scripts"][0]["matches"]
    assert "http://127.0.0.1:5000/*" in manifest["content_scripts"][0]["exclude_matches"]


def test_web_studio_html_has_xss_sanitizers():
    html = (REPO_ROOT / "web_studio.html").read_text(encoding="utf-8")
    assert "&#39;" in html
    assert "function escapeJsAttr(" in html
    assert "function safeCssColor(" in html
    assert "escapeHtml(modelName)" in html


def test_sanitize_folder_name():
    srv = _load_server_module()
    assert srv._sanitize_folder_name("My:Manga*Ch1_RU") == "My Manga Ch1"
    assert srv._sanitize_folder_name("   ///   ", fallback="Web_Manga") == "Web_Manga"


def test_no_silent_except_exception_pass_after_logger():
    lines = (REPO_ROOT / "server.py").read_text(encoding="utf-8").splitlines(keepends=True)
    logger_ln = next(i for i, l in enumerate(lines, 1) if l.startswith('logger = logging.getLogger("manga_server")'))
    tree = ast.parse("".join(lines))
    leftovers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.lineno > logger_ln:
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                if isinstance(node.type, ast.Name) and node.type.id in ("Exception", "OSError"):
                    leftovers.append(node.lineno)
    assert leftovers == [], f"Found silent except:pass blocks after logger init: {leftovers}"



def test_api_modules_extracted_and_dead_classes_removed():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from api import schemas, security

    assert hasattr(security, "_mask_api_key")
    assert hasattr(security, "_validate_remote_image_url")
    assert hasattr(security, "is_path_within_allowed_roots")
    assert hasattr(schemas, "TranslateRequest")
    assert hasattr(schemas, "ServerSettingsRequest")

    cleaner_src = (REPO_ROOT / "cleaner.py").read_text(encoding="utf-8")
    renderer_src = (REPO_ROOT / "renderer.py").read_text(encoding="utf-8")
    for dead in ("class _DoubleConvBlock", "class CleaningRegionTask"):
        assert dead not in cleaner_src, f"Dead class still in cleaner.py: {dead}"
    for dead in ("class TypesetBlockSpec", "class LayoutSolution", "class AdaptiveTypesetter"):
        assert dead not in renderer_src, f"Dead class still in renderer.py: {dead}"

def test_pick_folder_returns_busy_when_lock_held():
    srv = _load_server_module()

    async def _run():
        assert srv._native_picker_lock.acquire(blocking=False)
        try:
            res = await asyncio.wait_for(
                srv.pick_folder_endpoint(target_lang="ru"),
                timeout=2.0,
            )
            assert res["status"] == "busy"
            assert "message" in res and res["message"]
        finally:
            srv._native_picker_lock.release()

    asyncio.run(_run())


def test_health_reports_initializing_before_engines_ready():
    srv = _load_server_module()

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

    async def _run():
        pending_event = asyncio.Event()
        with patch.object(srv, "ocr_engine", None),              patch.object(srv, "cleaner_engine", None),              patch.object(srv, "translator_engine", None),              patch.object(srv, "renderer_engine", None),              patch.object(srv, "_engines_ready_event", pending_event),              patch.object(srv.torch, "cuda", _FakeCuda, create=True):
            res = await srv.health_check()
            assert res["status"] == "initializing"
            assert res["engines_ready"] is False
            assert res["version"] == srv.APP_VERSION

    asyncio.run(_run())


if __name__ == "__main__":
    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in test_funcs:
        fn()
        print(f"  [PASS] {fn.__name__}")
    print(f"\nALL {len(test_funcs)} TESTS PASSED!")