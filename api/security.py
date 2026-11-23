"""
Модуль безопасности локального FastAPI-сервера Miko-chan Manga Studio.

Содержит:
- маскировку API-ключей (`_mask_api_key`);
- проверку доверенного локального Origin (`_is_trusted_local_origin`) и список привилегированных эндпоинтов;
- строгий белый список локальных директорий сессии (`_register_allowed_root`, `is_path_within_allowed_roots`);
- защиту от SSRF (блокировка локальных/приватных/link-local IP, DNS-проверка, проверка редиректов)
  и ограничение размера скачиваемых по URL изображений до 25 МБ (`_download_remote_image_bytes`).
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import threading
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import HTTPException

logger = logging.getLogger("manga_server")

_TRUSTED_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_TRUSTED_EXTENSION_SCHEMES = frozenset({"chrome-extension", "moz-extension"})

_PRIVILEGED_ENDPOINTS = {
    "/models",
    "/pull_status",
    "/model_storage",
    "/translate",
    "/translate_page_file",
    "/page_blocks",
    "/page_romaji",
    "/translate_manual_region",
    "/retranslate_single_block",
    "/auto_diamond_wrap",
    "/shorten_block_text",
    "/folder_status",
    "/folder_report",
    "/ux_telemetry/record",
    "/ux_telemetry/report",
    "/local_image",
    "/settings",
    "/delete_model",
    "/shutdown",
    "/open_folder",
    "/open_web_save_folder",
    "/pick_folder",
    "/pick_file",
    "/pick_web_save_folder",
    "/folder_inspect",
    "/translate_folder",
    "/folder_stop",
    "/save_edited_page",
    "/glossary",
    "/pull_model",
    "/model_stack",
    "/folder_find_replace",
    "/folder_clean_noise",
    "/open_folder_report",
    "/chapter_book",
    "/chapter_book/import",
    "/chapter_book/proofread",
    "/diagnostics",
    "/diagnostics/report",
    "/diagnostics/download_log",
    "/diagnostics/open_logs",
    "/diagnostics/clear",
    "/ux_telemetry/open_report",
    "/ux_telemetry/clear",
}

_SESSION_ALLOWED_ROOTS_LOCK = threading.Lock()
_SESSION_ALLOWED_ROOTS: set[Path] = set()
_MAX_REMOTE_IMAGE_BYTES = 25 * 1024 * 1024  # 25 MB limit for remote image_url downloads
_MAX_REMOTE_IMAGE_REDIRECTS = 5
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


def _mask_api_key(raw_key: str | None) -> str:
    k = (raw_key or "").strip()
    if not k:
        return ""
    return f"{k[:5]}...{k[-4:]}" if len(k) > 10 else "***"


def _is_trusted_local_origin(origin: str | None) -> bool:
    if not origin:
        return True
    orig_s = origin.strip()
    if not orig_s or orig_s.lower() == "null":
        return False
    try:
        parsed = urlparse(orig_s)
    except Exception:
        return False
    scheme = (parsed.scheme or "").lower()
    if scheme in _TRUSTED_EXTENSION_SCHEMES:
        return bool(parsed.netloc)
    if scheme in ("http", "https"):
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        return hostname in _TRUSTED_LOCAL_HOSTS
    return False


def _is_trusted_host_header(host: str | None) -> bool:
    """Проверяет заголовок Host для защиты от DNS-rebinding атак."""
    if not host or not host.strip():
        return True
    h = host.strip().lower()
    if h.startswith("["):
        end_idx = h.find("]")
        if end_idx == -1:
            return False
        after = h[end_idx + 1 :]
        if after and (not after.startswith(":") or not after[1:].isdigit()):
            return False
        hostname = h[1:end_idx]
    else:
        parts = h.split(":")
        if len(parts) > 2:
            return False
        if len(parts) == 2 and (not parts[1] or not parts[1].isdigit()):
            return False
        hostname = parts[0]
    hostname = hostname.strip().rstrip(".")
    return hostname in _TRUSTED_LOCAL_HOSTS


def _is_trusted_referer(referer: str | None) -> bool:
    """Проверяет заголовок Referer для запросов без Origin к локальному API."""
    if not referer or not referer.strip():
        return True
    try:
        parsed = urlparse(referer.strip())
    except Exception:
        return False
    if not parsed.scheme or not parsed.netloc:
        return False
    return _is_trusted_local_origin(f"{parsed.scheme}://{parsed.netloc}")


def _reject_unc_path(raw_path: str | None, field_name: str = "path") -> None:
    """Блокирует сетевые UNC-пути (NTLM-утечки) и NUL-байты во входных путях."""
    if raw_path is None:
        return
    s = str(raw_path).strip()
    if not s:
        return
    if "\x00" in s or s.startswith("\\\\") or s.startswith("//") or s.lower().startswith("\\\\?\\unc\\"):
        raise HTTPException(
            status_code=400,
            detail=f"Сетевые UNC-пути и недопустимые символы в '{field_name}' запрещены.",
        )


def _register_allowed_root(path_val: Path | str | None) -> None:
    if not path_val:
        return
    try:
        resolved = Path(str(path_val)).resolve()
        root_dir = resolved.parent if resolved.is_file() else resolved
        with _SESSION_ALLOWED_ROOTS_LOCK:
            _SESSION_ALLOWED_ROOTS.add(root_dir)
    except Exception as _exc:
        logger.debug("Ignored Exception in _register_allowed_root: %s", _exc)


def is_path_within_allowed_roots(img_path: Path, extra_roots: Iterable[Path | str | None] = ()) -> bool:
    try:
        resolved_img = Path(img_path).resolve()
    except Exception:
        return False

    allowed_roots: list[Path] = []
    for raw_root in extra_roots:
        if not raw_root:
            continue
        try:
            allowed_roots.append(Path(str(raw_root)).resolve())
        except Exception as _exc:
            logger.debug("Ignored Exception resolving allowed root: %s", _exc)

    with _SESSION_ALLOWED_ROOTS_LOCK:
        allowed_roots.extend(_SESSION_ALLOWED_ROOTS)

    for root in allowed_roots:
        try:
            if resolved_img.is_relative_to(root):
                return True
        except Exception:
            continue
    return False


def _is_blocked_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        if _is_blocked_ip(ip_obj.ipv4_mapped):
            return True
    if isinstance(ip_obj, ipaddress.IPv4Address) and ip_obj in _CGNAT_NETWORK:
        return True
    return bool(
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    )


def _validate_remote_image_url(raw_url: str) -> str:
    url_str = (raw_url or "").strip()
    parsed = urlparse(url_str)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError("Разрешены только HTTP и HTTPS ссылки на изображения.")
    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise ValueError("URL изображения не содержит корректного домена.")
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        raise ValueError("Доступ к локальным адресам запрещен.")

    try:
        literal_ip = ipaddress.ip_address(hostname.split("%", 1)[0])
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        if _is_blocked_ip(literal_ip):
            raise ValueError("Доступ к локальным и приватным IP-адресам запрещен.")
        return url_str

    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        addr_infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError(f"Не удалось разрешить домен '{hostname}': {exc}") from exc

    if not addr_infos:
        raise ValueError(f"Домен '{hostname}' не вернул IP-адресов.")

    for info in addr_infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_str = str(sockaddr[0]).split("%", 1)[0]
        try:
            resolved_ip = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise ValueError(f"Некорректный IP-адрес '{ip_str}' для '{hostname}'.") from exc
        if _is_blocked_ip(resolved_ip):
            raise ValueError("Доступ к локальным и приватным IP-адресам запрещен.")
    return url_str


async def _download_remote_image_bytes(
    image_url: str,
    headers: dict[str, str],
    max_bytes: int = _MAX_REMOTE_IMAGE_BYTES,
    max_redirects: int = _MAX_REMOTE_IMAGE_REDIRECTS,
) -> bytes:
    current_url = await asyncio.to_thread(_validate_remote_image_url, image_url)
    async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=30.0) as dl_client:
        for _hop in range(max_redirects + 1):
            async with dl_client.stream("GET", current_url, headers=headers) as dl_resp:
                exts = getattr(dl_resp, "extensions", None)
                net_stream = exts.get("network_stream") if isinstance(exts, dict) else None
                if net_stream is not None and hasattr(net_stream, "get_extra_info"):
                    peer = net_stream.get_extra_info("peername")
                    if peer and isinstance(peer, (tuple, list)) and peer[0]:
                        peer_ip = ipaddress.ip_address(str(peer[0]).split("%", 1)[0])
                        if _is_blocked_ip(peer_ip):
                            raise ValueError("Доступ к локальным и приватным IP-адресам запрещен.")
                if dl_resp.status_code in (301, 302, 303, 307, 308):
                    location = dl_resp.headers.get("location")
                    if not location:
                        raise ValueError("Сервер вернул редирект без заголовка Location.")
                    if _hop >= max_redirects:
                        raise ValueError("Превышено максимальное количество редиректов.")
                    next_url = urljoin(current_url, location)
                    current_url = await asyncio.to_thread(_validate_remote_image_url, next_url)
                    continue

                dl_resp.raise_for_status()
                content_len_hdr = dl_resp.headers.get("content-length")
                if content_len_hdr is not None:
                    try:
                        if int(content_len_hdr) > max_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=f"Размер изображения по URL превышает лимит {max_bytes // (1024 * 1024)} МБ.",
                            )
                    except ValueError:
                        pass

                buf = bytearray()
                async for chunk in dl_resp.aiter_bytes(chunk_size=65536):
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Размер изображения по URL превышает лимит {max_bytes // (1024 * 1024)} МБ.",
                        )
                return bytes(buf)
    raise ValueError("Превышено максимальное количество редиректов.")
