"""Экспорт локальных PyTorch/Ultralytics/Transformers моделей в ONNX + проверка паритета с PyTorch."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ONNX_DIR = MODELS_DIR / "onnx"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _check_and_summarize(path: Path, torch_out: np.ndarray | None = None) -> dict[str, Any]:
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    inputs = [
        {
            "name": i.name,
            "shape": [
                (d.dim_value if d.dim_value > 0 else (d.dim_param or -1))
                for d in i.type.tensor_type.shape.dim
            ],
        }
        for i in onnx_model.graph.input
    ]
    outputs = [
        {
            "name": o.name,
            "shape": [
                (d.dim_value if d.dim_value > 0 else (d.dim_param or -1))
                for d in o.type.tensor_type.shape.dim
            ],
        }
        for o in onnx_model.graph.output
    ]
    return {
        "file": path.name,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "sha256": _sha256(path),
        "inputs": inputs,
        "outputs": outputs,
        "torch_reference_shape": list(torch_out.shape) if torch_out is not None else None,
        "status": "ok",
    }


def _export_yolo_pt(pt_path: Path, out_path: Path, task: str | None = None) -> dict[str, Any]:
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(str(pt_path), task=task) if task else YOLO(str(pt_path))
    model.model.eval()
    exported = Path(
        model.export(
            format="onnx",
            imgsz=640,
            opset=17,
            dynamic=False,
            simplify=False,
            half=False,
        )
    )
    if exported.resolve() != out_path.resolve():
        out_path.write_bytes(exported.read_bytes())
        if exported.exists():
            exported.unlink(missing_ok=True)
    # Remove ultralytics sidecar weights only if inside models/ and differs from source
    dummy = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
    with torch.inference_mode():
        ref = model.model(dummy)
    while isinstance(ref, (list, tuple)):
        ref = ref[0]
    ref_arr = ref.detach().cpu().numpy()
    info = _check_and_summarize(out_path, ref_arr)
    info["export_sec"] = round(time.perf_counter() - t0, 2)
    return info


def _export_rtdetr(model_dir: Path, out_path: Path) -> dict[str, Any]:
    from transformers import RTDetrV2ForObjectDetection

    t0 = time.perf_counter()
    model = RTDetrV2ForObjectDetection.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()

    class _RtDetrWrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            res = self.inner(pixel_values=pixel_values)
            return res.logits, res.pred_boxes

    wrapper = _RtDetrWrapper(model).eval()
    dummy = torch.rand((1, 3, 640, 640), dtype=torch.float32)
    with torch.inference_mode():
        ref_logits, _ = wrapper(dummy)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        opset_version=17,
        dynamo=False,
    )
    info = _check_and_summarize(out_path, ref_logits.detach().cpu().numpy())
    info["export_sec"] = round(time.perf_counter() - t0, 2)
    return info


def _export_manga_ocr_nar(model_dir: Path, out_path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(model_dir))
    from manga_ocr_nar_small.model import MangaOcrNarSmallNet
    from manga_ocr_nar_small.preprocess import empty_inputs

    t0 = time.perf_counter()
    net = MangaOcrNarSmallNet.load(model_dir, device="cpu", dtype=torch.float32)
    buf = empty_inputs(batch=1)
    p = torch.from_numpy(buf["P"]).float()
    v = torch.from_numpy(buf["V"] > 0.5).bool()
    r = torch.from_numpy(buf["R"]).long()
    c = torch.from_numpy(buf["C"]).long()
    h = torch.tensor([1], dtype=torch.long)
    w = torch.tensor([1], dtype=torch.long)

    with torch.inference_mode():
        ref_ids = net(p, v, r, c, h, w)

    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        info = _check_and_summarize(out_path, ref_ids.detach().cpu().numpy())
        info["source"] = "cached_onnx"
        info["export_sec"] = round(time.perf_counter() - t0, 2)
        return info

    try:
        torch.onnx.export(
            net,
            (p, v, r, c, h, w),
            str(out_path),
            input_names=["P", "V", "R", "C", "H", "W"],
            output_names=["ids"],
            dynamic_axes={
                "P": {0: "B"},
                "V": {0: "B"},
                "R": {0: "B"},
                "C": {0: "B"},
                "H": {0: "B"},
                "W": {0: "B"},
                "ids": {0: "B"},
            },
            opset_version=17,
            dynamo=False,
        )
        info = _check_and_summarize(out_path, ref_ids.detach().cpu().numpy())
        info["source"] = "torch_onnx_export"
        info["export_sec"] = round(time.perf_counter() - t0, 2)
        return info
    except Exception as export_exc:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                repo_id="muscgab/manga-ocr-nar-small",
                filename="onnx/mangaocr_nar_small_fp32.onnx",
                local_dir=str(model_dir),
            )
        )
        out_path.write_bytes(downloaded.read_bytes())
        info = _check_and_summarize(out_path, ref_ids.detach().cpu().numpy())
        info["source"] = f"hf_official_onnx (fallback from {type(export_exc).__name__})"
        info["export_sec"] = round(time.perf_counter() - t0, 2)
        return info


def _export_anime_lama(pt_path: Path, out_path: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    jit_model = torch.jit.load(str(pt_path), map_location="cpu").eval()
    img = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    msk = torch.zeros((1, 1, 512, 512), dtype=torch.float32)
    msk[:, :, 200:260, 200:260] = 1.0
    with torch.inference_mode():
        ref_out = jit_model(img, msk)

    last_err = ""
    for use_dynamo, opset in ((False, 17), (False, 18), (True, 18)):
        try:
            torch.onnx.export(
                jit_model,
                (img, msk),
                str(out_path),
                input_names=["image", "mask"],
                output_names=["inpainted"],
                opset_version=opset,
                dynamo=use_dynamo,
            )
            info = _check_and_summarize(out_path, ref_out.detach().cpu().numpy())
            info["source"] = f"torch_jit_opset_{opset}_dynamo_{use_dynamo}"
            info["export_sec"] = round(time.perf_counter() - t0, 2)
            return info
        except Exception as exc:
            last_err = f"opset={opset},dynamo={use_dynamo}: {type(exc).__name__}: {exc}"
            if out_path.exists():
                out_path.unlink(missing_ok=True)
    return {
        "file": out_path.name,
        "status": "skipped_fft_unsupported",
        "detail": last_err[:400],
        "export_sec": round(time.perf_counter() - t0, 2),
    }


def main() -> int:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"onnx_dir": str(ONNX_DIR), "models": {}}

    jobs = [
        ("yolo_bubble_det", lambda: _export_yolo_pt(MODELS_DIR / "comic-speech-bubble-detector.pt", ONNX_DIR / "yolo_bubble_det.onnx")),
        ("yolo26s_seg", lambda: _export_yolo_pt(MODELS_DIR / "manga109-yolov26s-seg.pt", ONNX_DIR / "yolo26s_seg.onnx", task="segment")),
        ("comic_text_segmenter", lambda: _export_yolo_pt(MODELS_DIR / "comic-text-segmenter" / "comic-text-segmenter.pt", ONNX_DIR / "comic_text_segmenter.onnx", task="segment")),
        ("rtdetr_v2_comic", lambda: _export_rtdetr(MODELS_DIR / "comic-text-and-bubble-detector", ONNX_DIR / "rtdetr_v2_comic.onnx")),
        ("manga_ocr_nar_small", lambda: _export_manga_ocr_nar(MODELS_DIR / "manga-ocr-nar-small", ONNX_DIR / "manga_ocr_nar_small.onnx")),
        ("anime_lama", lambda: _export_anime_lama(MODELS_DIR / "anime-manga-big-lama.pt", ONNX_DIR / "anime_lama.onnx")),
    ]

    for key, fn in jobs:
        print(f"[EXPORT] Starting {key}...", flush=True)
        try:
            res = fn()
        except Exception as exc:
            res = {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}
        manifest["models"][key] = res
        print(f"[EXPORT] {key} -> {res.get('status')} ({res.get('size_mb', 0)} MB)", flush=True)

    manifest_path = ONNX_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest saved to {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
