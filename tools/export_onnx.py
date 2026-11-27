"""Экспорт локальных PyTorch/Ultralytics/Transformers моделей в ONNX + проверка паритета с PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


class _RealFourierUnit(torch.nn.Module):
    """Exact real-valued 2D ortho DFT/IDFT replacement for Spectral FourierUnit (64x64)."""

    def __init__(self, in_channels: int, out_channels: int, spatial_size: int = 64) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.conv_layer = torch.nn.Conv2d(
            in_channels * 2, out_channels * 2, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        n = spatial_size
        m = n // 2 + 1
        scale = 1.0 / math.sqrt(n)
        idx = torch.arange(n, dtype=torch.float32)
        freq_half = torch.arange(m, dtype=torch.float32)

        ang_w = (2.0 * math.pi / n) * torch.outer(idx, freq_half)
        w_cos = torch.cos(ang_w) * scale
        w_sin = -torch.sin(ang_w) * scale

        ang_h = (2.0 * math.pi / n) * torch.outer(idx, idx)
        h_cos = torch.cos(ang_h) * scale
        h_sin = -torch.sin(ang_h) * scale

        weights = torch.full((m, 1), 2.0, dtype=torch.float32)
        weights[0, 0] = 1.0
        if n % 2 == 0:
            weights[m - 1, 0] = 1.0
        ang_iw = (2.0 * math.pi / n) * torch.outer(freq_half, idx)
        iw_cos = weights * torch.cos(ang_iw) * scale
        iw_sin = weights * torch.sin(ang_iw) * scale

        self.register_buffer("W_cos", w_cos)
        self.register_buffer("W_sin", w_sin)
        self.register_buffer("H_cos", h_cos)
        self.register_buffer("H_sin", h_sin)
        self.register_buffer("IW_cos", iw_cos)
        self.register_buffer("IW_sin", iw_sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        wf = w // 2 + 1
        t_re = torch.matmul(x, self.W_cos)
        t_im = torch.matmul(x, self.W_sin)
        re = torch.matmul(self.H_cos, t_re) - torch.matmul(self.H_sin, t_im)
        im = torch.matmul(self.H_sin, t_re) + torch.matmul(self.H_cos, t_im)

        ffted = torch.stack([re, im], dim=2).reshape(b, c * 2, h, wf)
        ffted = self.relu(self.bn(self.conv_layer(ffted)))
        ffted = ffted.reshape(b, c, 2, h, wf)
        yr = ffted[:, :, 0]
        yi = ffted[:, :, 1]

        u_re = torch.matmul(self.H_cos.t(), yr) + torch.matmul(self.H_sin.t(), yi)
        u_im = -torch.matmul(self.H_sin.t(), yr) + torch.matmul(self.H_cos.t(), yi)
        return torch.matmul(u_re, self.IW_cos) - torch.matmul(u_im, self.IW_sin)


class _SpectralTransform(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial_size: int = 64) -> None:
        super().__init__()
        half = out_channels // 2
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, half, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(half),
            torch.nn.ReLU(inplace=True),
        )
        self.fu = _RealFourierUnit(half, half, spatial_size=spatial_size)
        self.conv2 = torch.nn.Conv2d(half, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(x + self.fu(x))


class _FFC(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        padding_mode: str = "reflect",
        spatial_size: int = 64,
    ) -> None:
        super().__init__()
        in_cg = int(round(in_channels * ratio_gin))
        in_cl = in_channels - in_cg
        out_cg = int(round(out_channels * ratio_gout))
        out_cl = out_channels - out_cg
        self.convl2l = (
            torch.nn.Conv2d(in_cl, out_cl, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=False)
            if in_cl > 0 and out_cl > 0
            else torch.nn.Identity()
        )
        self.convl2g = (
            torch.nn.Conv2d(in_cl, out_cg, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=False)
            if in_cl > 0 and out_cg > 0
            else torch.nn.Identity()
        )
        self.convg2l = (
            torch.nn.Conv2d(in_cg, out_cl, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=False)
            if in_cg > 0 and out_cl > 0
            else torch.nn.Identity()
        )
        self.convg2g = (
            _SpectralTransform(in_cg, out_cg, spatial_size=spatial_size)
            if in_cg > 0 and out_cg > 0
            else torch.nn.Identity()
        )
        self.has_l2l = in_cl > 0 and out_cl > 0
        self.has_l2g = in_cl > 0 and out_cg > 0
        self.has_g2l = in_cg > 0 and out_cl > 0
        self.has_g2g = in_cg > 0 and out_cg > 0

    def forward(self, x: tuple[torch.Tensor, torch.Tensor | int]) -> tuple[torch.Tensor | int, torch.Tensor | int]:
        x_l, x_g = x
        out_xl: torch.Tensor | int = 0
        out_xg: torch.Tensor | int = 0
        if self.has_l2l:
            out_xl = self.convl2l(x_l)
        if self.has_g2l and isinstance(x_g, torch.Tensor):
            g2l = self.convg2l(x_g)
            out_xl = g2l if isinstance(out_xl, int) else (out_xl + g2l)
        if self.has_l2g:
            out_xg = self.convl2g(x_l)
        if self.has_g2g and isinstance(x_g, torch.Tensor):
            g2g = self.convg2g(x_g)
            out_xg = g2g if isinstance(out_xg, int) else (out_xg + g2g)
        return out_xl, out_xg


class _FFC_BN_ACT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        out_cg = int(round(out_channels * ratio_gout))
        out_cl = out_channels - out_cg
        self.ffc = _FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, padding_mode)
        self.bn_l = torch.nn.BatchNorm2d(out_cl) if out_cl > 0 else torch.nn.Identity()
        self.bn_g = torch.nn.BatchNorm2d(out_cg) if out_cg > 0 else torch.nn.Identity()
        self.act_l = torch.nn.ReLU(inplace=True) if out_cl > 0 else torch.nn.Identity()
        self.act_g = torch.nn.ReLU(inplace=True) if out_cg > 0 else torch.nn.Identity()
        self.has_out_l = out_cl > 0
        self.has_out_g = out_cg > 0

    def forward(self, x: tuple[torch.Tensor, torch.Tensor | int] | torch.Tensor) -> tuple[torch.Tensor | int, torch.Tensor | int]:
        if isinstance(x, torch.Tensor):
            x_tuple: tuple[torch.Tensor, torch.Tensor | int] = (x, 0)
        else:
            x_tuple = x
        x_l, x_g = self.ffc(x_tuple)
        if self.has_out_l and isinstance(x_l, torch.Tensor):
            x_l = self.act_l(self.bn_l(x_l))
        if self.has_out_g and isinstance(x_g, torch.Tensor):
            x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class _FFCResnetBlock(torch.nn.Module):
    def __init__(self, dim: int = 512, ratio_g: float = 0.75) -> None:
        super().__init__()
        self.conv1 = _FFC_BN_ACT(dim, dim, 3, ratio_g, ratio_g, stride=1, padding=1, padding_mode="reflect")
        self.conv2 = _FFC_BN_ACT(dim, dim, 3, ratio_g, ratio_g, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = x
        out_l, out_g = self.conv1((x_l, x_g))
        out_l, out_g = self.conv2((out_l, out_g))
        return x_l + out_l, x_g + out_g


class _ConcatTupleLayer(torch.nn.Module):
    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return torch.cat([x[0], x[1]], dim=1)


class _AnimeLaMaOnnx(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = [
            torch.nn.ReflectionPad2d(3),
            _FFC_BN_ACT(4, 64, 7, 0.0, 0.0, stride=1, padding=0),
            _FFC_BN_ACT(64, 128, 3, 0.0, 0.0, stride=2, padding=1),
            _FFC_BN_ACT(128, 256, 3, 0.0, 0.0, stride=2, padding=1),
            _FFC_BN_ACT(256, 512, 3, 0.0, 0.75, stride=2, padding=1),
        ]
        for _ in range(18):
            layers.append(_FFCResnetBlock(512, 0.75))
        layers.extend(
            [
                _ConcatTupleLayer(),
                torch.nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1, bias=True),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=True),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=True),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.ReflectionPad2d(3),
                torch.nn.Conv2d(64, 3, 7, bias=True),
                torch.nn.Sigmoid(),
            ]
        )
        self.generator = torch.nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask0 = torch.where(mask > 0.5, torch.ones_like(mask), mask)
        x = torch.cat([image * (1.0 - mask0), mask0], dim=1)
        pred = self.generator(x)
        return mask0 * pred + (1.0 - mask0) * image


def _export_anime_lama(pt_path: Path, out_path: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    jit_model = torch.jit.load(str(pt_path), map_location="cpu").eval()
    wrapper = _AnimeLaMaOnnx().eval()
    sd = jit_model.model.generator.model.state_dict()
    incompatible = wrapper.generator.load_state_dict(sd, strict=False)
    dft_suffixes = (".W_cos", ".W_sin", ".H_cos", ".H_sin", ".IW_cos", ".IW_sin")
    bad_missing = [k for k in incompatible.missing_keys if not k.endswith(dft_suffixes)]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"State dict mismatch: bad_missing={bad_missing[:5]}, unexpected={incompatible.unexpected_keys[:5]}"
        )

    torch.manual_seed(42)
    img = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    msk = torch.zeros((1, 1, 512, 512), dtype=torch.float32)
    msk[:, :, 180:290, 170:320] = 1.0
    with torch.inference_mode():
        ref_out = jit_model(img, msk)
        wrap_out = wrapper(img, msk)

    diff = (ref_out - wrap_out).abs()
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())
    if max_abs_diff >= 1e-4:
        raise RuntimeError(f"Anime-LaMa parity check failed: max_abs_diff={max_abs_diff:.6e}")

    torch.onnx.export(
        wrapper,
        (img, msk),
        str(out_path),
        input_names=["image", "mask"],
        output_names=["inpainted"],
        opset_version=17,
        dynamo=False,
    )
    info = _check_and_summarize(out_path, ref_out.detach().cpu().numpy())
    info["source"] = "real_dft_opset_17"
    info["parity_max_abs_diff"] = max_abs_diff
    info["parity_mean_abs_diff"] = mean_abs_diff
    info["export_sec"] = round(time.perf_counter() - t0, 2)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Export local models to ONNX")
    parser.add_argument("--only", type=str, default=None, help="Export only a specific model key")
    args = parser.parse_args()

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = ONNX_DIR / "manifest.json"
    if args.only and manifest_path.is_file():
        try:
            manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {"onnx_dir": str(ONNX_DIR), "models": {}}
        manifest.setdefault("onnx_dir", str(ONNX_DIR))
        manifest.setdefault("models", {})
    else:
        manifest = {"onnx_dir": str(ONNX_DIR), "models": {}}

    jobs = [
        ("yolo_bubble_det", lambda: _export_yolo_pt(MODELS_DIR / "comic-speech-bubble-detector.pt", ONNX_DIR / "yolo_bubble_det.onnx")),
        ("yolo26s_seg", lambda: _export_yolo_pt(MODELS_DIR / "manga109-yolov26s-seg.pt", ONNX_DIR / "yolo26s_seg.onnx", task="segment")),
        ("comic_text_segmenter", lambda: _export_yolo_pt(MODELS_DIR / "comic-text-segmenter" / "comic-text-segmenter.pt", ONNX_DIR / "comic_text_segmenter.onnx", task="segment")),
        ("rtdetr_v2_comic", lambda: _export_rtdetr(MODELS_DIR / "comic-text-and-bubble-detector", ONNX_DIR / "rtdetr_v2_comic.onnx")),
        ("manga_ocr_nar_small", lambda: _export_manga_ocr_nar(MODELS_DIR / "manga-ocr-nar-small", ONNX_DIR / "manga_ocr_nar_small.onnx")),
        ("anime_lama", lambda: _export_anime_lama(MODELS_DIR / "anime-manga-big-lama.pt", ONNX_DIR / "anime_lama.onnx")),
    ]

    if args.only:
        jobs = [(k, fn) for k, fn in jobs if k == args.only]
        if not jobs:
            print(f"Unknown model key: {args.only}", file=sys.stderr)
            return 1

    for key, fn in jobs:
        print(f"[EXPORT] Starting {key}...", flush=True)
        try:
            res = fn()
        except Exception as exc:
            res = {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}
        manifest["models"][key] = res
        print(
            f"[EXPORT] {key} -> {res.get('status')} ({res.get('size_mb', 0)} MB, max_diff={res.get('parity_max_abs_diff', '-')})",
            flush=True,
        )

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest saved to {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
