param(
    [ValidateSet("Lite", "Full", "OnnxPack", "All")]
    [string]$Mode = "All",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if (Test-Path $CargoBin) {
    $env:Path = "$CargoBin;$env:Path"
}

if (-not $SkipBuild) {
    Write-Host "[1/4] Сборка релизного бинарника MikoStudio.exe (Rust)..." -ForegroundColor Cyan
    & cargo build --release --bin miko-studio-rs
    if ($LASTEXITCODE -ne 0) {
        throw "Ошибка сборки cargo build --release"
    }
}

$SrcExe = Join-Path $Root "target\release\miko-studio-rs.exe"
if (-not (Test-Path $SrcExe)) {
    throw "Не найден собранный бинарник: $SrcExe"
}

$DistRoot = Join-Path $Root "dist"
New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null

$ReadmeLite = @"
======================================================================
  Miko-chan Manga Studio (Rust Portable Edition)
======================================================================

КАК ЗАПУСТИТЬ (В 1 КЛИК):
1. Распакуйте эту папку в любое удобное место (например D:\MikoStudio).
2. Запустите файл MikoStudio.exe двойным кликом.
3. Откроется браузер с окном «Быстрая настройка Miko-chan Studio»:
   - нажмите кнопку «Скачать и настроить всё автоматически (в 1 клик)»;
   - программа сама определит вашу видеокарту (NVIDIA CUDA или CPU),
     скачает официальный рантайм Microsoft / NVIDIA, шрифты и 6 моделей ИИ.
4. Для перевода укажите облачный API-ключ (DeepSeek / ModuleAPI / OpenRouter)
   или нажмите кнопку установки бесплатной локальной Ollama.

Приятного перевода манги!
"@

$ReadmeFull = @"
======================================================================
  Miko-chan Manga Studio (Full CUDA Offline Portable Edition)
======================================================================

Версия «ВСЁ ВКЛЮЧЕНО» (уже содержит все 6 нейросетей ONNX, комиксные
шрифты и ускоритель NVIDIA CUDA 13 + cuDNN 9 — интернет для моделей не нужен!).

КАК ЗАПУСТИТЬ:
1. Распакуйте архив в любую папку на диске.
2. Запустите MikoStudio.exe двойным кликом — студия сразу откроется в браузере!
3. В настройках укажите облачный API-ключ или выберите локальную модель Ollama.
"@

# ------------------------------------------------------------------
# 1. Lite Portable (~6-7 MB): только самодостаточный MikoStudio.exe
# ------------------------------------------------------------------
if ($Mode -in @("Lite", "All")) {
    Write-Host "[2/4] Упаковка лёгкой умной портативки (MikoStudio-Portable-Lite-win64.zip)..." -ForegroundColor Cyan
    $LiteStage = Join-Path $DistRoot "stage_lite\MikoStudio-Portable"
    if (Test-Path (Join-Path $DistRoot "stage_lite")) {
        Remove-Item -Recurse -Force (Join-Path $DistRoot "stage_lite")
    }
    New-Item -ItemType Directory -Force -Path $LiteStage | Out-Null
    Copy-Item -Force $SrcExe (Join-Path $LiteStage "MikoStudio.exe")
    Set-Content -Path (Join-Path $LiteStage "ПРОЧТИ_МЕНЯ.txt") -Value $ReadmeLite -Encoding UTF8

    $LiteZip = Join-Path $DistRoot "MikoStudio-Portable-Lite-win64.zip"
    if (Test-Path $LiteZip) { Remove-Item -Force $LiteZip }
    & tar.exe -a -cf $LiteZip -C (Join-Path $DistRoot "stage_lite") "MikoStudio-Portable"
    if ($LASTEXITCODE -ne 0) { throw "Ошибка создания архива $LiteZip" }
    Remove-Item -Recurse -Force (Join-Path $DistRoot "stage_lite")

    $LiteMb = [math]::Round((Get-Item $LiteZip).Length / 1MB, 2)
    Write-Host "  [OK] $LiteZip ($LiteMb МБ)" -ForegroundColor Green
}

# ------------------------------------------------------------------
# 2. One-Time GitHub Release staging folder (onnx-models-v1)
# ------------------------------------------------------------------
if ($Mode -in @("OnnxPack", "All")) {
    Write-Host "[3/4] Подготовка пакета кастомных моделей для релиза 'onnx-models-v1'..." -ForegroundColor Cyan
    $OnnxPackDir = Join-Path $DistRoot "onnx-models-v1"
    if (Test-Path $OnnxPackDir) { Remove-Item -Recurse -Force $OnnxPackDir }
    New-Item -ItemType Directory -Force -Path $OnnxPackDir | Out-Null

    $CustomModels = @(
        "rtdetr_v2_comic.onnx",
        "yolo_bubble_det.onnx",
        "yolo26s_seg.onnx",
        "comic_text_segmenter.onnx",
        "anime_lama.onnx"
    )
    $HashLines = @()
    foreach ($m in $CustomModels) {
        $src = Join-Path $Root "models\onnx\$m"
        if (-not (Test-Path $src)) {
            Write-Warning "Модель не найдена локально (пропущена): $src"
            continue
        }
        $dst = Join-Path $OnnxPackDir $m
        try {
            New-Item -ItemType HardLink -Path $dst -Target $src -ErrorAction Stop | Out-Null
        } catch {
            Copy-Item -Force $src $dst
        }
        $hash = (Get-FileHash -Algorithm SHA256 $src).Hash.ToLowerInvariant()
        $HashLines += "$hash  $m"
    }
    Set-Content -Path (Join-Path $OnnxPackDir "SHA256SUMS.txt") -Value ($HashLines -join "`n") -Encoding UTF8
    Write-Host "  [OK] Папка для одноразового GitHub-релиза 'onnx-models-v1': $OnnxPackDir" -ForegroundColor Green
}

# ------------------------------------------------------------------
# 3. Full Offline CUDA Portable (~1.2 GB): всё включено без мусора PyTorch
# ------------------------------------------------------------------
if ($Mode -in @("Full", "All")) {
    Write-Host "[4/4] Упаковка полной офлайн-портативки (MikoStudio-Portable-Full-CUDA-win64.zip)..." -ForegroundColor Cyan
    $FullRoot = Join-Path $DistRoot "stage_full"
    $FullStage = Join-Path $FullRoot "MikoStudio-Portable-Full-CUDA"
    if (Test-Path $FullRoot) { Remove-Item -Recurse -Force $FullRoot }

    $StageFonts = Join-Path $FullStage "fonts"
    $StageOnnx = Join-Path $FullStage "models\onnx"
    $StageOrt = Join-Path $FullStage "models\ort_runtime"
    New-Item -ItemType Directory -Force -Path $StageFonts, $StageOnnx, $StageOrt | Out-Null

    Copy-Item -Force $SrcExe (Join-Path $FullStage "MikoStudio.exe")
    Set-Content -Path (Join-Path $FullStage "ПРОЧТИ_МЕНЯ.txt") -Value $ReadmeFull -Encoding UTF8

    if (Test-Path (Join-Path $Root "fonts")) {
        Get-ChildItem (Join-Path $Root "fonts") -Include *.ttf,*.otf,*.ttc -Recurse | ForEach-Object {
            Copy-Item -Force $_.FullName (Join-Path $StageFonts $_.Name)
        }
    }

    $AllOnnx = @(
        "rtdetr_v2_comic.onnx",
        "yolo_bubble_det.onnx",
        "yolo26s_seg.onnx",
        "comic_text_segmenter.onnx",
        "manga_ocr_nar_small.onnx",
        "anime_lama.onnx"
    )
    foreach ($m in $AllOnnx) {
        $src = Join-Path $Root "models\onnx\$m"
        if (-not (Test-Path $src)) { throw "Отсутствует ONNX модель для Full-сборки: $src" }
        $dst = Join-Path $StageOnnx $m
        try {
            New-Item -ItemType HardLink -Path $dst -Target $src -ErrorAction Stop | Out-Null
        } catch {
            Copy-Item -Force $src $dst
        }
    }

    $OrtDlls = @(
        "onnxruntime.dll",
        "onnxruntime_providers_shared.dll",
        "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_tensorrt.dll"
    )
    foreach ($d in $OrtDlls) {
        $src = Join-Path $Root "models\ort_runtime\$d"
        if (Test-Path $src) {
            $dst = Join-Path $StageOrt $d
            try { New-Item -ItemType HardLink -Path $dst -Target $src -ErrorAction Stop | Out-Null } catch { Copy-Item -Force $src $dst }
        }
    }

    # Находим чистые библиотеки CUDA 13 + cuDNN 9 (без лишних 1.1 ГБ torch_cpu/torch_cuda!)
    $CudaSourceDir = Join-Path $Root "models\ort_runtime"
    if (-not (Test-Path (Join-Path $CudaSourceDir "cublas64_13.dll"))) {
        $pyCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($pyCmd) {
            $tl = (& python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')" 2>$null).Trim()
            if ($tl -and (Test-Path $tl)) { $CudaSourceDir = $tl }
        }
    }

    $CleanCudaDlls = @(
        "cudart64_13.dll",
        "cublas64_13.dll",
        "cublasLt64_13.dll",
        "cufft64_12.dll",
        "cudnn64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_graph64_9.dll",
        "cudnn_heuristic64_9.dll",
        "cudnn_ops64_9.dll",
        "nvrtc64_130_0.dll",
        "nvrtc-builtins64_130.dll",
        "zlibwapi.dll"
    )
    foreach ($d in $CleanCudaDlls) {
        $src = Join-Path $CudaSourceDir $d
        if (Test-Path $src) {
            $dst = Join-Path $StageOrt $d
            if (-not (Test-Path $dst)) {
                try { New-Item -ItemType HardLink -Path $dst -Target $src -ErrorAction Stop | Out-Null } catch { Copy-Item -Force $src $dst }
            }
        }
    }

    $FullZip = Join-Path $DistRoot "MikoStudio-Portable-Full-CUDA-win64.zip"
    if (Test-Path $FullZip) { Remove-Item -Force $FullZip }
    Write-Host "  Сжатие архива (может занять ~1-2 минуты)..." -ForegroundColor DarkGray
    & tar.exe -a -cf $FullZip -C $FullRoot "MikoStudio-Portable-Full-CUDA"
    if ($LASTEXITCODE -ne 0) { throw "Ошибка создания архива $FullZip" }
    Remove-Item -Recurse -Force $FullRoot

    $FullMb = [math]::Round((Get-Item $FullZip).Length / 1MB, 2)
    Write-Host "  [OK] $FullZip ($FullMb МБ)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Готово! Все файлы релиза лежат в папке: $DistRoot" -ForegroundColor Green
