param(
    [string]$Tag = "onnx-models-v1",
    [string]$Repo = "dudundundudun/sloppy-manga-translator"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$PackDir = Join-Path $Root "dist\onnx-models-v1"
if (-not (Test-Path $PackDir)) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "tools\package_release.ps1") -Mode OnnxPack -SkipBuild
}

$Files = Get-ChildItem $PackDir -File
if ($Files.Count -eq 0) {
    throw "В папке $PackDir нет файлов моделей для загрузки!"
}

$GhCmd = Get-Command gh -ErrorAction SilentlyContinue
if ($GhCmd) {
    Write-Host "Найден GitHub CLI (gh). Загружаем одноразовый релиз '$Tag' в $Repo..." -ForegroundColor Cyan
    $FilePaths = $Files | ForEach-Object { $_.FullName }
    & gh release view $Tag --repo $Repo 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        & gh release upload $Tag @FilePaths --clobber --repo $Repo
    } else {
        & gh release create $Tag @FilePaths --repo $Repo --title "Miko-chan Studio Custom ONNX Models Pack (v1)" --notes "Вечный пакет кастомно сконвертированных ONNX-моделей (Anime-LaMa Real-DFT, RT-DETR-v2, YOLOv8m, YOLO26s-seg, Comic-Text-Segmenter) для автоматической докачки портативной Rust-версии MikoStudio.exe."
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Модели успешно опубликованы в https://github.com/$Repo/releases/tag/$Tag !" -ForegroundColor Green
        exit 0
    }
}

Write-Host "==================================================================" -ForegroundColor Yellow
Write-Host "  Одноразовая публикация пакета моделей '$Tag' на GitHub" -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Yellow
Write-Host "1. Сейчас откроется страница создания релиза на GitHub и папка с 5 файлами."
Write-Host "2. Просто перетащите мышкой все файлы из открывшейся папки dist\onnx-models-v1"
Write-Host "   в поле 'Attach binaries by dropping them here' на странице GitHub"
Write-Host "   и нажмите зелёную кнопку 'Publish release'!"
Write-Host "   (Это делается ВСЕГО ОДИН РАЗ — при обновлениях кода повторять не нужно!)"
Write-Host ""

Start-Process explorer.exe $PackDir
Start-Process "https://github.com/$Repo/releases/new?tag=$Tag&title=Miko-chan+ONNX+Models+Pack+(v1)"
