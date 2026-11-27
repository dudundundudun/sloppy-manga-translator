param(
    [string]$HostAddr = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Rebuild,
    [switch]$OpenBrowser,
    [switch]$SelfCheck
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if (Test-Path $CargoBin) {
    $env:Path = "$CargoBin;$env:Path"
}

if (-not $env:MIKO_CUDA_DLL_DIR) {
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd) {
        try {
            $torchLib = (& python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')" 2>$null).Trim()
            if ($torchLib -and (Test-Path $torchLib)) {
                $env:MIKO_CUDA_DLL_DIR = $torchLib
            }
        } catch {
        }
    }
}

$ExePath = Join-Path $Root "target\release\miko-studio-rs.exe"
if ($Rebuild -or -not (Test-Path $ExePath)) {
    Write-Host "Building miko-studio-rs (release)..." -ForegroundColor Cyan
    & cargo build --release --bin miko-studio-rs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($SelfCheck) {
    & $ExePath --base-dir $Root --self-check
    exit $LASTEXITCODE
}

$TargetUrl = "http://$HostAddr`:$Port"
Write-Host "Starting Miko-chan Manga Studio (Rust) at $TargetUrl" -ForegroundColor Green

# Ensure local Ollama daemon is running in the background for instant 18+ censorship fallback
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaCmd) {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 1 -ErrorAction Stop
    } catch {
        Write-Host "Starting background Ollama server for local fallback..." -ForegroundColor DarkCyan
        Start-Process -FilePath $ollamaCmd.Source -ArgumentList "serve" -WindowStyle Hidden
    }
}
if ($env:MIKO_CUDA_DLL_DIR) {
    Write-Host "CUDA DLL dir: $env:MIKO_CUDA_DLL_DIR" -ForegroundColor DarkGray
}
if ($OpenBrowser) {
    Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile -Command `"Start-Sleep -Seconds 2; Start-Process '$TargetUrl'`""
}
& $ExePath --base-dir $Root --host $HostAddr --port $Port --no-browser
