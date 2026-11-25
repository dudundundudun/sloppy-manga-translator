param(
    [string]$HostAddr = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Rebuild
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

Write-Host "Starting Miko-chan Manga Studio (Rust) at http://$HostAddr`:$Port" -ForegroundColor Green
if ($env:MIKO_CUDA_DLL_DIR) {
    Write-Host "CUDA DLL dir: $env:MIKO_CUDA_DLL_DIR" -ForegroundColor DarkGray
}
& $ExePath --base-dir $Root --host $HostAddr --port $Port
