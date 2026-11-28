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

$env:NO_PROXY = "127.0.0.1,localhost,::1"
$env:no_proxy = "127.0.0.1,localhost,::1"

function Test-LocalPortOpen([string]$Addr, [int]$TestPort, [int]$TimeoutMs = 400) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $client.BeginConnect($Addr, $TestPort, $null, $null)
        if ($iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false) -and $client.Connected) {
            $client.EndConnect($iar)
            return $true
        }
    } catch {
    } finally {
        $client.Close()
    }
    return $false
}

$TargetUrl = "http://$HostAddr`:$Port"

# Проверяем, не занят ли уже порт предыдущим процессом miko-studio-rs
$existingProcs = @(Get-Process -Name "miko-studio-rs" -ErrorAction SilentlyContinue | Where-Object { $_.Id -ne $PID })
if ($existingProcs.Count -gt 0) {
    $hasLiveConsoleParent = $false
    foreach ($ep in $existingProcs) {
        try {
            $cim = Get-CimInstance Win32_Process -Filter "ProcessId = $($ep.Id)" -ErrorAction Stop
            if ($cim -and $cim.ParentProcessId) {
                $parentProc = Get-Process -Id $cim.ParentProcessId -ErrorAction SilentlyContinue
                if ($parentProc -and ($parentProc.ProcessName -match "^(powershell|pwsh|cmd)$")) {
                    $hasLiveConsoleParent = $true
                }
            }
        } catch {}
    }
    if (-not $hasLiveConsoleParent) {
        Write-Host "Перезапускаем фоновый процесс miko-studio-rs в этом окне консоли..." -ForegroundColor Yellow
        $existingProcs | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 350
    }
}

if (Test-LocalPortOpen -Addr $HostAddr -TestPort $Port) {
    Write-Host ""
    Write-Host "Miko-chan Manga Studio (Rust) уже запущена в другом окне на $TargetUrl — открываем браузер!" -ForegroundColor Green
    if ($OpenBrowser) { Start-Process $TargetUrl }
    $ans = Read-Host "Введите R и Enter, чтобы перезапустить сервер в ЭТОМ окне (или просто Enter, чтобы закрыть дубликат)"
    if ($ans -match "^[rRкК]$") {
        Get-Process -Name "miko-studio-rs" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 400
    } else {
        exit 0
    }
}

Write-Host "Starting Miko-chan Manga Studio (Rust) at $TargetUrl" -ForegroundColor Green

# Ensure local Ollama daemon is running in the background for instant 18+ censorship fallback
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaCmd) {
    if (-not (Test-LocalPortOpen -Addr "127.0.0.1" -TestPort 11434)) {
        Write-Host "Starting background Ollama server for local fallback..." -ForegroundColor DarkCyan
        Start-Process -FilePath $ollamaCmd.Source -ArgumentList "serve" -WindowStyle Hidden
    }
}
if ($env:MIKO_CUDA_DLL_DIR) {
    Write-Host "CUDA DLL dir: $env:MIKO_CUDA_DLL_DIR" -ForegroundColor DarkGray
}
$BrowserArg = if ($OpenBrowser) { "--open-browser" } else { "--no-browser" }
& $ExePath --base-dir $Root --host $HostAddr --port $Port $BrowserArg
