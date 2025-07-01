# Universal Distributed Computing System - Windows Setup Script
# Run this script as Administrator

Write-Host "Universal Distributed Computing System - Windows Setup" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green

# Check if running as Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Error: This script must be run as Administrator" -ForegroundColor Red
    exit 1
}

# Function to check if a command exists
function Test-Command {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Check prerequisites
Write-Host "`nChecking prerequisites..." -ForegroundColor Yellow

$missingTools = @()

if (-not (Test-Command "go")) {
    $missingTools += "Go"
    Write-Host "  [X] Go not found" -ForegroundColor Red
} else {
    Write-Host "  [✓] Go found: $(go version)" -ForegroundColor Green
}

if (-not (Test-Command "docker")) {
    Write-Host "  [!] Docker not found (optional)" -ForegroundColor Yellow
} else {
    Write-Host "  [✓] Docker found: $(docker --version)" -ForegroundColor Green
}

if (-not (Test-Command "nvcc")) {
    Write-Host "  [!] CUDA not found (optional for GPU support)" -ForegroundColor Yellow
} else {
    Write-Host "  [✓] CUDA found: $(nvcc --version | Select-String "release")" -ForegroundColor Green
}

# Check for Visual Studio or Build Tools
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -property installationPath
    if ($vsPath) {
        Write-Host "  [✓] Visual Studio found at: $vsPath" -ForegroundColor Green
    }
} else {
    $missingTools += "Visual Studio Build Tools"
    Write-Host "  [X] Visual Studio Build Tools not found" -ForegroundColor Red
}

if ($missingTools.Count -gt 0) {
    Write-Host "`nMissing required tools: $($missingTools -join ', ')" -ForegroundColor Red
    Write-Host "Please install the missing tools and run this script again." -ForegroundColor Red
    
    if ($missingTools -contains "Go") {
        Write-Host "`nTo install Go:" -ForegroundColor Yellow
        Write-Host "  choco install golang" -ForegroundColor Cyan
        Write-Host "  or download from: https://golang.org/dl/" -ForegroundColor Cyan
    }
    
    if ($missingTools -contains "Visual Studio Build Tools") {
        Write-Host "`nTo install Visual Studio Build Tools:" -ForegroundColor Yellow
        Write-Host "  choco install visualstudio2022buildtools" -ForegroundColor Cyan
        Write-Host "  or download from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Cyan
    }
    
    exit 1
}

# Create bin directory
Write-Host "`nCreating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "bin" | Out-Null
Write-Host "  [✓] Created bin directory" -ForegroundColor Green

# Download Go dependencies
Write-Host "`nDownloading Go dependencies..." -ForegroundColor Yellow
& go mod download
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [✓] Go dependencies downloaded" -ForegroundColor Green
} else {
    Write-Host "  [X] Failed to download Go dependencies" -ForegroundColor Red
    exit 1
}

# Build components
Write-Host "`nBuilding components..." -ForegroundColor Yellow

# Build coordinator
Write-Host "  Building coordinator..." -ForegroundColor Cyan
& go build -o bin\coordinator.exe src\coordinator\main.go
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [✓] Coordinator built successfully" -ForegroundColor Green
} else {
    Write-Host "  [X] Failed to build coordinator" -ForegroundColor Red
}

# Build worker
Write-Host "  Building worker..." -ForegroundColor Cyan
& go build -o bin\worker.exe src\worker\main.go
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [✓] Worker built successfully" -ForegroundColor Green
} else {
    Write-Host "  [X] Failed to build worker" -ForegroundColor Red
}

# Build hardware detector (if compiler available)
if (Test-Command "cl") {
    Write-Host "  Building hardware detector..." -ForegroundColor Cyan
    Push-Location src\detector
    & cl /O2 /EHsc hardware_detector.cpp /Fe:..\..\bin\hardware_detector.exe
    Pop-Location
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [✓] Hardware detector built successfully" -ForegroundColor Green
    } else {
        Write-Host "  [X] Failed to build hardware detector" -ForegroundColor Red
    }
}

# Copy web files
Write-Host "`nCopying web files..." -ForegroundColor Yellow
Copy-Item -Path "src\web" -Destination "bin\" -Recurse -Force
Write-Host "  [✓] Web files copied" -ForegroundColor Green

# Create start scripts
Write-Host "`nCreating start scripts..." -ForegroundColor Yellow

# Coordinator start script
@"
@echo off
echo Starting UDCS Coordinator...
cd /d "%~dp0"
coordinator.exe --port 8080 --web-port 8081
pause
"@ | Out-File -FilePath "bin\start-coordinator.bat" -Encoding ASCII

# Worker start script
@"
@echo off
echo Starting UDCS Worker...
set /p COORDINATOR="Enter coordinator address (e.g., localhost:8080): "
cd /d "%~dp0"
worker.exe --coordinator %COORDINATOR%
pause
"@ | Out-File -FilePath "bin\start-worker.bat" -Encoding ASCII

Write-Host "  [✓] Start scripts created" -ForegroundColor Green

# Create firewall rules
Write-Host "`nConfiguring Windows Firewall..." -ForegroundColor Yellow
try {
    New-NetFirewallRule -DisplayName "UDCS Coordinator" -Direction Inbound -LocalPort 8080,8081 -Protocol TCP -Action Allow -ErrorAction SilentlyContinue | Out-Null
    Write-Host "  [✓] Firewall rules created" -ForegroundColor Green
} catch {
    Write-Host "  [!] Could not create firewall rules automatically" -ForegroundColor Yellow
    Write-Host "     Please manually allow ports 8080 and 8081 in Windows Firewall" -ForegroundColor Yellow
}

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "`nTo start the system:" -ForegroundColor Yellow
Write-Host "  1. Start coordinator: .\bin\start-coordinator.bat" -ForegroundColor Cyan
Write-Host "  2. Start workers:     .\bin\start-worker.bat" -ForegroundColor Cyan
Write-Host "  3. Open web UI:       http://localhost:8081" -ForegroundColor Cyan
Write-Host "`nFor Python client:" -ForegroundColor Yellow
Write-Host "  pip install requests numpy websockets" -ForegroundColor Cyan
Write-Host "  python client.py" -ForegroundColor Cyan 