# PowerShell script to set up a clean, zero-C-drive-footprint Python environment on Drive J:
# Ensures pip cache, temporary build files, and all packages live strictly on Drive J:

$ErrorActionPreference = "Stop"

Write-Host "=== Setting up local environment on Drive J: ===" -ForegroundColor Cyan

# 1. Ensure temp and pip cache directories on J:
$J_TMP = "J:\tmp"
$J_PIP_CACHE = "J:\pip_cache"

if (-not (Test-Path $J_TMP)) { New-Item -ItemType Directory -Path $J_TMP | Out-Null }
if (-not (Test-Path $J_PIP_CACHE)) { New-Item -ItemType Directory -Path $J_PIP_CACHE | Out-Null }

$env:TEMP = $J_TMP
$env:TMP = $J_TMP
$env:PIP_CACHE_DIR = $J_PIP_CACHE

Write-Host "Redirected TEMP and PIP_CACHE_DIR to $J_TMP and $J_PIP_CACHE (C: drive is protected)." -ForegroundColor Green

# 2. Create virtual environment inside current project directory on J:
$VENV_DIR = ".\venv"
if (-not (Test-Path $VENV_DIR)) {
    Write-Host "Creating Python virtualenv at $VENV_DIR..." -ForegroundColor Yellow
    python -m venv $VENV_DIR
} else {
    Write-Host "Virtual environment already exists at $VENV_DIR." -ForegroundColor Yellow
}

$VENV_PYTHON = ".\venv\Scripts\python.exe"

# 3. Upgrade pip without using C: drive cache
Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $VENV_PYTHON -m pip install --upgrade pip --no-cache-dir

# 4. Install CPU-only PyTorch (Only ~200 MB, compared to 5+ GB CUDA PyTorch)
Write-Host "Installing lightweight CPU-only PyTorch (~200MB)..." -ForegroundColor Yellow
& $VENV_PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# 5. Install the rest of the dependencies
Write-Host "Installing project requirements..." -ForegroundColor Yellow
& $VENV_PYTHON -m pip install -r requirements.txt --no-cache-dir
& $VENV_PYTHON -m pip install streamlit plotly --no-cache-dir

Write-Host "`n=== SETUP COMPLETE ===" -ForegroundColor Cyan
Write-Host "All libraries and models reside completely on Drive J:."
Write-Host "To run commands using this environment:"
Write-Host "  .\venv\Scripts\python -m scripts.run_full_pipeline --quick"
Write-Host "  .\venv\Scripts\python -m scripts.demo --topology all"
Write-Host "  .\venv\Scripts\streamlit run app/main.py"
