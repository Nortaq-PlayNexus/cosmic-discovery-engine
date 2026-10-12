# Cosmic Discovery Engine — Windows PowerShell setup script
# Run from the project root directory.

$ErrorActionPreference = "Stop"

Write-Host "=== Cosmic Discovery Engine Setup ===" -ForegroundColor Cyan

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "Python: $pythonVersion"

# Create virtual environment
if (-Not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install package in editable mode with dev dependencies
Write-Host "Installing package and dependencies..." -ForegroundColor Yellow
pip install -e ".[dev]" --quiet

# Create data directory
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "data\cache\mast" | Out-Null

# Copy .env.example if .env doesn't exist
if (-Not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Quick start:" -ForegroundColor Cyan
Write-Host "  cosmic-discovery demo-ingest --all"
Write-Host "  cosmic-discovery list-candidates"
Write-Host "  cosmic-discovery dashboard"
Write-Host ""
Write-Host "To activate the environment manually:" -ForegroundColor Cyan
Write-Host "  .venv\Scripts\Activate.ps1"
