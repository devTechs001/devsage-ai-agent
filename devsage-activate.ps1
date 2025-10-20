# DevSage Virtual Environment Activator
param()

function Start-DevSageVenv {
    $venvPath = "$PWD\venv"
    
    Write-Host "🚀 Starting DevSage Virtual Environment..." -ForegroundColor Cyan
    
    # Check if venv exists
    if (-not (Test-Path "$venvPath")) {
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
        if (-not (Test-Path "$venvPath")) {
            Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
            return
        }
    }
    
    # Set environment variables
    $env:VIRTUAL_ENV = $venvPath
    $env:PYTHONHOME = $null
    
    # Clean PATH and add venv
    $oldPath = $env:PATH
    $cleanPath = ($oldPath -split ';' | Where-Object { 
        $_ -notmatch 'Python.*Scripts' -and 
        $_ -notmatch 'Python.*bin' -and
        $_ -notmatch 'Python.*python'
    }) -join ';'
    
    # Add venv paths (try both possible locations)
    $env:PATH = "$venvPath\bin;$venvPath\Scripts;$cleanPath"
    
    # Custom prompt
    function global:prompt {
        Write-Host "(venv) " -NoNewline -ForegroundColor Green
        "PS $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) "
    }
    
    Write-Host "✅ Virtual Environment Activated!" -ForegroundColor Green
    Write-Host "💻 Python: $(python --version)" -ForegroundColor Yellow
    Write-Host "📁 Venv path: $venvPath" -ForegroundColor Yellow
}

Start-DevSageVenv
