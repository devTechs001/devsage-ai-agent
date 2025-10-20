Write-Host "🚀 Installing DevSage Dependencies..." -ForegroundColor Cyan

# Function to install with fallback
function Install-WithFallback {
    param([string], [string] = )
    
    try {
        Write-Host "Installing ..." -ForegroundColor Yellow -NoNewline
        pip install 
        Write-Host " ✅" -ForegroundColor Green
        return True
    } catch {
        Write-Host " ❌" -ForegroundColor Red
        if () {
            Write-Host "Trying fallback: ..." -ForegroundColor Yellow -NoNewline
            try {
                pip install 
                Write-Host " ✅" -ForegroundColor Green
                return True
            } catch {
                Write-Host " ❌" -ForegroundColor Red
                return False
            }
        }
        return False
    }
}

# Install in order with fallbacks
Write-Host "
📦 Installing Core Framework..." -ForegroundColor Magenta
Install-WithFallback "fastapi==0.104.1"
Install-WithFallback "uvicorn[standard]==0.24.0"
Install-WithFallback "pydantic==2.5.0"
Install-WithFallback "pydantic-settings==2.1.0"

Write-Host "
🤖 Installing AI Packages..." -ForegroundColor Magenta
Install-WithFallback "langchain==0.1.14"
Install-WithFallback "langchain-community==0.0.29"
Install-WithFallback "openai==1.30.2"
Install-WithFallback "ollama==0.1.9" "ollama==0.1.7"

Write-Host "
🧠 Installing ML Packages..." -ForegroundColor Magenta
Install-WithFallback "sentence-transformers==2.7.0"
Install-WithFallback "transformers==4.37.2"
Install-WithFallback "torch==2.1.2"
Install-WithFallback "numpy==1.26.4"

Write-Host "
💾 Installing Vector Store..." -ForegroundColor Magenta
if (-not (Install-WithFallback "faiss-cpu==1.10.0")) {
    Write-Host "Using ChromaDB as FAISS alternative..." -ForegroundColor Yellow
    Install-WithFallback "chromadb==0.4.22"
}

Write-Host "
🔧 Installing Utilities..." -ForegroundColor Magenta
Install-WithFallback "sqlalchemy==2.0.23"
Install-WithFallback "redis==5.0.1"
Install-WithFallback "aiofiles==23.2.1"
Install-WithFallback "black==23.11.0"

Write-Host "
🎉 Installation Complete!" -ForegroundColor Green
Write-Host "Testing installation..." -ForegroundColor Cyan

python -c "
try:
    import fastapi, langchain, ollama, openai
    import sentence_transformers, transformers
    print('✅ All core AI packages installed!')
    
    # Test vector store
    try:
        import faiss
        print('✅ FAISS installed for vector storage')
    except:
        try:
            import chromadb
            print('✅ ChromaDB installed for vector storage')
        except:
            print('⚠️  No vector store installed')
    
    print('🚀 DevSage is ready for development!')
    
except ImportError as e:
    print(f'❌ Missing package: {e}')
"
