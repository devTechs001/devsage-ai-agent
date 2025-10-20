devsage/
├── 📁 agents/                          # Core AI Agents
│   ├── __init__.py
│   ├── base_agent.py                   # Base agent class
│   ├── coordinator.py                  # Main routing agent
│   ├── reader.py                       # File reading agent
│   ├── writer.py                       # File writing agent
│   ├── editor.py                       # Code editing agent
│   ├── executor.py                     # Code execution agent
│   ├── analyzer.py                     # Code analysis agent
│   ├── debugger.py                     # Debugging assistant
│   ├── security.py                     # Security scanning agent
│   ├── memory.py                       # Conversation memory
│   └── language_specialist.py          # Language-specific operations
├── 📁 api/                             # FastAPI Backend
│   ├── __init__.py
│   ├── main.py                         # FastAPI app entry point
│   ├── models.py                       # Pydantic models
│   ├── dependencies.py                 # API dependencies
│   ├── middleware/                     # Custom middleware
│   │   ├── __init__.py
│   │   ├── auth.py                     # Authentication
│   │   ├── logging.py                  # Request logging
│   │   └── cors.py                     # CORS handling
│   ├── routes/                         # API route handlers
│   │   ├── __init__.py
│   │   ├── agents.py                   # Agent endpoints
│   │   ├── files.py                    # File operations
│   │   ├── projects.py                 # Project management
│   │   ├── chat.py                     # Chat endpoints
│   │   └── health.py                   # Health checks
│   └── websockets/                     # Real-time communication
│       ├── __init__.py
│       ├── manager.py                  # Connection manager
│       └── handlers.py                 # WebSocket handlers
├── 📁 core/                            # Core Utilities
│   ├── __init__.py
│   ├── config.py                       # Configuration management
│   ├── logger.py                       # Structured logging
│   ├── security.py                     # Security utilities
│   ├── exceptions.py                   # Custom exceptions
│   ├── constants.py                    # Project constants
│   ├── utils.py                        # Common utilities
│   └── monitoring.py                   # Metrics and monitoring
├── 📁 languages/                       # Multi-language Support
│   ├── __init__.py
│   ├── base.py                         # Base language class
│   ├── python/
│   │   ├── __init__.py
│   │   ├── executor.py                 # Python execution
│   │   ├── analyzer.py                 # Python analysis
│   │   └── formatter.py                # Python formatting
│   ├── javascript/
│   │   ├── __init__.py
│   │   ├── executor.py                 # Node.js execution
│   │   ├── analyzer.py                 # JS/TS analysis
│   │   └── formatter.py                # Prettier integration
│   ├── java/
│   │   ├── __init__.py
│   │   ├── executor.py                 # Java execution
│   │   └── analyzer.py                 # Java analysis
│   ├── go/
│   │   ├── __init__.py
│   │   ├── executor.py                 # Go execution
│   │   └── analyzer.py                 # Go analysis
│   └── manager.py                      # Language manager
├── 📁 vectorstore/                     # Semantic Search & Memory
│   ├── __init__.py
│   ├── index.py                        # FAISS index management
│   ├── embeddings.py                   # Embedding generation
│   ├── chunking.py                     # Text chunking strategies
│   ├── retriever.py                    # Document retrieval
│   └── memory_manager.py               # Long-term memory
├── 📁 gui/                             # Electron Desktop App
│   ├── package.json
│   ├── main.js                         # Electron main process
│   ├── preload.js                      # Preload script
│   ├── src/
│   │   ├── components/                 # React components
│   │   │   ├── common/
│   │   │   │   ├── Header.jsx
│   │   │   │   ├── Sidebar.jsx
│   │   │   │   └── Footer.jsx
│   │   │   ├── chat/
│   │   │   │   ├── ChatWindow.jsx
│   │   │   │   ├── MessageList.jsx
│   │   │   │   └── InputBox.jsx
│   │   │   ├── files/
│   │   │   │   ├── FileExplorer.jsx
│   │   │   │   ├── CodeEditor.jsx
│   │   │   │   └── FileTree.jsx
│   │   │   └── agents/
│   │   │       ├── AgentStatus.jsx
│   │   │       └── AgentControls.jsx
│   │   ├── hooks/                      # Custom React hooks
│   │   │   ├── useAgents.js
│   │   │   ├── useFiles.js
│   │   │   └── useWebSocket.js
│   │   ├── utils/                      # Frontend utilities
│   │   │   ├── api.js                  # API client
│   │   │   ├── constants.js
│   │   │   └── helpers.js
│   │   ├── styles/                     # CSS/styling
│   │   │   ├── main.css
│   │   │   ├── components.css
│   │   │   └── themes/
│   │   │       ├── light.css
│   │   │       └── dark.css
│   │   └── stores/                     # State management
│   │       ├── agentStore.js
│   │       ├── fileStore.js
│   │       └── chatStore.js
│   ├── public/
│   │   ├── index.html
│   │   ├── favicon.ico
│   │   └── assets/
│   │       ├── icons/
│   │       └── images/
│   └── build/                          # Build artifacts
│       ├── linux/
│       ├── windows/
│       └── mac/
├── 📁 extensions/                      # IDE Extensions
│   ├── vscode/
│   │   ├── package.json
│   │   ├── extension.js                # Main extension file
│   │   ├── src/
│   │   │   ├── providers/
│   │   │   │   ├── CompletionProvider.js
│   │   │   │   ├── HoverProvider.js
│   │   │   │   └── CodeActionProvider.js
│   │   │   ├── commands/
│   │   │   │   ├── askDevSage.js
│   │   │   │   ├── explainCode.js
│   │   │   │   └── refactorCode.js
│   │   │   └── utils/
│   │   │       ├── apiClient.js
│   │   │       └── documentParser.js
│   │   ├── media/                      # Extension assets
│   │   │   └── icon.png
│   │   └── CHANGELOG.md
│   ├── jupyter/
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   ├── src/
│   │   │   ├── index.ts                # Plugin entry point
│   │   │   ├── widgets/
│   │   │   │   ├── DevSageWidget.ts
│   │   │   │   └── ChatWidget.ts
│   │   │   ├── handlers/
│   │   │   │   ├── MessageHandler.ts
│   │   │   │   └── CodeHandler.ts
│   │   │   └── utils/
│   │   │       ├── ApiClient.ts
│   │   │       └── JupyterIntegration.ts
│   │   ├── style/                      # Jupyter styling
│   │   │   └── index.css
│   │   └── schema/                     # Jupyter schemas
│   │       └── plugin.json
│   └── intellij/                       # IntelliJ Platform Plugin
│       ├── build.gradle
│       ├── src/
│       │   ├── main/
│       │   │   ├── kotlin/
│       │   │   │   └── com/
│       │   │   │       └── devsage/
│       │   │   │           ├── DevSageAction.kt
│       │   │   │           ├── DevSageToolWindow.kt
│       │   │   │           └── ApiClient.kt
│       │   │   └── resources/
│       │   │       └── META-INF/
│       │   │           └── plugin.xml
│       │   └── test/
│       └── gradle.properties
├── 📁 data/                            # Application Data
│   ├── models/                         # Local AI models
│   ├── vectorstores/                   # FAISS indices
│   ├── projects/                       # User projects
│   ├── cache/                          # Temporary cache
│   └── logs/                           # Application logs
├── 📁 notebooks/                       # Jupyter Notebooks
│   ├── 01_ai_model_testing.ipynb
│   ├── 02_agent_interaction.ipynb
│   ├── 03_code_generation.ipynb
│   ├── 04_language_support.ipynb
│   ├── 05_integration_testing.ipynb
│   └── templates/                      # Notebook templates
│       ├── agent_testing_template.ipynb
│       └── code_analysis_template.ipynb
├── 📁 tests/                           # Comprehensive Testing
│   ├── __init__.py
│   ├── unit/
│   │   ├── agents/
│   │   │   ├── test_reader.py
│   │   │   ├── test_writer.py
│   │   │   ├── test_editor.py
│   │   │   └── test_coordinator.py
│   │   ├── api/
│   │   │   ├── test_routes.py
│   │   │   └── test_models.py
│   │   ├── core/
│   │   │   ├── test_config.py
│   │   │   └── test_utils.py
│   │   └── languages/
│   │       ├── test_python.py
│   │       └── test_javascript.py
│   ├── integration/
│   │   ├── test_agent_workflow.py
│   │   ├── test_api_integration.py
│   │   ├── test_gui_integration.py
│   │   └── test_extension_integration.py
│   ├── fixtures/                       # Test fixtures
│   │   ├── sample_code/
│   │   │   ├── python_sample.py
│   │   │   ├── javascript_sample.js
│   │   │   └── java_sample.java
│   │   ├── test_projects/
│   │   │   ├── simple_python/
│   │   │   ├── react_app/
│   │   │   └── multi_lang/
│   │   └── mock_data.py
│   ├── conftest.py                     # pytest configuration
│   └── coverage/                       # Coverage reports
├── 📁 docs/                            # Documentation
│   ├── setup/
│   │   ├── installation.md
│   │   ├── configuration.md
│   │   └── deployment.md
│   ├── api/
│   │   ├── endpoints.md
│   │   ├── models.md
│   │   └── examples.md
│   ├── guides/
│   │   ├── getting_started.md
│   │   ├── agent_usage.md
│   │   ├── extension_usage.md
│   │   └── best_practices.md
│   ├── examples/                       # Code examples
│   │   ├── basic_usage.py
│   │   ├── custom_agents.py
│   │   └── integration_examples/
│   ├── images/                         # Documentation images
│   │   ├── architecture.png
│   │   ├── workflow.png
│   │   └── screenshots/
│   └── api-reference/                  # Auto-generated API docs
│       ├── agents.html
│       └── api.html
├── 📁 scripts/                         # Development Scripts
│   ├── setup.sh                        # Environment setup
│   ├── install.sh                      # Installation script
│   ├── build.sh                        # Build script
│   ├── deploy.sh                       # Deployment script
│   ├── test.sh                         # Test runner
│   ├── format.sh                       # Code formatting
│   ├── docker/                         # Docker-related scripts
│   │   ├── build-images.sh
│   │   ├── deploy-stack.sh
│   │   └── cleanup.sh
│   └── release/                        # Release scripts
│       ├── package-extensions.sh
│       ├── build-installers.sh
│       └── upload-assets.sh
├── 📁 docker/                          # Docker Configuration
│   ├── Dockerfile                      # Main application
│   ├── Dockerfile.api                  # API-only image
│   ├── Dockerfile.gui                  # GUI-only image
│   ├── docker-compose.yml              # Development
│   ├── docker-compose.prod.yml         # Production
│   ├── docker-compose.test.yml         # Testing
│   └── nginx/
│       ├── nginx.conf                  # Nginx configuration
│       ├── ssl/                        # SSL certificates
│       │   ├── devsage.crt
│       │   └── devsage.key
│       └── templates/                  # Config templates
│           ├── api.conf
│           └── gui.conf
├── 📁 config/                          # Configuration Files
│   ├── default.yaml                    # Default configuration
│   ├── development.yaml                # Development settings
│   ├── production.yaml                 # Production settings
│   ├── testing.yaml                    # Test settings
│   ├── agents/                         # Agent-specific configs
│   │   ├── coordinator.yaml
│   │   ├── reader.yaml
│   │   └── executor.yaml
│   └── languages/                      # Language configs
│       ├── python.yaml
│       ├── javascript.yaml
│       └── java.yaml
├── 📁 templates/                       # Code Templates
│   ├── project_templates/
│   │   ├── python-fastapi/
│   │   ├── react-typescript/
│   │   ├── node-express/
│   │   └── java-spring/
│   ├── agent_templates/
│   │   ├── base_agent.py.template
│   │   ├── language_agent.py.template
│   │   └── specialized_agent.py.template
│   ├── api_templates/
│   │   ├── fastapi_endpoint.py.template
│   │   └── pydantic_model.py.template
│   └── extension_templates/
│       ├── vscode_extension.js.template
│       └── jupyter_widget.ts.template
├── 📁 examples/                        # Usage Examples
│   ├── basic_usage/
│   │   ├── simple_agent_usage.py
│   │   ├── file_operations.py
│   │   └── code_generation.py
│   ├── advanced_usage/
│   │   ├── custom_agents.py
│   │   ├── workflow_orchestration.py
│   │   └── multi_language_project.py
│   ├── integration_examples/
│   │   ├── vs_code_integration.js
│   │   ├── jupyter_integration.py
│   │   └── web_app_integration/
│   └── demo_projects/
│       ├── todo_app/                   # Full demo project
│       ├── chat_app/
│       └── data_analysis/
├── 📁 .vscode/                         # VS Code Configuration
│   ├── settings.json                   # Workspace settings
│   ├── extensions.json                 # Recommended extensions
│   ├── launch.json                     # Debug configurations
│   ├── tasks.json                      # Task definitions
│   └── snippets/                       # Custom code snippets
│       ├── python.json
│       ├── javascript.json
│       └── devsage.json
├── 📁 .github/                         # GitHub Configuration
│   ├── workflows/
│   │   ├── ci.yml                      # Continuous Integration
│   │   ├── cd.yml                      # Continuous Deployment
│   │   ├── tests.yml                   # Test runner
│   │   └── release.yml                 # Release automation
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── documentation.md
│   └── PULL_REQUEST_TEMPLATE.md
├── 📁 .devcontainer/                   # Dev Container Configuration
│   ├── devcontainer.json               # VS Code Dev Container
│   ├── Dockerfile                      # Dev container image
│   └── setup.sh                        # Container setup script
├── 📄 README.md                        # Project overview
├── 📄 LICENSE                          # License file
├── 📄 pyproject.toml                   # Python project config
├── 📄 requirements.txt                 # Python dependencies
├── 📄 requirements-dev.txt             # Development dependencies
├── 📄 setup.py                         # Python setup script
├── 📄 Makefile                         # Development tasks
├── 📄 .env.example                     # Environment template
├── 📄 .gitignore                       # Git ignore rules
├── 📄 .pre-commit-config.yaml          # Pre-commit hooks
├── 📄 docker-compose.override.yml      # Local overrides
└── 📄 CHANGELOG.md                     # Project changelog