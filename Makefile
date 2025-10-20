.PHONY: install dev research backend frontend test format clean

install:
	python -m venv venv
	./venv/bin/pip install -r requirements.txt
	cd gui && npm install
	cd extensions/vscode && npm install

dev:
	docker-compose up -d ollama
	make backend & make frontend

backend:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd gui && npm start

research:
	jupyter lab notebooks/ --port 8888

test:
	pytest tests/ -v

format:
	black .
	ruff --fix .

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

docker-dev:
	docker-compose up --build

docker-prod:
	docker-compose -f docker-compose.prod.yml up --build -d