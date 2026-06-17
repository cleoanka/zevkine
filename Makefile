# OmniVision — geliştirme kısayolları
.PHONY: help install dev run lint fmt test compile docker clean

help: ## Bu yardımı göster
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Çalışma bağımlılıklarını kur
	pip install -r requirements.txt

dev: ## Geliştirme bağımlılıklarını kur (ruff + pytest + headless cv2)
	pip install -e ".[dev]"

run: ## Sunucuyu başlat (http://localhost:8000)
	python main.py

lint: ## Ruff ile lint
	ruff check .

fmt: ## Ruff ile otomatik düzelt + format
	ruff check --fix .
	ruff format .

test: ## Testleri çalıştır
	pytest

compile: ## Tüm kaynakları byte-compile et (hızlı sağlık kontrolü)
	python -m py_compile main.py core/*.py core/modules/*.py api/*.py

docker: ## Docker imajını derle
	docker build -t omnivision:latest .

clean: ## Geçici dosyaları temizle
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .pytest_cache
