.PHONY: help install setup data train train-fast train-full export infer serve test lint clean docker-up docker-down mlflow evaluate

# Цвета для вывода
YELLOW := \033[1;33m
GREEN := \033[1;32m
CYAN := \033[1;36m
NC := \033[0m

help:  ## Показать справку
	@echo "$(CYAN)╔══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║     Auto Ticket Classification - Команды управления            ║$(NC)"
	@echo "$(CYAN)╚══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ==================== УСТАНОВКА ====================

install:  ## Установить зависимости
	poetry install

setup: install  ## Полная настройка проекта
	poetry run pre-commit install
	@echo "$(GREEN)✓ Проект настроен!$(NC)"

# ==================== ДАННЫЕ ====================

data:  ## Скачать и подготовить данные (синтетические)
	poetry run python -m ticket_triage_ml.commands download_data
	poetry run python -m ticket_triage_ml.commands preprocess
	@echo "$(GREEN)✓ Данные готовы!$(NC)"

kaggle-download:  ## Скачать реальный датасет с Kaggle
	@echo "$(YELLOW)📥 Загрузка датасета с Kaggle...$(NC)"
	@mkdir -p data/raw
	poetry run kaggle datasets download -d suraj520/customer-support-ticket-dataset -p data/raw --unzip
	@if [ -f "data/raw/customer_support_tickets.csv" ]; then \
		mv data/raw/customer_support_tickets.csv data/raw/tickets.csv; \
	fi
	@echo "$(GREEN)✓ Датасет загружен!$(NC)"
	@echo "Теперь выполните: make preprocess"

kaggle-alt:  ## Альтернативный датасет (IT tickets)
	@echo "$(YELLOW)📥 Загрузка альтернативного датасета...$(NC)"
	@mkdir -p data/raw
	poetry run kaggle datasets download -d adisongoh/it-helpdesk-ticket-dataset -p data/raw --unzip
	@echo "$(GREEN)✓ Датасет загружен!$(NC)"

preprocess:  ## Препроцессинг данных
	poetry run python -m ticket_triage_ml.commands preprocess
	@echo "$(GREEN)✓ Данные обработаны!$(NC)"

# ==================== ОБУЧЕНИЕ ====================

train:  ## Обучить модель (3 эпохи, стандартные параметры)
	@echo "$(YELLOW)🚀 Запуск обучения...$(NC)"
	poetry run python -m ticket_triage_ml.commands train
	@echo "$(GREEN)✓ Обучение завершено!$(NC)"

train-fast:  ## Быстрое обучение (1 эпоха, для теста)
	@echo "$(YELLOW)⚡ Быстрое обучение (1 эпоха)...$(NC)"
	poetry run python -m ticket_triage_ml.commands train --overrides='["train.max_epochs=1", "train.batch_size=16"]'
	@echo "$(GREEN)✓ Обучение завершено!$(NC)"

train-full:  ## Полное обучение (10 эпох)
	@echo "$(YELLOW)🔥 Полное обучение (10 эпох)...$(NC)"
	poetry run python -m ticket_triage_ml.commands train --overrides='["train.max_epochs=10", "train.batch_size=16"]'
	@echo "$(GREEN)✓ Обучение завершено!$(NC)"

train-custom:  ## Обучение с кастомными параметрами (EPOCHS=5 BATCH=8 make train-custom)
	@echo "$(YELLOW)🔧 Кастомное обучение (epochs=$(EPOCHS), batch=$(BATCH))...$(NC)"
	poetry run python -m ticket_triage_ml.commands train --overrides='["train.max_epochs=$(EPOCHS)", "train.batch_size=$(BATCH)"]'
	@echo "$(GREEN)✓ Обучение завершено!$(NC)"

baseline:  ## Обучить baseline модель (TF-IDF + LogReg)
	@echo "$(YELLOW)📊 Обучение baseline...$(NC)"
	poetry run python -m ticket_triage_ml.commands baseline
	@echo "$(GREEN)✓ Baseline готов!$(NC)"

# ==================== ЭКСПОРТ И ИНФЕРЕНС ====================

export:  ## Экспортировать модель в ONNX
	@echo "$(YELLOW)📦 Экспорт в ONNX...$(NC)"
	poetry run python -m ticket_triage_ml.commands export_onnx
	@echo "$(GREEN)✓ Модель экспортирована!$(NC)"

infer:  ## Тестовый инференс
	@echo "$(CYAN)Пример инференса:$(NC)"
	poetry run python -m ticket_triage_ml.commands infer --text "Cannot login to VPN"

serve:  ## Запустить API сервер локально
	@echo "$(YELLOW)🌐 Запуск API на http://localhost:8000$(NC)"
	poetry run python -m ticket_triage_ml.commands serve

# ==================== ПОЛНЫЙ ПАЙПЛАЙН ====================

pipeline: data train export  ## Полный пайплайн: данные → обучение → экспорт
	@echo "$(GREEN)✓ Пайплайн завершён!$(NC)"

pipeline-fast: data train-fast export  ## Быстрый пайплайн (1 эпоха)
	@echo "$(GREEN)✓ Быстрый пайплайн завершён!$(NC)"

# ==================== ТЕСТЫ И ОЦЕНКА ====================

test:  ## Запустить тесты
	poetry run pytest tests/ -v

evaluate:  ## Оценить модель на тестовой выборке (метрики + графики)
	@echo "$(YELLOW)📊 Запуск оценки модели...$(NC)"
	poetry run python evaluate_model.py
	@echo "$(GREEN)✓ Результаты сохранены в evaluation_results/$(NC)"
	@echo "  • 5 графиков (PNG)"
	@echo "  • test_results.json"
	@echo "  • evaluation_report.txt"

lint:  ## Проверить код (pre-commit)
	poetry run pre-commit run -a

# ==================== DOCKER ====================

docker-up:  ## Запустить Docker Compose (API + MLflow)
	docker compose up -d --build
	@echo "$(GREEN)✓ Контейнеры запущены!$(NC)"
	@echo "  API:    http://localhost:8000/docs"
	@echo "  MLflow: http://localhost:8080"

docker-down:  ## Остановить Docker Compose
	docker compose down
	@echo "$(GREEN)✓ Контейнеры остановлены$(NC)"

docker-logs:  ## Показать логи контейнеров
	docker compose logs -f

# ==================== MLFLOW ====================

mlflow:  ## Запустить MLflow сервер локально
	@echo "$(YELLOW)📈 MLflow UI: http://localhost:8080$(NC)"
	poetry run mlflow server --host 0.0.0.0 --port 8080

# ==================== ОЧИСТКА ====================

clean:  ## Очистить временные файлы
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf ticket_triage_ml/__pycache__
	rm -rf lightning_logs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Очищено$(NC)"

clean-all: clean  ## Полная очистка (включая модели)
	rm -rf checkpoints/*.ckpt
	rm -rf artifacts/model.onnx
	rm -rf plots/*.png
	@echo "$(GREEN)✓ Полная очистка выполнена$(NC)"
