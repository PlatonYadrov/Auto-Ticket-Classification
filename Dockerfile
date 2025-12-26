# Auto Ticket Classification - Production Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы зависимостей
COPY pyproject.toml poetry.lock* ./
COPY README.md ./

# Устанавливаем Poetry и зависимости
RUN pip install --no-cache-dir poetry==1.7.1 \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Копируем код приложения
COPY ticket_triage_ml/ ./ticket_triage_ml/
COPY configs/ ./configs/

# Копируем артефакты модели (должны быть готовы до сборки)
COPY artifacts/ ./artifacts/

# Порт для API
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Запуск API сервера
CMD ["python", "-m", "uvicorn", "ticket_triage_ml.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
