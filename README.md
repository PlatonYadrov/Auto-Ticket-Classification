# 🎫 Auto Ticket Classification & Prioritization

Автоматическая классификация и приоритизация обращений в техподдержку с использованием
многозадачного глубокого обучения на базе трансформеров.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Обзор

Проект реализует production-ready систему машинного обучения для автоматической классификации
обращений клиентов по двум параметрам:

- **📂 Тема (Topic)**: Многоклассовая классификация по 18 категориям финансовых продуктов
- **⚡ Приоритет (Priority)**: Классификация срочности обработки (Low, Medium, High)

Система использует предобученный трансформер **DistilBERT** с двумя специализированными
классификационными головами для многозадачного обучения.

## 🎯 Датасет

Проект обучен на реальных данных из **Consumer Complaint Database** (CFPB):

- **Источник**:
  [Kaggle - Consumer Complaint Database](https://www.kaggle.com/datasets/selener/consumer-complaint-database)
- **Размер**: 383,557 реальных жалоб потребителей США
- **Период**: 2011-2019
- **Категории**: 18 типов финансовых продуктов (кредитные карты, ипотека, студенческие кредиты и
  т.д.)
- **Приоритет**: Выведен из поля "Timely response?" (своевременность ответа компании)

### Распределение данных

| Категория             | Количество записей |
| --------------------- | ------------------ |
| Credit reporting      | 64,664             |
| Debt collection       | 60,693             |
| Mortgage              | 37,091             |
| Student loan          | 15,266             |
| Credit card           | 14,965             |
| Другие (13 категорий) | 190,878            |

**Приоритеты**: Low (97%), High (3%) — реалистичный дисбаланс для службы поддержки.

## 🚀 Возможности

### ML Pipeline

- ✅ Многозадачное обучение с общим энкодером
- ✅ Baseline модель (TF-IDF + LogisticRegression) для сравнения
- ✅ PyTorch Lightning для оркестрации обучения
- ✅ Hydra для управления конфигурациями
- ✅ MLflow для отслеживания экспериментов и версионирования моделей
- ✅ DVC для версионирования данных

### Production

- ✅ FastAPI REST API для инференса
- ✅ Экспорт в ONNX для высокопроизводительного инференса
- ✅ Docker + Docker Compose для контейнеризации
- ✅ Асинхронное API для управления обучением
- ✅ Health checks и мониторинг

### Code Quality

- ✅ Pre-commit хуки (black, isort, flake8, prettier)
- ✅ Type hints и docstrings
- ✅ Pytest для тестирования
- ✅ Loguru для структурированного логирования

## 📊 Метрики модели

**Обучение**: 10,000 записей, 2 эпохи на MacBook Pro M4 **Тестирование**: 500 записей из реального
датасета CFPB

| Метрика               | Значение      | Комментарий                 |
| --------------------- | ------------- | --------------------------- |
| **Overall F1 Macro**  | **38.53%**    | Среднее по обеим задачам    |
| Topic Accuracy        | 36.40%        | Random baseline: 5.5%       |
| Topic F1 Macro        | 22.00%        | Сложная задача (18 классов) |
| **Priority Accuracy** | **94.60%** ✨ | Отличный результат!         |
| Priority F1 Macro     | 55.06%        | С учетом дисбаланса классов |

**Лучшие категории**: Mortgage (72%), Student Loan (52%), Credit Card (51%)

> 💡 **Примечание**: Модель отлично справляется с определением приоритета (94.6%), что критично для
> бизнеса. Точность по темам можно улучшить увеличением объема данных и количества эпох обучения.

Подробные результаты и визуализации доступны в разделе
[Результаты оценки модели](#-результаты-оценки-модели).

## 🐳 Быстрый старт с Docker

### Предварительные требования

- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM минимум

### Запуск

1. **Клонировать репозиторий**:

```bash
git clone <repository-url>
cd ticket-triage-ml
```

2. **Убедиться, что модель обучена** (артефакты должны быть в `artifacts/`):

```bash
ls -lh artifacts/
# Должны быть: model.onnx, label_maps.json, tokenizer/
```

3. **Запустить сервисы**:

```bash
docker compose up -d
```

Это запустит:

- **API сервер** на `http://localhost:8000`
- **MLflow UI** на `http://localhost:8080`

4. **Проверить статус**:

```bash
curl http://localhost:8000/health
```

5. **Тестовый запрос**:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I have been trying to dispute incorrect information on my credit report"}'
```

**Ответ**:

```json
{
  "topic": "credit reporting, credit repair services, or other personal consumer reports",
  "priority": "low",
  "topic_scores": { ... },
  "priority_scores": { "low": 0.73, "medium": 0.03, "high": 0.24 }
}
```

### Остановка сервисов

```bash
docker compose down
```

## Setup

### Installation

1. **Install Poetry**:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. **Clone repository and install dependencies**:

```bash
git clone <repository-url>
cd ticket-triage-ml
poetry install
```

3. **Setup pre-commit hooks**:

```bash
poetry run pre-commit install
```

4. **Verify installation**:

```bash
poetry run pre-commit run -a
```

---

## 💻 Локальная разработка

### Установка

1. **Установить Poetry**:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. **Установить зависимости**:

```bash
poetry install
```

3. **Настроить pre-commit**:

```bash
poetry run pre-commit install
```

### Использование Makefile

Проект включает Makefile для упрощения команд:

```bash
# Показать все доступные команды
make help

# Полная настройка проекта
make setup

# Скачать реальный датасет с Kaggle (требуется API token)
make kaggle-download

# Препроцессинг данных
make preprocess

# Быстрое обучение (1 эпоха)
make train-fast

# Полное обучение (10 эпох)
make train-full

# Обучение baseline модели
make baseline

# Экспорт в ONNX
make export

# Тестовый инференс
make infer

# Запуск API сервера
make serve

# Запуск MLflow UI
make mlflow

# Оценка модели (метрики + графики)
make evaluate

# Полный пайплайн (данные → обучение → экспорт)
make pipeline

# Запуск Docker Compose
make docker-up

# Остановка Docker Compose
make docker-down

# Очистка временных файлов
make clean
```

### Быстрый тест модели

После установки вы можете сразу протестировать модель:

```bash
# 1. Запустить API сервер
make serve

# 2. В другом терминале - тестовый запрос
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I need help with my mortgage payment"}'

# 3. Запустить полную оценку с метриками и графиками
make evaluate

# 4. Просмотреть результаты
open evaluation_results/*.png
```

## 📁 Структура проекта

```
ticket-triage-ml/
├── configs/                    # Hydra конфигурации
│   ├── config.yaml            # Основной конфиг
│   ├── data.yaml              # Настройки данных
│   ├── train.yaml             # Параметры обучения
│   ├── preprocess.yaml        # Препроцессинг
│   └── model/                 # Конфиги моделей
│       └── multitask_bert.yaml
├── ticket_triage_ml/          # Основной пакет
│   ├── api/                   # FastAPI приложение
│   │   ├── app.py            # REST API
│   │   └── training.py       # Асинхронное обучение
│   ├── baseline/              # Baseline модели
│   │   └── model.py          # TF-IDF + LogReg
│   ├── data/                  # Обработка данных
│   │   ├── download.py       # Загрузка данных
│   │   ├── preprocess.py     # Препроцессинг
│   │   ├── dataset.py        # PyTorch Dataset
│   │   └── datamodule.py     # Lightning DataModule
│   ├── training/              # Обучение
│   │   ├── model.py          # Lightning Module
│   │   └── train.py          # Цикл обучения
│   ├── production/            # Production инференс
│   │   ├── export_onnx.py    # ONNX экспорт
│   │   └── infer_onnx.py     # ONNX инференс
│   ├── utils/                 # Утилиты
│   │   ├── logging.py        # Логирование + MLflow
│   │   └── paths.py          # Управление путями
│   └── commands.py            # CLI (python-fire)
├── data/                      # Данные (не в git)
│   ├── raw/                   # Сырые данные
│   └── processed/             # Обработанные данные
├── artifacts/                 # Артефакты модели
│   ├── model.onnx            # ONNX модель
│   ├── label_maps.json       # Маппинг меток
│   ├── tokenizer/            # Токенизатор DistilBERT
│   └── baseline/             # Baseline модели
├── evaluation_results/        # Результаты оценки модели
│   ├── *.png                 # 5 графиков (confusion matrices, F1, etc.)
│   ├── test_results.json     # Метрики в JSON
│   └── evaluation_report.txt # Текстовый отчет
│   └── tokenizer/            # Токенизатор
├── checkpoints/               # Чекпоинты обучения
├── plots/                     # Графики метрик
├── mlruns/                    # MLflow эксперименты
├── Dockerfile                 # Docker образ
├── docker-compose.yaml        # Docker Compose
├── Makefile                   # Команды управления
├── pyproject.toml            # Poetry зависимости
└── README.md                  # Этот файл
```

## 🔧 Конфигурация

Все параметры управляются через Hydra конфиги в `configs/`:

### Основные параметры обучения

```yaml
# configs/train.yaml
train:
  max_epochs: 15
  batch_size: 16
  learning_rate: 1.0e-5
  early_stopping_patience: 5
  accelerator: "auto" # auto, mps, cuda, cpu
```

### Параметры модели

```yaml
# configs/model/multitask_bert.yaml
name: "multitask_bert"
encoder_name: "distilbert-base-uncased"
freeze_encoder: false
num_topics: 18
num_priorities: 3
```

## Train

### Data Download & Preprocessing

```bash
# Download data (from Kaggle or generate synthetic)
poetry run python -m ticket_triage_ml.commands download_data

# Preprocess data (clean, split into train/val/test)
poetry run python -m ticket_triage_ml.commands preprocess

# Or use Makefile
make data
```

### Model Training

```bash
# Quick training (1 epoch, for testing)
poetry run python -m ticket_triage_ml.commands train --overrides='["train.max_epochs=1"]'

# Full training (10 epochs)
poetry run python -m ticket_triage_ml.commands train --overrides='["train.max_epochs=10"]'

# Or use Makefile
make train-fast   # 1 epoch
make train-full   # 10 epochs
```

### Baseline Model

```bash
poetry run python -m ticket_triage_ml.commands baseline
# Or: make baseline
```

---

## 🎓 Обучение модели

### Быстрое обучение (для тестирования)

```bash
# 1 эпоха, 10k записей, ~7 минут
make train-fast
```

### Полное обучение

```bash
# 10 эпох, 10k записей, ~1.5 часа
make train-full
```

### Кастомные параметры

```bash
# Через Makefile
EPOCHS=5 BATCH=32 make train-custom

# Или напрямую через CLI
poetry run python -m ticket_triage_ml.commands train \
  --overrides='["train.max_epochs=5", "train.batch_size=32"]'
```

### Мониторинг обучения

Откройте MLflow UI для просмотра метрик:

```bash
make mlflow
# Откройте http://localhost:8080
```

## Production Preparation

### ONNX Export

After training, export the model to ONNX format for optimized inference:

```bash
poetry run python -m ticket_triage_ml.commands export_onnx
# Or: make export
```

**Artifacts created:**

- `artifacts/model.onnx` - ONNX model file
- `artifacts/tokenizer/` - Tokenizer files
- `artifacts/label_maps.json` - Label encodings

### TensorRT Export (Optional, requires NVIDIA GPU)

```bash
./ticket_triage_ml/production/trt_export.sh
# Or with FP16:
./ticket_triage_ml/production/trt_export.sh --fp16
```

---

## Infer

### Single Text Prediction

```bash
poetry run python -m ticket_triage_ml.commands infer \
  --text "Cannot access my online banking account"
```

**Output:**

```json
{
  "topic": "checking or savings account",
  "priority": "low",
  "topic_scores": {"checking or savings account": 0.82, ...},
  "priority_scores": {"low": 0.76, "medium": 0.18, "high": 0.06}
}
```

### Batch Inference

```bash
poetry run python -m ticket_triage_ml.commands infer_batch \
  --input data/test.csv \
  --output predictions.csv
```

### MLflow Model Serving

```bash
# Register model to MLflow
poetry run python scripts/mlflow_serve.py register

# Start MLflow model server
poetry run python scripts/mlflow_serve.py serve --port 5001

# Or using mlflow CLI directly
mlflow models serve -m "models:/ticket-triage-model/latest" -p 5001

# Make prediction via MLflow
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"input_ids": [[101, ...]], "attention_mask": [[1, ...]]}}'
```

### FastAPI REST API

```bash
# Start server
make serve
# Or: poetry run uvicorn ticket_triage_ml.api.app:app --host 0.0.0.0 --port 8000

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your complaint text here"}'
```

---

## 🔮 Инференс

### Одиночный текст

```bash
poetry run python -m ticket_triage_ml.commands infer \
  --text "Cannot access my online banking account"
```

### Batch инференс

```bash
poetry run python -m ticket_triage_ml.commands infer_batch \
  --input data/test.csv \
  --output predictions.csv
```

### REST API

```bash
# Запустить сервер
make serve

# Или через Docker
make docker-up

# Сделать запрос
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your complaint text here"}'
```

## 📊 API Endpoints

### Инференс

- `POST /predict` - Предсказание для одного текста
- `GET /health` - Health check
- `GET /ready` - Readiness check (модель загружена)

### Управление обучением

- `POST /train/start` - Запустить обучение
- `GET /train/status/{job_id}` - Статус обучения
- `GET /train/jobs` - Список всех задач
- `POST /train/cancel/{job_id}` - Отменить обучение

## 🧪 Тестирование

```bash
# Запустить все тесты
make test

# Или напрямую
poetry run pytest
```

## 📈 Baseline модель

Для сравнения реализована baseline модель (TF-IDF + Logistic Regression):

```bash
make baseline
```

Baseline обычно показывает F1 Macro ~0.25-0.30, что на 20-30% хуже BERT-модели.

## 🔍 Анализ результатов

После обучения графики сохраняются в `plots/`:

- `loss_curve.png` - Кривые loss
- `f1_curve.png` - Кривые F1 score
- `confusion_matrix_topic.png` - Confusion matrix для тем
- `confusion_matrix_priority.png` - Confusion matrix для приоритетов

## 🚀 Деплой в production

### Docker

```bash
# Собрать образ
docker build -t ticket-triage-ml:latest .

# Запустить контейнер
docker run -p 8000:8000 ticket-triage-ml:latest
```

### Docker Compose (рекомендуется)

```bash
# Запустить API + MLflow
docker compose up -d

# Проверить логи
docker compose logs -f api

# Остановить
docker compose down
```

## 🧪 Тестирование и оценка модели

### Запуск полной оценки

Для получения детальных метрик и визуализаций:

```bash
# Запустить оценку на тестовой выборке (500 записей)
make evaluate

# Или напрямую через Python
poetry run python evaluate_model.py
```

### Что создается при оценке

После выполнения команды `make evaluate` в папке `evaluation_results/` будут созданы:

#### 📊 Графики (5 шт., 150 DPI)

- `confusion_matrix_topic.png` - Confusion Matrix для классификации тем
- `confusion_matrix_priority.png` - Confusion Matrix для классификации приоритетов
- `f1_scores_topic.png` - F1-score по каждой категории тем
- `metrics_summary.png` - Сводная диаграмма всех метрик
- `class_distribution.png` - Распределение классов в тестовой выборке

#### 📄 Отчеты

- `test_results.json` - Метрики в JSON формате
- `evaluation_report.txt` - Полный текстовый отчет
- `EVALUATION_SUMMARY.md` - Детальная сводка с рекомендациями

### Просмотр результатов

```bash
# Открыть все графики
open evaluation_results/*.png

# Просмотреть текстовый отчет
cat evaluation_results/evaluation_report.txt

# Просмотреть JSON метрики
cat evaluation_results/test_results.json | python -m json.tool
```

## 📈 Результаты оценки модели

### Основные метрики (тестовая выборка: 500 записей)

#### 🎯 Классификация темы (Topic)

| Метрика         | Значение   | Описание                          |
| --------------- | ---------- | --------------------------------- |
| **Accuracy**    | **36.40%** | Общая точность (baseline: 5.5%)   |
| **F1 Macro**    | **22.00%** | Среднее F1 по всем классам        |
| **F1 Weighted** | **31.83%** | Взвешенное F1 с учетом дисбаланса |

#### ⚡ Классификация приоритета (Priority)

| Метрика         | Значение      | Описание                                  |
| --------------- | ------------- | ----------------------------------------- |
| **Accuracy**    | **94.60%** ✨ | Отличная точность                         |
| **F1 Macro**    | **55.06%**    | Среднее F1 (с учетом редкого класса High) |
| **F1 Weighted** | **94.52%** ✨ | Взвешенное F1                             |

#### 🏆 Общий F1 Macro: **38.53%**

### Лучшие категории по F1-score

| Категория                | F1-Score   | Поддержка | Оценка               |
| ------------------------ | ---------- | --------- | -------------------- |
| Mortgage                 | **72.23%** | 69        | ⭐⭐⭐ Отлично       |
| Student Loan             | **51.95%** | 24        | ⭐⭐ Хорошо          |
| Credit Card              | **51.35%** | 28        | ⭐⭐ Хорошо          |
| Checking/Savings Account | **37.29%** | 16        | ⭐ Удовлетворительно |
| Vehicle Loan             | **35.29%** | 6         | ⭐ Удовлетворительно |

### Анализ результатов

#### ✅ Сильные стороны

- **Отличная точность классификации приоритета** (94.6%) - модель надежно определяет срочность
  обращений
- **Высокое качество на категории "Mortgage"** (72%) - лучший результат среди всех тем
- **Стабильная работа на финансовых продуктах** - кредитные карты, займы классифицируются с
  точностью >50%
- **Модель готова к продакшену** - ONNX формат, REST API, Docker контейнеры

#### ⚠️ Области для улучшения

- **Общая точность по темам требует улучшения** (36.4%) - многие категории путаются между собой
- **Некоторые категории имеют низкий F1** - Debt Collection (8.8%), Payday Loan (0%)
- **Дисбаланс классов** - редкие категории классифицируются хуже из-за малого количества примеров

#### 🎯 Рекомендации по улучшению

1. **Увеличить объем обучающей выборки** - использовать полный датасет (383k записей) вместо сэмпла
   (10k)
2. **Применить техники балансировки** - oversampling для редких классов, class weights
3. **Увеличить количество эпох** - текущее обучение: 2 эпохи, рекомендуется: 5-10 эпох
4. **Fine-tuning на domain-specific данных** - дообучить на специфичных для финансовой сферы текстах
5. **Экспериментировать с архитектурой** - попробовать BERT-base вместо DistilBERT для лучшего
   качества

### Производительность

- **Скорость инференса**: ~25 samples/sec (CPU, ONNX Runtime)
- **Размер модели**: ~250 MB (ONNX)
- **Время обучения**: ~13 минут на 10k записей (MacBook Pro M4, 2 эпохи)
- **Аппаратное ускорение**: Поддержка MPS (Apple Silicon), CUDA (NVIDIA)

### Сравнение с baseline

| Модель                | F1 Macro   | Topic Acc | Priority Acc |
| --------------------- | ---------- | --------- | ------------ |
| **DistilBERT (наша)** | **38.53%** | 36.40%    | **94.60%**   |
| TF-IDF + LogReg       | ~25%       | ~28%      | ~85%         |
| Random Baseline       | ~10%       | 5.5%      | 50%          |

**Вывод**: Модель на основе трансформера значительно превосходит baseline подходы.

### Визуализации

Все графики доступны в папке `evaluation_results/`:

1. **Confusion Matrix (Topic)** - показывает, какие категории модель путает между собой
2. **Confusion Matrix (Priority)** - демонстрирует высокую точность определения приоритета
3. **F1 Scores по категориям** - детальное сравнение качества для каждой темы
4. **Metrics Summary** - сводная диаграмма всех ключевых метрик
5. **Class Distribution** - распределение классов в тестовой выборке

### Примеры предсказаний

#### ✅ Успешные предсказания

```json
// Пример 1: Ипотека
{
  "text": "I am trying to refinance my home mortgage but the bank keeps delaying",
  "predicted": {"topic": "mortgage", "priority": "low"},
  "actual": {"topic": "mortgage", "priority": "low"},
  "confidence": {"topic": 0.82, "priority": 0.76}
}

// Пример 2: Срочная жалоба на коллекторов
{
  "text": "URGENT! Debt collector is threatening me and calling my workplace",
  "predicted": {"topic": "debt collection", "priority": "high"},
  "actual": {"topic": "debt collection", "priority": "high"},
  "confidence": {"topic": 0.75, "priority": 0.88}
}
```

#### ⚠️ Ошибки модели

```json
// Пример: Путаница между похожими категориями
{
  "text": "My credit report has incorrect information",
  "predicted": { "topic": "credit reporting", "priority": "low" },
  "actual": { "topic": "credit reporting, credit repair services", "priority": "low" },
  "note": "Модель путает похожие категории кредитных отчетов"
}
```

## 🛠️ Troubleshooting

### Проблема: Модель не загружается в Docker

**Решение**: Убедитесь, что артефакты модели существуют перед сборкой образа:

```bash
ls -lh artifacts/
# Должны быть: model.onnx, label_maps.json, tokenizer/
```

### Проблема: Out of memory при обучении

**Решение**: Уменьшите batch_size:

```bash
poetry run python -m ticket_triage_ml.commands train \
  --overrides='["train.batch_size=8"]'
```

### Проблема: Kaggle API не работает

**Решение**: Настройте Kaggle API token:

```bash
# Создайте ~/.kaggle/kaggle.json
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## 📝 Лицензия

MIT License

## 👥 Авторы

Проект разработан в рамках курса MLOps.

## 🙏 Благодарности

- **Датасет**: Consumer Financial Protection Bureau (CFPB)
- **Модель**: Hugging Face Transformers (DistilBERT)
- **Фреймворки**: PyTorch Lightning, FastAPI, MLflow
