# Auto Ticket Classification

Автоматическая классификация и приоритизация тикетов технической поддержки с использованием
многозадачного глубокого обучения.

## Обзор

Проект реализует систему машинного обучения для автоматической классификации тикетов поддержки по
двум параметрам:

- **Тема (Topic)**: Многоклассовая классификация (например, Техническая поддержка, Биллинг, Доступ к
  аккаунту)
- **Приоритет (Priority)**: Многоклассовая классификация (Low, Medium, High)

Система использует общий трансформер-энкодер (DistilBERT) с двумя классификационными головами для
многозадачного обучения.

## Возможности

- Многозадачное обучение с общим энкодером
- Baseline модель (TF-IDF + LogisticRegression) для сравнения
- PyTorch Lightning для оркестрации обучения
- Hydra для управления конфигурациями
- MLflow для отслеживания экспериментов
- FastAPI REST API для продакшн-сервиса
- Экспорт в ONNX для продакшн-инференса
- Docker для контейнеризации
- DVC для версионирования данных
- Pre-commit хуки для качества кода

## Форматы входных/выходных данных

### Офлайн данные (обучение)

CSV или Parquet файлы со столбцами:

| Столбец    | Тип | Описание                |
| ---------- | --- | ----------------------- |
| `text`     | str | Текст тикета поддержки  |
| `topic`    | str | Метка категории темы    |
| `priority` | str | Метка уровня приоритета |

### Онлайн инференс

**Входной JSON:**

```json
{
  "text": "Не могу подключиться к VPN с ноутбука"
}
```

**Выходной JSON:**

```json
{
  "topic": "technical support",
  "priority": "high",
  "topic_scores": {
    "technical support": 0.85,
    "billing": 0.05,
    "account access": 0.1
  },
  "priority_scores": {
    "low": 0.1,
    "medium": 0.2,
    "high": 0.7
  }
}
```

## Установка

### Требования

- Python 3.10+
- Менеджер пакетов Poetry

### Инструкция по установке

1. Клонировать репозиторий:

```bash
git clone https://github.com/your-org/auto-ticket-classification.git
cd auto-ticket-classification
```

2. Установить зависимости:

```bash
poetry install
```

3. Настроить pre-commit хуки:

```bash
poetry run pre-commit install
```

4. Проверить, что pre-commit проходит:

```bash
poetry run pre-commit run -a
```

## Использование

### Загрузка данных

Скачать или сгенерировать обучающие данные:

```bash
poetry run python -m ticket_triage_ml.commands download_data
```

Источники данных (в порядке приоритета):

1. DVC pull (если настроен)
2. Загрузка по резервному URL
3. Генерация синтетических данных (крайний случай, только для тестирования)

### Предобработка

Очистка и разбиение данных:

```bash
poetry run python -m ticket_triage_ml.commands preprocess
```

Результат:

- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`
- `artifacts/label_maps.json`

### Обучение

Обучить многозадачный классификатор:

```bash
poetry run python -m ticket_triage_ml.commands train
```

С переопределениями Hydra:

```bash
poetry run python -m ticket_triage_ml.commands train \
    --overrides='["train.max_epochs=10", "train.batch_size=32"]'
```

Результаты обучения:

- Чекпоинты в `checkpoints/`
- Графики в `plots/` (кривые потерь, F1, матрицы ошибок)
- Метрики логируются в MLflow

### Baseline модель

Обучить простую baseline модель (TF-IDF + Logistic Regression):

```bash
poetry run python -m ticket_triage_ml.commands baseline
```

Baseline нужен для:

- Быстрой оценки сложности задачи
- Сравнения с нейросетевой моделью
- Отладки пайплайна

Ожидаемые метрики baseline: accuracy ~0.60-0.70, macro F1 чуть ниже.

### Отслеживание в MLflow

Запустить MLflow сервер (в отдельном терминале):

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Затем открыть http://127.0.0.1:8080 для просмотра экспериментов.

### Экспорт в ONNX

Экспортировать обученную модель в ONNX:

```bash
poetry run python -m ticket_triage_ml.commands export_onnx
```

Результат: `artifacts/model.onnx`

### Экспорт в TensorRT (опционально)

Для NVIDIA GPU с установленным TensorRT:

```bash
chmod +x ticket_triage_ml/production/trt_export.sh
./ticket_triage_ml/production/trt_export.sh --fp16
```

Результат: `artifacts/model.plan`

Примечание: TensorRT опционален и не требуется для инференса.

### Инференс

#### Единичный текст

```bash
poetry run python -m ticket_triage_ml.commands infer \
    --text="Не могу подключиться к VPN с ноутбука"
```

#### Пакетный инференс

```bash
poetry run python -m ticket_triage_ml.commands infer \
    --input_path=data/test_tickets.csv \
    --output_path=predictions.parquet
```

Выходные столбцы:

- `predicted_topic`
- `predicted_priority`
- `topic_scores` (JSON строка)
- `priority_scores` (JSON строка)

## REST API сервис

### Запуск сервера

```bash
poetry run python -m ticket_triage_ml.commands serve
```

Или с указанием порта:

```bash
poetry run python -m ticket_triage_ml.commands serve --port=8080
```

### API эндпоинты

| Эндпоинт   | Метод | Описание                   |
| ---------- | ----- | -------------------------- |
| `/health`  | GET   | Проверка здоровья сервиса  |
| `/ready`   | GET   | Проверка готовности модели |
| `/predict` | POST  | Классификация тикета       |

### Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Cannot access VPN from home"}'
```

### Пример ответа

```json
{
  "topic": "network",
  "priority": "high",
  "topic_scores": { "network": 0.82, "access": 0.1, "other": 0.08 },
  "priority_scores": { "high": 0.76, "medium": 0.18, "low": 0.06 }
}
```

## Docker

### Сборка образа

```bash
docker build -t auto-ticket-classification .
```

### Запуск контейнера

```bash
docker run -p 8000:8000 auto-ticket-classification
```

### Docker Compose

Запуск API + MLflow:

```bash
docker-compose up -d
```

Сервисы будут доступны:

- API: http://localhost:8000
- MLflow: http://localhost:8080

## Конфигурация

Вся конфигурация управляется через Hydra конфиги в `configs/`:

| Файл                        | Описание                   |
| --------------------------- | -------------------------- |
| `config.yaml`               | Главный конфиг с импортами |
| `data.yaml`                 | Пути к данным и настройки  |
| `preprocess.yaml`           | Очистка и разбиение        |
| `train.yaml`                | Гиперпараметры обучения    |
| `infer.yaml`                | Настройки инференса        |
| `export.yaml`               | Настройки экспорта ONNX    |
| `logging.yaml`              | MLflow и графики           |
| `model/multitask_bert.yaml` | Архитектура модели         |

### Ключевые параметры конфигурации

```yaml
# train.yaml
train:
  max_epochs: 3
  batch_size: 16
  learning_rate: 2.0e-5
  accelerator: "auto" # "cpu", "gpu", "auto"

# model/multitask_bert.yaml
model:
  encoder_name: "distilbert-base-uncased"
  freeze_encoder: false
```

## Управление данными через DVC

Файлы данных отслеживаются через DVC (не коммитятся в git):

```bash
# Получить данные из удалённого хранилища
dvc pull

# Добавить новый файл данных
dvc add data/raw/tickets.csv
git add data/raw/tickets.csv.dvc
git commit -m "Добавлены новые обучающие данные"
dvc push
```

## Структура проекта

```text
auto-ticket-classification/
├── ticket_triage_ml/
│   ├── __init__.py
│   ├── commands.py           # CLI точка входа (Fire)
│   ├── api/
│   │   └── app.py            # FastAPI REST сервис
│   ├── baseline/
│   │   └── model.py          # TF-IDF + LogisticRegression
│   ├── data/
│   │   ├── download.py       # Получение данных
│   │   ├── preprocess.py     # Очистка и разбиение
│   │   ├── io.py             # Утилиты чтения/записи
│   │   ├── schema.py         # Pydantic схемы
│   │   └── dvc_utils.py      # DVC хелперы
│   ├── training/
│   │   ├── datamodule.py     # Lightning DataModule
│   │   ├── model.py          # Многозадачный классификатор
│   │   ├── train.py          # Оркестрация обучения
│   │   └── metrics.py        # Вычисление метрик
│   ├── production/
│   │   ├── export_onnx.py    # Экспорт ONNX
│   │   ├── infer_onnx.py     # ONNX Runtime инференс
│   │   ├── mlflow_pyfunc.py  # MLflow обёртка модели
│   │   └── trt_export.sh     # Скрипт экспорта TensorRT
│   └── utils/
│       ├── paths.py          # Утилиты путей
│       ├── git.py            # Отслеживание коммитов
│       ├── logging.py        # MLflow и графики
│       └── seeding.py        # Воспроизводимость
├── configs/                  # Hydra конфигурации
├── plots/                    # Визуализации обучения
├── scripts/
│   └── self_check.sh         # Симуляция грейдера
├── Dockerfile                # Docker образ
├── docker-compose.yaml       # Docker Compose
├── pyproject.toml            # Poetry зависимости
├── .pre-commit-config.yaml   # Pre-commit хуки
└── README.md                 # Документация
```

## Метрики

Модель оценивается по:

- **Accuracy**: По классам и общая
- **Macro F1**: Сбалансированная метрика для несбалансированных классов

Графики, создаваемые во время обучения:

1. `loss_curve.png` - Потери на обучении и валидации
2. `f1_curve.png` - Macro F1 по эпохам
3. `confusion_matrix_topic.png` - Матрица ошибок для тем
4. `confusion_matrix_priority.png` - Матрица ошибок для приоритетов

## Решение проблем

### MLflow сервер недоступен

Если MLflow сервер не запущен, обучение продолжится без трекинга:

```text
WARNING: MLflow setup failed: ...
WARNING: Continuing without MLflow tracking
```

Для запуска MLflow:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

### Нет подключения к интернету

Если данные нельзя скачать, будут сгенерированы синтетические данные:

```text
WARNING: SYNTHETIC FALLBACK: Generating synthetic dataset for smoke-run only
```

Это подходит для тестирования, но не для продакшн моделей.

### CUDA Out of Memory

Уменьшите размер батча:

```bash
poetry run python -m ticket_triage_ml.commands train \
    --overrides='["train.batch_size=8"]'
```

### Ошибки pre-commit

Исправьте проблемы форматирования:

```bash
poetry run black .
poetry run isort .
poetry run flake8
```

## Разработка

### Запуск тестов

```bash
poetry run pytest
```

### Форматирование кода

```bash
poetry run black .
poetry run isort .
poetry run flake8
```

### Самопроверка (симуляция грейдера)

```bash
chmod +x scripts/self_check.sh
./scripts/self_check.sh
```

## Лицензия

MIT License
