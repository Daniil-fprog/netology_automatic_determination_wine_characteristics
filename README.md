# Автоматическое определение характеристик вина

ML-проект определяет характеристики вина по русскоязычному текстовому описанию и предоставляет модель через минимальный FastAPI-сервис.

Предсказываемые признаки:

- категориальные: кислотность, танинность, плотность и интенсивность вкуса;
- мультиметочные: ароматические и вкусовые ноты.

## Статус данных и честное ограничение

`data/wines_annotated.csv` содержит 3 036 строк с частичной LLM-разметкой. После удаления повторных текстов доступно 770 меток кислотности, 1 518 — танинности, 409 — тела и 1 269 — интенсивности. Каждая модель обучается на своей подвыборке, поэтому полностью заполненные строки не требуются.

Текущий bundle работает в режиме `supervised`, но LLM-метки остаются псевдоразметкой. Метрики показывают согласие модели с этой разметкой, а не экспертно подтверждённую точность.

## Результаты

После очистки использовано 3 030 уникальных текстов. На hold-out-части macro-F1 составил 0.720 для кислотности, 0.685 для танинности, 0.773 для тела и 0.676 для интенсивности. Micro-F1 мультиметок равен 0.547 для ароматов и 0.440 для вкусовых нот. Подробный протокол и интерпретация находятся в [отчёте об эксперименте](docs/experiments.md).

## Структура репозитория

```text
.
├── artifacts/                  # модель, метрики и тестовые прогнозы
├── data/wines.csv              # исходный набор данных
├── data/wines_annotated.csv    # частичная LLM-разметка
├── docs/                       # архитектура, API, деплой, мониторинг, тестирование
├── notebooks/
│   ├── 01_wine_characteristics_project.ipynb
│   └── 02_model_inference.ipynb
├── src/wine_characteristics/   # inference-пакет и FastAPI
├── tests/                      # unit- и API-тесты
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Ноутбуки

- [`01_wine_characteristics_project.ipynb`](notebooks/01_wine_characteristics_project.ipynb) — аудит и EDA, подготовка текста, слабая/ручная разметка, обучение, метрики, анализ ошибок и сохранение модели.
- [`02_model_inference.ipynb`](notebooks/02_model_inference.ipynb) — загрузка bundle и прогноз для нового описания.

Оба ноутбука выполнены на текущем датасете, содержат сохранённые графики и выводы и завершаются без ошибок.

## Быстрый запуск API

Требуется Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
uvicorn wine_characteristics.api:app --reload
```

Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Пример запроса:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"description":"Красное вино с ароматом вишни и пряностей, плотным вкусом и мягкими танинами."}'
```

Основные endpoint-ы:

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | готовность сервиса и модели |
| GET | `/model/info` | режим, цели и метаданные модели |
| POST | `/predict` | прогноз одного описания |
| POST | `/predict/batch` | пакет до 100 описаний |
| GET | `/metrics` | технические Prometheus-метрики |

Полный контракт: [docs/api.md](docs/api.md).

## Тестирование

```bash
pytest
```

Автоматизированы загрузка bundle, нормализация текста, все API-контракты, валидация, batch-инференс, healthcheck и metrics. Дополнительный план: [docs/testing.md](docs/testing.md).

## Docker

```bash
docker build -t wine-characteristics-api .
docker run --rm -p 8000:8000 wine-characteristics-api
```

Путь к артефакту можно изменить переменной окружения `MODEL_PATH`.

## Документация решения

- [Итоговый отчёт 7 страниц](reports/wine_characteristics_report.docx)
- [Архитектура](docs/architecture.md)
- [Материалы эксперимента](docs/experiments.md)
- [API и примеры контрактов](docs/api.md)
- [Деплой и CI/CD](docs/deployment.md)
- [Мониторинг качества и сервиса](docs/monitoring.md)
- [Тесты и план тестирования](docs/testing.md)

## Артефакты

- `artifacts/wine_characteristics_model.joblib` — TF–IDF, классификаторы, пороги и метаданные;
- `artifacts/categorical_metrics.csv` — метрики четырёх категориальных целей;
- `artifacts/multilabel_metrics.csv` — метрики ароматических и вкусовых нот;
- `artifacts/test_predictions.csv` — данные для анализа ошибок на hold-out.

## Дальнейшее развитие

1. Разметить репрезентативную часть корпуса с участием винного эксперта и измерить согласованность разметчиков.
2. Зафиксировать допустимые уровни и единую таксономию нот.
3. Выполнить group split, подобрать пороги на validation и один раз оценить независимый test.
4. Сравнить baseline с ruBERT/Sentence Transformers.
5. Версионировать датасет, таксономию, код и bundle; добавить canary-деплой и мониторинг дрейфа.
