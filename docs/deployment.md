# Деплой

## Локально

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
uvicorn wine_characteristics.api:app --reload
```

Переменная `MODEL_PATH` позволяет указать другой `joblib`-артефакт.

## Docker

```bash
docker build -t wine-characteristics-api .
docker run --rm -p 8000:8000 wine-characteristics-api
```

Для production рекомендуется хранить версионированную модель в объектном хранилище или model registry, загружать её на этапе сборки/старта, запускать контейнер без root, ограничивать CPU/RAM и размещать сервис за reverse proxy с TLS и лимитом размера запроса.

## CI/CD

Минимальный pipeline: установка зависимостей → `pytest` → smoke-загрузка модели → сборка Docker → сканирование образа → публикация → деплой canary → проверка `/health` → постепенное переключение трафика. Откат выполняется переключением на предыдущий образ и версию bundle.
