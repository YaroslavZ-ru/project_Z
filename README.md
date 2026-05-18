# AI-Terminator

Интеллектуальный помощник по терминам — система для извлечения параметров понятий на основе векторных эмбеддингов и лемматизации.

## Установка

### Требования

- Python 3.10+
- pip (устанавливается вместе с Python)

### Шаги установки

1. **Клонируйте репозиторий** (или скачайте проект)

2. **Создайте виртуальное окружение:**
   ```bash
   python -m venv venv
   ```

3. **Активируйте виртуальное окружение:**
   - Windows (cmd):
     ```cmd
     venv\Scripts\activate
     ```
   - Windows (PowerShell):
     ```powershell
     venv\Scripts\Activate.ps1
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Инициализируйте базу данных:**
   ```bash
   python scripts/init_db.py
   ```

6. **Наполните базу тестовыми данными:**
   ```bash
   python scripts/seed_data.py
   ```

7. **Запустите проект:**
   ```bash
   python main.py
   ```

## Зависимости

Основные зависимости (устанавливаются через `requirements.txt`):

| Пакет | Версия | Назначение |
|-------|--------|------------|
| `pymorphy3` | >=2.0.0 | Морфологический анализатор для русского языка |
| `pymorphy3-dicts-ru` | >=2.4.0 | Словари для pymorphy3 |
| `numpy` | >=1.24.0 | Работа с числовыми массивами |
| `fasttext` | >=0.9.0 | Векторные эмбеддинги (опционально) |
| `faiss-cpu` | >=1.7.0 | Быстрый поиск по векторам (опционально) |
| `transformers` | >=4.30.0 | Генеративные модели (опционально) |
| `torch` | >=2.0.0 | PyTorch для генеративных моделей (опционально) |
| `pytest` | >=7.0.0 | Тестирование |

## Структура проекта

```
project_Z/
├── src/              # Исходный код Python
│   ├── __init__.py
│   ├── config.py     # Класс конфигурации
│   ├── preprocess.py # Предобработка текста (очистка, лемматизация, синонимы)
│   ├── vectorize.py  # Векторизация запроса
│   ├── embeddings.py # FastTextWrapper для эмбеддингов
│   ├── lemmatizer.py # Лемматизатор с LRU-кэшированием
│   ├── synonyms.py   # Словарь синонимов с весами
│   ├── text_cleaner.py # Очистка текста
│   ├── cache.py      # Кэширование (LRU для слов и запросов)
│   ├── knowledge_base.py # Класс для работы с БД
│   ├── search.py     # Поиск в базе знаний
│   ├── fallback.py   # Fallback-режим с шаблонами
│   ├── aggregation.py # Агрегация и ранжирование параметров
│   └── __pycache__/
├── scripts/          # Скрипты инициализации
│   ├── init_db.py    # Инициализация БД
│   └── seed_data.py  # Наполнение тестовыми данными
├── data/             # Базы данных, словари синонимов
│   ├── synonyms.json # Пример словаря синонимов
│   └── knowledge_base.db # SQLite база знаний
├── models/           # Модели эмбеддингов (fastText и др.)
├── tests/            # Тесты
│   ├── test_lemmatizer.py
│   ├── test_synonyms.py
│   ├── test_text_cleaner.py
│   ├── test_preprocess.py
│   ├── test_vectorize.py
│   ├── test_embeddings.py
│   ├── test_cache.py
│   ├── test_pipeline_integration.py
│   ├── test_knowledge_base.py
│   ├── test_search.py
│   ├── test_fallback.py
│   ├── test_aggregation.py
│   └── __pycache__/
├── logs/             # Логи
├── configs/          # Конфигурационные файлы
│   ├── config.json           # Основная конфигурация
│   └── domain_templates.json # Шаблоны предметных областей
├── main.py           # Точка входа
├── requirements.txt  # Зависимости
└── README.md         # Документация
```

## Использование

Запустите основной скрипт:

```bash
python main.py
```

### Пример входных данных

В файле `main.py` заданы параметры по умолчанию:

```python
test_input = {
    "term": "ключи",
    "hints": ["техника", "вращение"],
    "debug": False,
}
```

### Пример вывода

```json
{
  "status": "ok",
  "term": "ключи",
  "selected_context": {
    "domain": "техника",
    "confidence": 0.4
  },
  "parameters": [
    {
      "name": "power_source",
      "label_ru": "Источник энергии",
      "type": "string",
      "description": "Ручной, электрический, пневматический",
      "confidence": 0.4,
      "source": "template"
    },
    {
      "name": "material",
      "label_ru": "Материал",
      "type": "string",
      "description": "Материал изготовления",
      "confidence": 0.4,
      "source": "template"
    }
  ],
  "suggested_refinements": [
    "Можно добавить параметр 'мощность' или 'тип привода'"
  ],
  "warnings": [
    "Термин не найден в базе знаний, параметры предложены на основе шаблона предметной области",
    "Понятие не найдено в базе знаний, параметры предположительные"
  ],
  "session_id": "70759923211a41b7a4abceae4dc1b733"
}
```

## Формат коммитов

Проект использует **Conventional Commits** для стандартизации сообщений.

### Типы коммитов

| Тип | Описание |
|-----|----------|
| `feat` | Новая функция (feature) |
| `fix` | Исправление бага |
| `docs` | Изменения в документации |
| `refactor` | Рефакторинг кода |
| `test` | Добавление или изменение тестов |
| `chore` | Технические правки (обновление зависимостей и т.д.) |

### Формат

```
<тип>: <краткое описание>

<подробности>
```

### Примеры

```
feat: добавлен модуль векторизации с fastText

Реализован FastTextWrapper с LRU-кэшем слов, поддержкой fallback-словаря
и методами get_word_vector/get_phrase_vector.
```

```
fix: исправлена ошибка в обработке пустых подсказок

При передаче пустого списка hints система теперь корректно
обрабатывает запрос без дополнительных параметров.
```

```
docs: обновлена документация с описанием новых функций

Добавлены разделы для генеративного режима, сессий,
обработки омонимии и fallback через центроиды.
```

## Конфигурация

Конфигурация хранится в `configs/config.json`. Основные параметры:

| Параметр | Тип | Описание |
|----------|-----|----------|
| `db_path` | string | Путь к базе данных |
| `fasttext_model_path` | string | Путь к модели fastText |
| `synonyms_path` | string | Путь к словарю синонимов |
| `min_confidence` | float | Минимальное пороговое значение сходства |
| `max_candidates` | int | Максимальное количество кандидатов |
| `max_parameters` | int | Максимальное количество параметров |
| `use_generative` | bool | Включить генеративный режим |
| `generative_model` | string | Имя модели в Hugging Face Hub |
| `ambiguity_threshold` | float | Порог для обнаружения омонимии |
| `ambiguity_delta` | float | Максимальная разница для признания омонимии |
| `domain_centroid_threshold` | float | Порог сходства для fallback через центроид |
| `session_ttl_seconds` | int | Время жизни сессии без активности |
| `session_cache_size` | int | Максимальное количество сессий |

Полный список параметров смотрите в `configs/config.json`.
