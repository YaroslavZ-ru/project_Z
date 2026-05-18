# AI-Terminator

Интеллектуальный помощник по терминам - система для извлечения параметров понятий на основе векторных эмбеддингов и лемматизации.

## Установка

### Требования

- Python 3.10+
- pip (установлен с Python)

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

Основные зависимости (устанавливаются через requirements.txt):

- `pymorphy3>=2.0.0` - морфологический анализатор для русского языка
- `pymorphy3-dicts-ru>=2.4.0` - словари для pymorphy3
- `numpy>=1.24.0` - работа с числовыми массивами
- `fasttext>=0.9.0` - векторные эмбеддинги (опционально)
- `faiss-cpu>=1.7.0` - быстрый поиск по векторам (опционально)
- `transformers>=4.30.0` - генеративные модели (опционально)
- `torch>=2.0.0` - PyTorch для генеративных моделей (опционально)
- `pytest>=7.0.0` - тестирование

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

Пример входных данных (в main.py):
```python
test_input = {
    "term": "ключи",
    "hints": ["техника", "вращение"],
    "debug": False,
}
```

### Конфигурация

Конфигурация хранится в `configs/config.json`:

```json
{
    "db_path": "data/knowledge_base.db",
    "fasttext_model_path": "models/cc.ru.300.bin",
    "synonyms_path": "data/synonyms.json",
    "domain_templates_path": "configs/domain_templates.json",
    "min_confidence": 0.3,
    "max_candidates": 20,
    "max_parameters": 15,
    "use_generative": false,
    "generative_model": "rugpt3small_based_on_gpt2",
    "timeout_seconds": 2.0,
    "cache_embeddings": true,
    "log_level": "INFO",
    "cache_lemma_size": 1000,
    "max_synonyms_per_token": 2,
    "use_synonyms": true,
    "max_term_length": 100,
    "max_hint_length": 50
}
```

### Параметры конфигурации

| Параметр | Тип | Описание |
|----------|-----|----------|
| `cache_lemma_size` | int | Размер LRU-кэша лемматизатора (по умолчанию 1000) |
| `max_synonyms_per_token` | int | Максимальное количество синонимов на токен (по умолчанию 2) |
| `use_synonyms` | bool | Включить использование синонимов (по умолчанию true) |
| `max_term_length` | int | Максимальная длина термина (по умолчанию 100) |
| `max_hint_length` | int | Максимальная длина подсказки (по умолчанию 50) |

## Алгоритм работы

1. **Предобработка** - очистка текста, лемматизация, расширение синонимами с весами (0.7/0.3/0.1)
2. **Векторизация** - вычисление взвешенного вектора запроса через fastText
3. **Поиск** - поиск похожих понятий в базе знаний (косинусное сходство)
4. **Агрегация** - группировка и ранжирование параметров
5. **Fallback** - если ничего не найдено, используется шаблонная генерация

### Веса токенов

- Исходные слова термина: суммарный вес 0.7 (распределяется равномерно)
- Исходные слова подсказок: суммарный вес 0.3 (распределяется равномерно)
- Синонимы: суммарный вес 0.1 (распределяется равномерно)

## Тестирование

Запустить все тесты:
```bash
pytest tests/ -v
```

В проекте 144 теста, покрывающих все модули и частные случаи.

### Тесты изменений 6-10

- `test_lemmatizer.py` - 15 тестов (LRU-кэш, дефисы, составные слова)
- `test_synonyms.py` - 13 тестов (загрузка, сортировка, ограничение)
- `test_text_cleaner.py` - 22 теста (очистка, символы, пробелы)
- `test_preprocess.py` - 24 теста (валидация, длина, синонимы, веса)

## Формат коммитов

Проект использует **Conventional Commits** для стандартизации сообщений коммитов.

### Типы коммитов

- `feat` — новая функция (feature)
- `fix` — исправление бага
- `docs` — изменения в документации
- `refactor` — рефакторинг кода
- `test` — добавление или изменение тестов
- `chore` — технические правки (обновление зависимостей и т.д.)

### Формат

```
<тип>: <краткое описание>

<подробности>
Изменение: #<номер>
Тесты: <перечислены>
```
