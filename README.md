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
- `fasttext>=0.9.2` - векторные эмбеддинги (опционально)
- `pytest>=7.0.0` - тестирование

## Структура проекта

```
project_Z/
├── src/              # Исходный код Python
│   ├── __init__.py
│   ├── config.py     # Класс конфигурации
│   ├── preprocess.py # Предобработка текста
│   ├── vectorize.py  # Векторизация запроса
│   ├── embeddings.py # FastTextWrapper для эмбеддингов
│   ├── lemmatizer.py # Лемматизатор с кэшированием
│   ├── synonyms.py   # Словарь синонимов
│   ├── text_cleaner.py # Очистка текста
│   ├── knowledge_base.py # Класс для работы с БД
│   ├── search.py     # Поиск в базе знаний
│   └── fallback.py   # Fallback-режим с шаблонами
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
│   └── test_fallback.py
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
    "log_level": "INFO"
}
```

## Алгоритм работы

1. **Предобработка** - очистка текста, лемматизация, расширение синонимами
2. **Векторизация** - вычисление взвешенного вектора запроса
3. **Поиск** - поиск похожих понятий в базе знаний
4. **Fallback** - если ничего не найдено, используется шаблонная генерация

## Тестирование

Запустить все тесты:
```bash
pytest tests/ -v
```

В проекте 130 тестов, покрывающих все модули и частные случаи.

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
