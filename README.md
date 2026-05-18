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

### Опциональные зависимости для новых функций

- `transformers`, `torch` - для генеративного режима (изменение 20)
- `pymorphy3`, `pymorphy3-dicts-ru` - для лемматизации и работы с синонимами
- `sqlite3` - для работы с базой знаний (встроен в Python)

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
│   ├── pipeline.py   # Основной конвейер обработки
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

Пример входных данных (в main.py):
```python
test_input = {
    "term": "ключи",
    "hints": ["техника", "вращение"],
    "debug": False,
}
```

### Новые возможности (изменения 20-25)

#### Генеративный режим
При недостатке параметров из базы знаний система может использовать генеративную модель (rugpt3small) для дополнения параметров. Включается через `use_generative: true`.

#### Сессии для интерактивного уточнения
Система поддерживает пошаговое уточнение понятия без потери контекста. При повторных запросах с тем же `session_id` подсказки накапливаются, а выбранный домен сохраняется.

#### Обработка омонимии
Если термин может относиться к нескольким доменам с одинаковой вероятностью, система возвращает специальный ответ со списком возможных доменов и просит пользователя уточнить.

#### Связи между понятиями
Система поддерживает связи между понятиями (is_a, part_of, related_to, synonym) для расширения поиска и построения онтологий.

#### Fallback через центроиды
При отсутствии кандидатов система определяет наиболее вероятный домен на основе центроидов (средних векторов понятий в домене), что точнее, чем fallback по ключевым словам.

### Конфигурация

Конфигурация хранится в `configs/config.json`:

```json
{
    "db_path": "data/knowledge_base.db",
    "fasttext_model_path": "models/cc.ru.300.bin",
    "fasttext_mmap": true,
    "fallback_embeddings_path": "models/fallback_embeddings.npy",
    "synonyms_path": "data/synonyms.json",
    "domain_templates_path": "configs/domain_templates.json",
    "min_confidence": 0.3,
    "max_candidates": 20,
    "max_parameters": 15,
    "use_generative": false,
    "generative_model": "rugpt3small_based_on_gpt2",
    "generative_max_new_tokens": 100,
    "generative_temperature": 0.7,
    "generative_max_new_params": 3,
    "generative_timeout_seconds": 2.0,
    "min_parameters_for_generative": 5,
    "generative_keywords": ["материал", "размер", "тип", "скорость", "мощность", "вес", "длина", "цвет", "напряжение", "ёмкость", "частота", "температура", "давление", "форма", "покрытие", "привод", "источник энергии"],
    "timeout_seconds": 2.0,
    "cache_embeddings": true,
    "log_level": "INFO",
    "cache_lemma_size": 1000,
    "word_vector_cache_size": 20000,
    "query_cache_size": 100,
    "max_synonyms_per_token": 2,
    "use_synonyms": true,
    "max_term_length": 100,
    "max_hint_length": 50,
    "use_faiss": false,
    "faiss_index_path": "",
    "session_ttl_seconds": 1800,
    "session_cache_size": 1000,
    "session_cleanup_interval_seconds": 60,
    "auto_save_domain_on_ok": true,
    "ambiguity_threshold": 0.7,
    "ambiguity_delta": 0.1,
    "domain_centroid_threshold": 0.3,
    "auto_save_domain_on_fallback": false,
    "use_relations": false,
    "relation_max_depth": 1,
    "relation_decay_factor": 0.5,
    "domain_centroids_min_concepts": 2
}
```

### Параметры конфигурации

| Параметр | Тип | Описание |
|----------|-----|----------|
| `cache_lemma_size` | int | Размер LRU-кэша лемматизатора (по умолчанию 1000) |
| `word_vector_cache_size` | int | Размер LRU-кэша векторов слов (по умолчанию 20000) |
| `query_cache_size` | int | Размер LRU-кэша векторов запросов (по умолчанию 100) |
| `max_synonyms_per_token` | int | Максимальное количество синонимов на токен (по умолчанию 2) |
| `use_synonyms` | bool | Включить использование синонимов (по умолчанию true) |
| `max_term_length` | int | Максимальная длина термина (по умолчанию 100) |
| `max_hint_length` | int | Максимальная длина подсказки (по умолчанию 50) |
| `fasttext_model_path` | string | Путь к модели fastText (.bin) |
| `fasttext_mmap` | bool | Использовать memory-mapping для fastText (по умолчанию true) |
| `fallback_embeddings_path` | string | Путь к fallback-словарю эмбеддингов (.npy) |
| `use_generative` | bool | Включить генеративный режим (по умолчанию false) |
| `generative_model` | string | Имя модели в Hugging Face Hub (по умолчанию rugpt3small_based_on_gpt2) |
| `generative_max_new_tokens` | int | Максимальное количество новых токенов (по умолчанию 100) |
| `generative_temperature` | float | Температура генерации (по умолчанию 0.7) |
| `generative_max_new_params` | int | Максимальное количество новых параметров (по умолчанию 3) |
| `generative_timeout_seconds` | float | Таймаут на генерацию (по умолчанию 2.0) |
| `min_parameters_for_generative` | int | Минимум параметров для запуска генерации (по умолчанию 5) |
| `generative_keywords` | list | Ключевые слова для извлечения параметров |
| `ambiguity_threshold` | float | Порог для обнаружения омонимии (по умолчанию 0.7) |
| `ambiguity_delta` | float | Максимальная разница для признания омонимии (по умолчанию 0.1) |
| `domain_centroid_threshold` | float | Порог сходства для fallback через центроид (по умолчанию 0.3) |
| `auto_save_domain_on_fallback` | bool | Сохранять домен из fallback в сессию (по умолчанию false) |
| `use_relations` | bool | Использовать связи между понятиями (по умолчанию false) |
| `relation_max_depth` | int | Максимальная глубина обхода связей (по умолчанию 1) |
| `relation_decay_factor` | float | Коэффициент затухания similarity (по умолчанию 0.5) |
| `domain_centroids_min_concepts` | int | Минимум понятий для центроида (по умолчанию 2) |
| `session_ttl_seconds` | int | Время жизни сессии без активности (по умолчанию 1800) |
| `session_cache_size` | int | Максимальное количество сессий (по умолчанию 1000) |
| `session_cleanup_interval_seconds` | int | Интервал фоновой очистки (по умолчанию 60) |
| `auto_save_domain_on_ok` | bool | Автоматически сохранять домен из успешного ответа (по умолчанию true) |

## Алгоритм работы

1. **Предобработка** - очистка текста, лемматизация, расширение синонимами с весами (0.7/0.3/0.1)
2. **Векторизация** - вычисление взвешенного вектора запроса через fastText с LRU-кэшированием слов
3. **Кэширование запросов** - сохранение векторов для повторяющихся запросов (LRU)
4. **Поиск** - поиск похожих понятий в базе знаний (косинусное сходство, с поддержкой domain_filter)
5. **Агрегация** - группировка и ранжирование параметров
6. **Омонимия** - обнаружение и обработка множественных контекстов
7. **Fallback** - если ничего не найдено, используется шаблонная генерация с центроидами доменов

### Веса токенов

- Исходные слова термина: суммарный вес 0.7 (распределяется равномерно)
- Исходные слова подсказок: суммарный вес 0.3 (распределяется равномерно)
- Синонимы: суммарный вес 0.1 (распределяется равномерно)

### Кэширование

- **LRU-кэш лемматизатора** (`cache_lemma_size`): хранит результаты лемматизации слов
- **LRU-кэш векторов слов** (`word_vector_cache_size`): хранит векторы для отдельных слов из fastText
- **LRU-кэш запросов** (`query_cache_size`): хранит итоговые векторы запросов для повторяющихся входов

### Сессии

Сессии позволяют накапливать подсказки и запоминать выбранный домен:
- `session_ttl_seconds`: время жизни сессии без активности (по умолчанию 30 минут)
- `session_cache_size`: максимальное количество сессий в памяти
- `auto_save_domain_on_ok`: автоматически сохранять домен из успешного ответа

### Генеративный режим

При недостатке параметров (< min_parameters_for_generative) система использует генеративную модель для дополнения:
- Модель: rugpt3small_based_on_gpt2 (или другая из Hugging Face Hub)
- Параметры генерации: max_new_tokens, temperature, num_return_sequences=3
- Извлечение параметров через ключевые слова (материал, размер, тип и т.д.)

### Обработка омонимии

Если найдено несколько доменов с близкой уверенностью (разница < ambiguity_delta), система возвращает статус "ambiguous" со списком кандидатов для уточнения.

### Связи между понятиями

При включенном `use_relations` система расширяет кандидатов через связи:
- Типы связей: related_to, synonym (настраивается)
- Глубина обхода: relation_max_depth (по умолчанию 1)
- Затухание similarity: relation_decay_factor (по умолчанию 0.5)

### Fallback через центроиды

При отсутствии кандидатов система определяет домен через центроиды:
- Центроид = средний вектор всех понятий в домене
- Выбирается домен с максимальным сходством (>= domain_centroid_threshold)
- Если сходство недостаточно, используется fallback по ключевым словам

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

### Тесты изменений 11-15

- `test_embeddings.py` - 18 тестов (загрузка fastText, fallback, LRU-кэш слов, фразы)
- `test_vectorize.py` - 12 тестов (формула взвешенной суммы, L2-нормализация, обработка ошибок)
- `test_cache.py` - 10 тестов (LRU-кэш запросов, инвалидация при изменении конфига)
- `test_pipeline_integration.py` - 16 тестов (полный конвейер, debug-режим, моки)

## Тесты изменений 16-19

- `test_aggregation.py` - 14 тестов (агрегация параметров, ранжирование, омонимия)
- `test_fallback.py` - 12 тестов (шаблоны, ключевые слова, fallback-режим)
- `test_knowledge_base.py` - 18 тестов (CRUD операции, эмбеддинги, связи)
- `test_search.py` - 10 тестов (поиск, FAISS, фильтрация по домену)

## Тесты изменений 20-25

- `test_sessions.py` - 12 тестов (создание, получение, обновление, TTL, вытеснение)
- `test_generative.py` - 10 тестов (загрузка модели, генерация, извлечение параметров)
- `test_omonymy.py` - 8 тестов (обнаружение омонимии, фильтрация по домену)
- `test_relations.py` - 8 тестов (добавление связей, расширение кандидатов)
- `test_centroids.py` - 6 тестов (вычисление центроидов, fallback через центроиды)

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
Тесты: <перечислены>
```
