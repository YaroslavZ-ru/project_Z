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

5. **Запустите проект:**
   ```bash
   python main.py
   ```

## Зависимости

Основные зависимости (устанавливаются через requirements.txt):

- `pymorphy3>=2.0.0` - морфологический анализатор для русского языка
- `pymorphy3-dicts-ru>=2.4.0` - словари для pymorphy3
- `numpy>=1.24.0` - работа с числовыми массивами
- `scikit-learn>=1.8.0` - машинное обучение
- `pytest>=7.0.0` - тестирование

## Структура проекта

```
project_Z/
├── src/              # Исходный код Python
│   ├── __init__.py
│   ├── config.py     # Класс конфигурации
│   ├── preprocess.py # Предобработка текста
│   ├── vectorize.py  # Векторизация запроса
│   └── search.py     # Поиск в базе знаний
├── data/             # Базы данных, словари синонимов
├── models/           # Модели эмбеддингов (fastText и др.)
├── tests/            # Тесты
├── logs/             # Логи
├── configs/          # Конфигурационные файлы
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

## Тестирование

Запустить все тесты:
```bash
pytest tests/ -v
```

## Лицензия

MIT