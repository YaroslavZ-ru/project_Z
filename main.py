#!/usr/bin/env python3
"""Точка входа в приложение AI-Terminator.

Принимает входные данные (термин и подсказки), возвращает структурированный
ответ с параметрами понятия.
"""

import json
import sys
from pathlib import Path

# Добавляем корень проекта в path для импортов
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config


def process_term(input_data: dict) -> dict:
    """Обработать входные данные и вернуть фиктивный ответ.

    Пока реализована заглушка с фиктивными данными.

    Args:
        input_data: Словарь с полями:
            - term: str - анализируемый термин
            - hints: list[str] - уточняющие слова (опционально)
            - debug: bool - режим отладки (опционально)

    Returns:
        dict: Структурированный ответ с полями:
            - status: str - статус выполнения
            - term: str - обработанный термин
            - selected_context: dict - выбранный контекст
            - parameters: list - список параметров
            - suggested_refinements: list - предложения по уточнению
            - warnings: list - предупреждения
    """
    term = input_data.get("term", "")
    hints = input_data.get("hints", [])
    debug = input_data.get("debug", False)

    # Фиктивный ответ в соответствии со спецификацией
    response = {
        "status": "ok",
        "term": term,
        "selected_context": {"domain": "не определено", "confidence": 0.0},
        "parameters": [],
        "suggested_refinements": [],
        "warnings": ["Алгоритм ещё не реализован"],
    }

    if debug:
        response["debug_info"] = {
            "query_vector": [0.0] * 10,
            "candidates_raw": [],
            "scores_distribution": [],
        }

    return response


def main():
    """Основная точка входа."""
    # Пример входных данных
    test_input = {
        "term": "ключ",
        "hints": ["техника", "вращение"],
        "debug": False,
    }

    print("AI-Terminator: запуск...")

    # Загрузка конфигурации
    try:
        config = Config.from_json("configs/config.json")
        print(f"Конфигурация загружена: {config.log_level}")
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")
        return

    # Обработка входных данных
    result = process_term(test_input)

    # Вывод результата
    print("\nРезультат обработки:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()