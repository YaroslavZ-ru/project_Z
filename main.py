#!/usr/bin/env python3
"""Точка входа в приложение AI-Terminator.

Принимает входные данные (термин и подсказки), возвращает структурированный
ответ с параметрами понятия.
"""

import json
import logging
import sys
from pathlib import Path
from functools import lru_cache

import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Добавляем корень проекта в path для импортов
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.preprocess import preprocess
from src.embeddings import FastTextWrapper
from src.vectorize import vectorize


def _compute_query_vector(term: str, hints_tuple: tuple, config: Config) -> np.ndarray:
    """Вычислить вектор запроса (внутренняя функция для кэширования).

    Args:
        term: Анализируемый термин.
        hints_tuple: Кортеж уточняющих слов (для хешируемости в lru_cache).
        config: Экземпляр конфигурации.

    Returns:
        Нормализованный вектор запроса.
    """
    # Предобработка
    processed = preprocess(term, list(hints_tuple))

    if processed.get("status") == "error":
        raise ValueError(processed.get("message", "Ошибка предобработки"))

    # Векторизация
    emb_model = FastTextWrapper(
        str(config.fasttext_model_path),
        str(config.db_path.parent / "models" / "static_embeddings.npy")
    )
    query_vector = vectorize(processed, emb_model, normalize=True)

    return query_vector


@lru_cache(maxsize=100)
def _cached_vector(term: str, hints_tuple: tuple, config_hash: int) -> tuple:
    """Кэшированная функция для получения вектора запроса.

    Возвращает кортеж чисел вместо ndarray для возможности кэширования.

    Args:
        term: Анализируемый термин.
        hints_tuple: Кортеж уточняющих слов.
        config_hash: Хеш конфигурации (для инвалидации кэша при изменении).

    Returns:
        Кортеж чисел (вектор).
    """
    # Создаем фиктивный config для получения хеша
    # В реальном проекте можно использовать версию конфига
    _ = config_hash

    # Вычисляем вектор
    processed = preprocess(term, list(hints_tuple))
    if processed.get("status") == "error":
        raise ValueError(processed.get("message", "Ошибка предобработки"))

    emb_model = FastTextWrapper(
        "models/cc.ru.300.bin",
        "models/static_embeddings.npy"
    )
    query_vector = vectorize(processed, emb_model, normalize=True)

    # Преобразуем в кортеж для кэширования
    return tuple(query_vector.tolist())


def run_pipeline(term: str, hints: list[str], config: Config, debug: bool = False) -> dict:
    """Запустить полный конвейер обработки термина.

    Args:
        term: Анализируемый термин.
        hints: Список уточняющих слов (0-3 слова).
        config: Экземпляр конфигурации.
        debug: Режим отладки (добавить промежуточные данные в ответ).

    Returns:
        dict: Структурированный ответ.
    """
    # Проверка кэша
    hints_tuple = tuple(hints)
    try:
        cached_vec = _cached_vector(term, hints_tuple, hash(str(config.to_dict())))
        query_vector = np.array(cached_vec, dtype=np.float32)
        logger.info("Вектор запроса получен из кэша")
    except (ValueError, FileNotFoundError) as e:
        # Кэш не сработал или ошибка - вычисляем заново
        try:
            query_vector = _compute_query_vector(term, hints_tuple, config)
        except ValueError as e:
            return {
                "status": "error",
                "message": str(e),
                "term": term,
                "selected_context": {"domain": "не определено", "confidence": 0.0},
                "parameters": [],
                "suggested_refinements": [],
                "warnings": [str(e)],
            }

    # Фиктивный ответ в соответствии со спецификацией
    response = {
        "status": "ok",
        "term": term,
        "selected_context": {"domain": "не определено", "confidence": 0.0},
        "parameters": [],
        "suggested_refinements": [],
        "warnings": [],
    }

    if debug:
        response["debug_info"] = {
            "query_vector": query_vector[:10].tolist(),  # Первые 10 значений
            "candidates_raw": [],
            "scores_distribution": [],
        }

    return response


def main():
    """Основная точка входа."""
    # Пример входных данных
    test_input = {
        "term": "ключи",
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
    result = run_pipeline(
        test_input["term"],
        test_input.get("hints", []),
        config,
        debug=test_input.get("debug", False)
    )

    # Вывод результата
    print("\nРезультат обработки:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()