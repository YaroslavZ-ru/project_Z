"""Модуль векторизации запроса для AI-Terminator.

Предоставляет функции для преобразования токенов в векторы эмбеддингов.
"""

import logging
import numpy as np
from typing import Optional

from src.embeddings import FastTextWrapper

logger = logging.getLogger(__name__)


def vectorize(
    processed_query: dict,
    embedding_model: FastTextWrapper,
    normalize: bool = True,
) -> np.ndarray:
    """Векторизовать предобработанный запрос.

    Формула (согласно ТЗ):
    - Исходный термин: вес 0.7 (распределяется равномерно между словами термина)
    - Подсказки: суммарный вес 0.3 (распределяется равномерно между всеми словами подсказок)
    - Синонимы: суммарный вес 0.1 (распределяется равномерно между всеми синонимами)

    ВАЖНО: Веса не делятся на сумму весов - используется прямое взвешивание,
    затем L2-нормализация.

    Args:
        processed_query: Словарь с предобработанными данными, должен содержать
            ключ "tokens_with_weights" - список кортежей (token, weight).
        embedding_model: Экземпляр FastTextWrapper для получения векторов.
        normalize: Применять L2-нормализацию (по умолчанию True).

    Returns:
        Нормализованный вектор запроса (np.float32, размерность 300).
    """
    tokens_weights = processed_query.get("tokens_with_weights", [])
    dim = embedding_model.get_dimension()

    if not tokens_weights:
        logger.warning("Нет токенов с весами, возвращаем нулевой вектор")
        return np.zeros(dim, dtype=np.float32)

    # Инициализируем нулевой вектор (float64 для точности)
    weighted_sum = np.zeros(dim, dtype=np.float64)

    for token, weight in tokens_weights:
        # Проверка веса на корректность
        if not isinstance(weight, (int, float)):
            logger.warning(f"Некорректный тип веса {weight} для токена {token}, пропускаем")
            continue
        if np.isnan(weight) or np.isinf(weight):
            logger.warning(f"Некорректный вес {weight} для токена {token}, пропускаем")
            continue

        # Получаем вектор (поддержка фраз через get_phrase_vector)
        vec = embedding_model.get_phrase_vector(token)
        weighted_sum += weight * vec

    if normalize:
        norm = np.linalg.norm(weighted_sum)
        if norm > 1e-9:
            weighted_sum = weighted_sum / norm
        else:
            logger.warning("Нулевой вектор после взвешивания, нормализация не изменит")

    # Приведение к float32
    return weighted_sum.astype(np.float32)