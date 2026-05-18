"""Модуль кэширования для AI-Terminator.

Предоставляет класс QueryVectorCache для кэширования векторов запросов.
"""

import hashlib
import logging
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QueryVectorCache:
    """Кэш для векторов запросов с LRU-стратегией.

    Кэширует результат preprocess + vectorize для повторяющихся запросов.
    Ключ: (term, tuple(sorted(hints)), config_hash).

    Attributes:
        max_size: Максимальное количество элементов в кэше.
    """

    def __init__(self, max_size: int = 100):
        """Инициализация кэша.

        Args:
            max_size: Максимальное количество элементов в кэше.
        """
        self.max_size = max_size
        self._cache: Dict[Tuple[str, tuple, int], Tuple[tuple, Dict[str, Any]]] = {}
        self._order: list[Tuple[str, tuple, int]] = []  # Для LRU

    def _compute_config_hash(self, config: Any) -> int:
        """Вычислить хеш конфигурации для инвалидации кэша.

        Args:
            config: Объект конфигурации.

        Returns:
            Хеш конфигурации.
        """
        # Параметры, влияющие на векторизацию
        params = {
            "use_synonyms": getattr(config, "use_synonyms", True),
            "max_synonyms_per_token": getattr(config, "max_synonyms_per_token", 2),
        }
        # Сериализуем в JSON и берем MD5
        import json
        config_str = json.dumps(params, sort_keys=True)
        return int(hashlib.md5(config_str.encode()).hexdigest(), 16) & 0xFFFFFFFF

    def _normalize_hints(self, hints: list[str]) -> tuple:
        """Нормализовать подсказки для ключа кэша.

        Args:
            hints: Список подсказок.

        Returns:
            Отсортированный кортеж подсказок.
        """
        return tuple(sorted(hints))

    def get(
        self, term: str, hints: list[str], config: Any
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Получить вектор из кэша.

        Args:
            term: Термин.
            hints: Список подсказок.
            config: Объект конфигурации.

        Returns:
            Кортеж (вектор, промежуточные данные) или None, если нет в кэше.
        """
        key = (term, self._normalize_hints(hints), self._compute_config_hash(config))

        if key in self._cache:
            # Обновляем порядок (LRU)
            self._order.remove(key)
            self._order.append(key)
            logger.debug(f"Кэш попадание: term='{term}', hints={hints}")
            vec_tuple, result = self._cache[key]
            return np.array(vec_tuple, dtype=np.float32), result

        logger.debug(f"Кэш промах: term='{term}', hints={hints}")
        return None

    def set(
        self,
        term: str,
        hints: list[str],
        config: Any,
        vector: np.ndarray,
        result: Dict[str, Any],
    ) -> None:
        """Добавить вектор в кэш.

        Args:
            term: Термин.
            hints: Список подсказок.
            config: Объект конфигурации.
            vector: Вектор запроса.
            result: Промежуточные данные.
        """
        key = (term, self._normalize_hints(hints), self._compute_config_hash(config))

        # Если ключ уже есть, обновляем
        if key in self._cache:
            self._cache[key] = (tuple(vector.tolist()), result)
            self._order.remove(key)
            self._order.append(key)
            return

        # Удаляем старый элемент если кэш полон
        while len(self._cache) >= self.max_size:
            oldest_key = self._order.pop(0)
            del self._cache[oldest_key]
            logger.debug(f"Кэш вытеснен: {oldest_key}")

        self._cache[key] = (tuple(vector.tolist()), result)
        self._order.append(key)
        logger.info(f"Кэш добавлен: term='{term}', hints={hints}")

    def clear(self) -> None:
        """Очистить кэш."""
        self._cache.clear()
        self._order.clear()
        logger.info("Кэш очищен")
