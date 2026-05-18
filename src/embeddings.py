"""Модуль эмбеддингов для AI-Terminator.

Предоставляет класс FastTextWrapper для работы с fastText моделью
и резервным режимом загрузки статических эмбеддингов.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)


class FastTextWrapper:
    """Обертка для fastText модели с fallback-режимом и кэшированием.

    При инициализации принимает путь к модели и (опционально) путь к резервному
    npy-файлу со статическими эмбеддингами.

    Поддерживает:
    - Ленивую загрузку модели (при первом вызове get_vector)
    - Кэширование векторов слов (словарь _word_cache)
    - get_word_vector() для одного слова
    - get_phrase_vector() для фраз (усреднение векторов слов)
    - Fallback-режим при отсутствии fastText (загрузка из .npy файла)
    """

    def __init__(self, model_path: str, fallback_npy_path: Optional[str] = None, cache_size: int = 20000):
        """Инициализация обертки fastText.

        Args:
            model_path: Путь к fastText модели (.bin файл).
            fallback_npy_path: Путь к резервному словарю {word: vector} в .npy формате.
            cache_size: Максимальный размер кэша векторов слов (LRU, по умолчанию 20000).
        """
        self.model_path = Path(model_path)
        self.fallback_path = Path(fallback_npy_path) if fallback_npy_path else None
        self._model: Optional[object] = None
        self._fallback_vectors: Optional[dict[str, np.ndarray]] = None
        self._dim: int = 300
        self._word_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_maxsize = cache_size
        self._loaded = False
        self._load_attempted = False  # Флаг, что проверка уже была выполнена
        self._logged_oov: set[str] = set()  # Для логирования OOV один раз на слово

    def _load_model(self) -> None:
        """Загрузить fastText модель или переключиться на fallback-режим."""
        # Проверяем наличие модели только один раз
        if self._load_attempted:
            return

        self._load_attempted = True

        if not self.model_path.exists():
            logger.warning(f"FastText модель не найдена: {self.model_path}")
            self._try_fallback()
            # Если fallback тоже не найден - продолжаем с None (случайные векторы)
            return

        try:
            import fasttext

            # Используем memory-mapping для экономии оперативной памяти
            self._model = fasttext.load_model(str(self.model_path), mmap='r')
            # Получаем размерность через тестовое слово
            test_vec = self._model.get_word_vector("a")
            self._dim = len(test_vec)
            logger.info(f"FastText модель загружена (mmap='r'), размерность {self._dim}")
            self._loaded = True
        except Exception as e:
            logger.warning(f"Ошибка загрузки fastText: {e}")
            self._try_fallback()

    def _try_fallback(self) -> None:
        """Попытаться загрузить fallback-словарь со статическими эмбеддингами."""
        if self.fallback_path and self.fallback_path.exists():
            logger.warning(f"Используется fallback-словарь {self.fallback_path}")
            try:
                self._fallback_vectors = np.load(
                    self.fallback_path, allow_pickle=True
                ).item()
                sample_vec = next(iter(self._fallback_vectors.values()))
                self._dim = len(sample_vec)
                logger.info(
                    f"Fallback-словарь загружен, размерность {self._dim}, "
                    f"слов в словаре: {len(self._fallback_vectors)}"
                )
            except Exception as e:
                logger.error(f"Ошибка загрузки fallback-словаря: {e}")
                self._fallback_vectors = None
        else:
            # Это не ошибка, а штатная ситуация при отсутствии модели
            logger.info("Резервный словарь отсутствует, будут использованы случайные векторы")
            self._fallback_vectors = None

    def get_word_vector(self, word: str) -> np.ndarray:
        """Получить вектор для одного слова.

        Args:
            word: Слово для получения вектора.

        Returns:
            Вектор размерности _dim (np.float32).
        """
        word = word.lower().strip()
        if not word:
            return np.zeros(self._dim, dtype=np.float32)

        # Проверка кэша (LRU)
        if word in self._word_cache:
            self._word_cache.move_to_end(word)
            logger.debug(f"Кэш попадание (word): '{word}'")
            return self._word_cache[word]

        logger.debug(f"Кэш промах (word): '{word}'")

        # Если модель не загружена и нет fallback - загружаем
        if not self._loaded and self._fallback_vectors is None:
            self._load_model()

        vec = None

        # Попытка получить из fastText
        if self._model is not None:
            try:
                vec = self._model.get_word_vector(word)
            except Exception as e:
                logger.warning(f"FastText ошибка для '{word}': {e}")

        # Fallback: попытка получить из словаря
        if vec is None and self._fallback_vectors is not None:
            vec = self._fallback_vectors.get(word)

        # Last resort: нулевой вектор (если нет модели и fallback)
        if vec is None:
            # Логируем OOV только один раз на слово
            if word not in self._logged_oov:
                logger.warning(f"Вектор для '{word}' не найден, возвращаем нулевой вектор")
                self._logged_oov.add(word)
            vec = np.zeros(self._dim, dtype=np.float32)

        vec = np.array(vec, dtype=np.float32)
        
        # Добавляем в кэш (LRU)
        if word in self._word_cache:
            self._word_cache.move_to_end(word)
        else:
            self._word_cache[word] = vec
            # Если кэш полон, удаляем самый старый
            if len(self._word_cache) > self._cache_maxsize:
                oldest_key = next(iter(self._word_cache))
                del self._word_cache[oldest_key]

        return vec

    def get_phrase_vector(self, phrase: str) -> np.ndarray:
        """Получить вектор для фразы (усреднение векторов слов).

        Args:
            phrase: Фраза (несколько слов через пробел).

        Returns:
            Вектор размерности _dim (np.float32).
        """
        words = phrase.lower().split()
        if not words:
            return np.zeros(self._dim, dtype=np.float32)

        vecs = [self.get_word_vector(w) for w in words]
        # Усреднение
        mean_vec = np.mean(vecs, axis=0)
        return mean_vec

    def get_dimension(self) -> int:
        """Получить размерность вектора.

        Returns:
            Размерность вектора (по умолчанию 300).
        """
        return self._dim
