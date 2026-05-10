"""Модуль векторизации запроса для AI-Terminator.

Предоставляет функции для преобразования токенов в векторы эмбеддингов.
"""

import numpy as np
from typing import Optional

# Глобальная переменная для хранения модели
_embedding_model: Optional["EmbeddingModel"] = None


class EmbeddingModel:
    """Базовый класс для модели эмбеддингов."""

    def __init__(self, dimension: int = 300):
        """Инициализация модели.

        Args:
            dimension: Размерность вектора (по умолчанию 300).
        """
        self.dimension = dimension
        self._vectors: dict[str, np.ndarray] = {}

    def get_vector(self, word: str) -> np.ndarray:
        """Получить вектор для слова.

        Args:
            word: Слово для получения вектора.

        Returns:
            Вектор размерности dimension.
        """
        if word not in self._vectors:
            # Если вектора нет, генерируем случайный вектор
            self._vectors[word] = np.random.randn(self.dimension).astype(np.float32)
        return self._vectors[word]

    def get_dimension(self) -> int:
        """Получить размерность вектора."""
        return self.dimension


class FastTextModel(EmbeddingModel):
    """Модель fastText для получения эмбеддингов."""

    def __init__(self, model_path: str, dimension: int = 300):
        """Инициализация модели fastText.

        Args:
            model_path: Путь к .bin файлу модели fastText.
            dimension: Размерность вектора (по умолчанию 300).
        """
        super().__init__(dimension)
        self.model_path = model_path
        self._loaded = False
        self._model = None

    def load(self) -> bool:
        """Загрузить модель fastText.

        Returns:
            True если загрузка успешна, False иначе.
        """
        try:
            import fasttext

            self._model = fasttext.load_model(self.model_path)
            self.dimension = self._model.get_dimension()
            self._loaded = True
            return True
        except (ImportError, FileNotFoundError):
            self._loaded = False
            return False

    def get_vector(self, word: str) -> np.ndarray:
        """Получить вектор для слова через fastText.

        Args:
            word: Слово для получения вектора.

        Returns:
            Вектор размерности dimension.
        """
        if not self._loaded:
            return super().get_vector(word)

        try:
            vector = self._model.get_word_vector(word)
            return vector.astype(np.float32)
        except Exception:
            return super().get_vector(word)


def get_embedding_model(model_path: Optional[str] = None) -> EmbeddingModel:
    """Получить экземпляр модели эмбеддингов.

    Args:
        model_path: Путь к модели fastText (опционально).

    Returns:
        Экземпляр EmbeddingModel.
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    if model_path:
        _embedding_model = FastTextModel(model_path)
        if not _embedding_model.load():
            # Если fastText не загрузилась, используем базовую модель
            _embedding_model = EmbeddingModel()
    else:
        _embedding_model = EmbeddingModel()

    return _embedding_model


def vectorize(
    tokens_with_weights: list[tuple[str, float]],
    model_path: Optional[str] = None,
) -> np.ndarray:
    """Векторизовать токены с весами.

    Args:
        tokens_with_weights: Список кортежей (токен, вес).
        model_path: Путь к модели fastText (опционально).

    Returns:
        Нормализованный вектор запроса (L2-норма = 1).
    """
    model = get_embedding_model(model_path)
    dimension = model.get_dimension()

    # Инициализируем нулевой вектор
    query_vector = np.zeros(dimension, dtype=np.float32)
    total_weight = 0.0

    # Взвешенное суммирование векторов
    for token, weight in tokens_with_weights:
        token_vector = model.get_vector(token)
        query_vector += weight * token_vector
        total_weight += weight

    # Нормализация (если total_weight > 0)
    if total_weight > 0:
        query_vector /= total_weight

    # L2-нормализация
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    return query_vector


def get_model_dimension(model_path: Optional[str] = None) -> int:
    """Получить размерность вектора модели.

    Args:
        model_path: Путь к модели fastText (опционально).

    Returns:
        Размерность вектора.
    """
    model = get_embedding_model(model_path)
    return model.get_dimension()