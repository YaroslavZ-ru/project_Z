"""Модуль лемматизатора для AI-Terminator.

Предоставляет класс Lemmatizer как синглтон с кэшированием результатов.
Поддерживает обработку составных терминов.
"""

import logging
from typing import List

from pymorphy3 import MorphAnalyzer

logger = logging.getLogger(__name__)


class Lemmatizer:
    """Лемматизатор с кэшированием результатов (синглтон).

    Инициализирует MorphAnalyzer один раз и кэширует результаты лемматизации.
    Поддерживает обработку отдельных слов и фраз.
    """

    _instance: "Lemmatizer" = None
    _morph: MorphAnalyzer = None
    _cache: dict[str, str] = {}

    def __new__(cls) -> "Lemmatizer":
        """Создать или вернуть существующий экземпляр (синглтон)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._morph = MorphAnalyzer()
            cls._instance._cache = {}
        return cls._instance

    def lemmatize_word(self, word: str) -> str:
        """Лемматизировать одно слово.

        Args:
            word: Исходное слово.

        Returns:
            Лемма (нормальная форма слова). Если слово не удалось распарсить,
            возвращает его в нижнем регистре. Для пустой строки возвращает пустую строку.
        """
        if not word:
            return ""

        # Проверка кэша
        if word in self._cache:
            return self._cache[word]

        try:
            # Парсим слово
            parsed = self._morph.parse(word)

            if parsed:
                # Выбираем вариант с максимальным score
                best = max(parsed, key=lambda p: p.score)
                res = best.normal_form
            else:
                # Если не удалось распарсить, возвращаем в нижнем регистре
                res = word.lower()

        except Exception as e:
            logger.warning(f"Ошибка лемматизации слова '{word}': {e}")
            res = word.lower()

        # Кэшируем результат
        self._cache[word] = res
        return res

    def lemmatize_phrase(self, phrase: str) -> List[str]:
        """Лемматизировать фразу (разбить на слова и обработать каждое).

        Args:
            phrase: Исходная фраза.

        Returns:
            Список лемм. Если фраза пуста, возвращает пустой список.
            Пустые результаты отфильтровываются.
        """
        if not phrase:
            return []

        words = phrase.split()
        result = []

        for w in words:
            if w:  # Пропускаем пустые строки
                lemma = self.lemmatize_word(w)
                if lemma:  # Пропускаем пустые леммы
                    result.append(lemma)

        return result
