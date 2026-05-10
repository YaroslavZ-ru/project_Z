"""Модуль словаря синонимов для AI-Terminator.

Предоставляет класс SynonymDict для загрузки и работы со словарем синонимов.
"""

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class SynonymDict:
    """Словарь синонимов, загружаемый из JSON-файла.

    Формат JSON: {"лемма": ["синоним1", "синоним2", ...]}
    Вес каждого синонима вычисляется алгоритмически (см. ТЗ).
    """

    def __init__(self, json_path: str):
        """Инициализация словаря синонимов.

        Args:
            json_path: Путь к JSON-файлу со синонимами.
        """
        self._data: dict[str, list[str]] = {}
        self._load(json_path)

    def _load(self, json_path: str) -> None:
        """Загрузить словарь синонимов из JSON-файла.

        Args:
            json_path: Путь к JSON-файлу.
        """
        path = Path(json_path)

        if not path.exists():
            logger.warning(f"Файл синонимов не найден: {json_path}. Будет использован пустой словарь.")
            self._data = {}
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            logger.info(f"Загружен словарь синонимов из {json_path} ({len(self._data)} лемм)")
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON в {json_path}: {e}")
            self._data = {}

    def get_synonyms(self, lemma: str, max_synonyms: int = 2) -> List[str]:
        """Получить синонимы для леммы.

        Args:
            lemma: Лемма (нормальная форма слова).
            max_synonyms: Максимальное количество синонимов (по умолчанию 2).

        Returns:
            Список синонимов (лемм), не более max_synonyms.
            Если лемма не найдена, возвращает пустой список.
        """
        synonyms = self._data.get(lemma, [])
        return synonyms[:max_synonyms]

    def get_all_synonyms(self, lemmas: List[str], max_synonyms: int = 2) -> List[str]:
        """Получить все синонимы для списка лемм.

        Args:
            lemmas: Список лемм.
            max_synonyms: Максимальное количество синонимов на лемму.

        Returns:
            Список уникальных синонимов.
        """
        all_synonyms = set()
        for lemma in lemmas:
            for syn in self.get_synonyms(lemma, max_synonyms):
                all_synonyms.add(syn)
        return list(all_synonyms)

    def has_synonyms(self, lemma: str) -> bool:
        """Проверить, есть ли синонимы для леммы.

        Args:
            lemma: Лемма.

        Returns:
            True если есть синонимы, False иначе.
        """
        return lemma in self._data and len(self._data[lemma]) > 0

    def get_statistics(self) -> dict:
        """Получить статистику по словарю.

        Returns:
            Словарь с информацией:
            - total Lemmas: общее количество лемм
            - total synonyms: общее количество синонимов
            - max synonyms per lemma: макс. синонимов для одной леммы
        """
        total_synonyms = sum(len(syns) for syns in self._data.values())
        max_synonyms = max((len(syns) for syns in self._data.values()), default=0)

        return {
            "total_lemmas": len(self._data),
            "total_synonyms": total_synonyms,
            "max_synonyms_per_lemma": max_synonyms,
        }
