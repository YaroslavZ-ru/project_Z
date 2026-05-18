"""Интеграционные тесты полного конвейера (run_pipeline)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.cache import QueryVectorCache
from src.config import Config
from main import run_pipeline


class MockEmbedding:
    """Мок-эмбеддинг для тестов."""

    def __init__(self, dimension: int = 300):
        self._dimension = dimension

    def get_phrase_vector(self, phrase: str) -> np.ndarray:
        """Вернуть фиксированный вектор для теста."""
        vec = np.zeros(self._dimension, dtype=np.float32)
        if "ключ" in phrase.lower():
            vec[0] = 1.0
        elif "техника" in phrase.lower():
            vec[1] = 1.0
        elif "вращение" in phrase.lower():
            vec[2] = 1.0
        return vec

    def get_dimension(self) -> int:
        """Получить размерность вектора."""
        return self._dimension


class MockKnowledgeBase:
    """Мок-база знаний для тестов."""

    def __init__(self, *args, **kwargs):
        self.concepts = [
            {
                "id": 1,
                "term": "гаечный ключ",
                "embedding": np.array([1.0, 0.0, 0.0] + [0.0] * 297, dtype=np.float32),
                "domain": "инструменты",
                "parameters": [
                    {"name": "размер", "value": "10mm", "confidence": 0.9},
                    {"name": "тип", "value": "регулируемый", "confidence": 0.8},
                ],
            }
        ]

    def get_all_concepts(self, *args, **kwargs):
        return self.concepts

    def update_all_embeddings(self):
        pass


class MockSynonymDict:
    """Мок-словарь синонимов для тестов."""

    def get_synonyms(self, lemma: str, max_synonyms: int = 2):
        if lemma == "ключ":
            return [("замок", 0.4), ("отвертка", 0.3)]
        return []


class MockConfig:
    """Мок-конфигурация для тестов."""

    def __init__(self):
        self.db_path = Path("data/knowledge_base.db")
        self.fasttext_model_path = Path("models/cc.ru.300.bin")
        self.synonyms_path = Path("data/synonyms.json")
        self.domain_templates_path = Path("configs/domain_templates.json")
        self.min_confidence = 0.3
        self.max_candidates = 20
        self.max_parameters = 15
        self.use_generative = False
        self.generative_model = "rugpt3small_based_on_gpt2"
        self.timeout_seconds = 2.0
        self.cache_embeddings = True
        self.log_level = "INFO"
        self.cache_lemma_size = 1000
        self.max_synonyms_per_token = 2
        self.use_synonyms = True
        self.max_term_length = 100
        self.max_hint_length = 50


def test_run_pipeline_success():
    """Успешный запуск pipeline с реальными данными."""
    # Создаем временные файлы
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Создаем синонимы (новый формат)
        synonyms_path = tmpdir / "synonyms.json"
        with open(synonyms_path, "w", encoding="utf-8") as f:
            json.dump({"ключ": [{"word": "замок", "weight": 0.4}]}, f)

        # Создаем конфиг
        config = Config.from_json(
            "configs/config.json",
            project_root=Path(__file__).parent.parent,
        )

        # Подменяем пути
        config.synonyms_path = synonyms_path

        # Запускаем pipeline
        result = run_pipeline(
            term="ключи",
            hints=["техника"],
            config=config,
            debug=False,
        )

        assert result["status"] == "ok"
        assert result["term"] == "ключи"
        assert "selected_context" in result
        assert "parameters" in result


def test_run_pipeline_debug():
    """Pipeline с debug=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        synonyms_path = tmpdir / "synonyms.json"
        with open(synonyms_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

        config = Config.from_json(
            "configs/config.json",
            project_root=Path(__file__).parent.parent,
        )
        config.synonyms_path = synonyms_path

        result = run_pipeline(
            term="ключи",
            hints=["техника"],
            config=config,
            debug=True,
        )

        assert result["status"] == "ok"
        assert "debug_info" in result
        assert "query_vector" in result["debug_info"]
        assert "candidates_raw" in result["debug_info"]


def test_run_pipeline_empty_term():
    """Pipeline с пустым термином."""
    config = Config.from_json(
        "configs/config.json",
        project_root=Path(__file__).parent.parent,
    )

    result = run_pipeline(
        term="",
        hints=[],
        config=config,
        debug=False,
    )

    assert result["status"] == "error"
    assert "message" in result


def test_run_pipeline_with_cache():
    """Pipeline с кэшированием."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        synonyms_path = tmpdir / "synonyms.json"
        with open(synonyms_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

        config = Config.from_json(
            "configs/config.json",
            project_root=Path(__file__).parent.parent,
        )
        config.synonyms_path = synonyms_path

        cache = QueryVectorCache(max_size=10)

        # Первый вызов
        result1 = run_pipeline(
            term="ключи",
            hints=["техника"],
            config=config,
            debug=False,
            cache=cache,
        )

        # Второй вызов с теми же параметрами - должен использовать кэш
        result2 = run_pipeline(
            term="ключи",
            hints=["техника"],
            config=config,
            debug=False,
            cache=cache,
        )

        assert result1["status"] == "ok"
        assert result2["status"] == "ok"
        # Векторы должны быть одинаковыми
        assert result1.get("query_vector") == result2.get("query_vector")


def test_run_pipeline_different_hints():
    """Pipeline с разным порядком подсказок (должно использовать кэш)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        synonyms_path = tmpdir / "synonyms.json"
        with open(synonyms_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

        config = Config.from_json(
            "configs/config.json",
            project_root=Path(__file__).parent.parent,
        )
        config.synonyms_path = synonyms_path

        cache = QueryVectorCache(max_size=10)

        # Первый вызов
        result1 = run_pipeline(
            term="ключи",
            hints=["техника", "вращение"],
            config=config,
            debug=False,
            cache=cache,
        )

        # Второй вызов с другим порядком подсказок
        result2 = run_pipeline(
            term="ключи",
            hints=["вращение", "техника"],
            config=config,
            debug=False,
            cache=cache,
        )

        assert result1["status"] == "ok"
        assert result2["status"] == "ok"


def test_run_pipeline_no_synonyms():
    """Pipeline без использования синонимов."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        synonyms_path = tmpdir / "synonyms.json"
        with open(synonyms_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

        config = Config.from_json(
            "configs/config.json",
            project_root=Path(__file__).parent.parent,
        )
        config.synonyms_path = synonyms_path
        config.use_synonyms = False

        result = run_pipeline(
            term="ключи",
            hints=["техника"],
            config=config,
            debug=False,
        )

        assert result["status"] == "ok"
