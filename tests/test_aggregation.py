"""Тесты модуля агрегации параметров для AI-Terminator."""

import pytest
from src.aggregation import aggregate_parameters, determine_context


class TestAggregateParameters:
    """Тесты функции aggregate_parameters."""

    def test_aggregate_basic(self):
        """Базовый тест агрегации параметров."""
        candidates = [
            {
                "concept_id": "c1",
                "term": "ключ гаечный",
                "domain": "слесарный инструмент",
                "similarity": 0.8,
                "parameters": [
                    {
                        "name": "size_mm",
                        "label_ru": "Размер в мм",
                        "type": "float",
                        "description": "Диаметр зева",
                        "unit": "мм",
                        "enum_values": None,
                        "confidence": 1.0,
                        "source": "knowledge_base",
                    }
                ],
            },
            {
                "concept_id": "c2",
                "term": "ключ разводной",
                "domain": "слесарный инструмент",
                "similarity": 0.6,
                "parameters": [
                    {
                        "name": "size_mm",
                        "label_ru": "Размер в мм",
                        "type": "float",
                        "description": "Диаметр зева",
                        "unit": "мм",
                        "enum_values": None,
                        "confidence": 1.0,
                        "source": "knowledge_base",
                    }
                ],
            },
        ]

        hints_lemmas = [["техника"]]
        result = aggregate_parameters(candidates, hints_lemmas, max_parameters=10)

        assert len(result) == 1
        assert result[0]["name"] == "size_mm"
        assert result[0]["source"] == "knowledge_base"
        assert "confidence" in result[0]

    def test_aggregate_multiple_params(self):
        """Тест с несколькими разными параметрами."""
        candidates = [
            {
                "concept_id": "c1",
                "term": "ключ гаечный",
                "domain": "слесарный инструмент",
                "similarity": 0.9,
                "parameters": [
                    {"name": "size_mm", "label_ru": "Размер", "type": "float", "description": "", "unit": "мм", "enum_values": None},
                    {"name": "material", "label_ru": "Материал", "type": "string", "description": "", "unit": None, "enum_values": None},
                ],
            },
            {
                "concept_id": "c2",
                "term": "ключ разводной",
                "domain": "слесарный инструмент",
                "similarity": 0.7,
                "parameters": [
                    {"name": "size_mm", "label_ru": "Размер", "type": "float", "description": "", "unit": "мм", "enum_values": None},
                    {"name": "material", "label_ru": "Материал", "type": "string", "description": "", "unit": None, "enum_values": None},
                ],
            },
            {
                "concept_id": "c3",
                "term": "молоток",
                "domain": "слесарный инструмент",
                "similarity": 0.5,
                "parameters": [
                    {"name": "weight_g", "label_ru": "Вес", "type": "integer", "description": "", "unit": "г", "enum_values": None},
                ],
            },
        ]

        hints_lemmas = [["техника"]]
        result = aggregate_parameters(candidates, hints_lemmas, max_parameters=10)

        assert len(result) == 3
        names = [p["name"] for p in result]
        assert "size_mm" in names
        assert "material" in names
        assert "weight_g" in names

    def test_aggregate_empty_candidates(self):
        """Тест с пустым списком кандидатов."""
        result = aggregate_parameters([], [["техника"]], max_parameters=10)
        assert result == []

    def test_aggregate_max_limit(self):
        """Тест ограничения количества параметров."""
        candidates = [
            {
                "concept_id": f"c{i}",
                "term": f"термин {i}",
                "domain": "тест",
                "similarity": 0.5,
                "parameters": [
                    {"name": f"param_{i}", "label_ru": f"Параметр {i}", "type": "string", "description": "", "unit": None, "enum_values": None},
                ],
            }
            for i in range(10)
        ]

        hints_lemmas = []
        result = aggregate_parameters(candidates, hints_lemmas, max_parameters=3)

        assert len(result) == 3

    def test_aggregate_with_hint_match(self):
        """Тест с совпадением подсказок."""
        candidates = [
            {
                "concept_id": "c1",
                "term": "ключ гаечный",
                "domain": "слесарный инструмент",
                "similarity": 0.8,
                "parameters": [
                    {
                        "name": "size_mm",
                        "label_ru": "Размер в миллиметрах",
                        "type": "float",
                        "description": "Гаечный ключ размер",
                        "unit": "мм",
                        "enum_values": None,
                        "confidence": 1.0,
                        "source": "knowledge_base",
                    }
                ],
            },
        ]

        # Подсказка "мм" должна совпасть с label_ru и description
        hints_lemmas = [["мм"]]
        result = aggregate_parameters(candidates, hints_lemmas, max_parameters=10)

        assert len(result) == 1
        # Параметр должен иметь высокий confidence из-за совпадения с подсказкой
        assert result[0]["confidence"] > 0.5

    def test_aggregate_no_candidates_with_hint(self):
        """Тест без кандидатов, но с подсказками."""
        candidates = []
        hints_lemmas = [["техника"]]
        result = aggregate_parameters(candidates, hints_lemmas, max_parameters=10)
        assert result == []


class TestDetermineContext:
    """Тесты функции determine_context."""

    def test_determine_context_basic(self):
        """Базовый тест определения контекста."""
        candidates = [
            {
                "concept_id": "c1",
                "term": "ключ гаечный",
                "domain": "слесарный инструмент",
                "similarity": 0.8,
                "parameters": [],
            },
            {
                "concept_id": "c2",
                "term": "ключ разводной",
                "domain": "слесарный инструмент",
                "similarity": 0.6,
                "parameters": [],
            },
        ]

        result = determine_context(candidates)

        assert result["domain"] == "слесарный инструмент"
        assert result["confidence"] > 0

    def test_determine_context_multiple_domains(self):
        """Тест с несколькими доменами."""
        candidates = [
            {
                "concept_id": "c1",
                "term": "ключ гаечный",
                "domain": "слесарный инструмент",
                "similarity": 0.9,
                "parameters": [],
            },
            {
                "concept_id": "c2",
                "term": "скрипичный ключ",
                "domain": "музыка",
                "similarity": 0.7,
                "parameters": [],
            },
        ]

        result = determine_context(candidates)

        assert result["domain"] == "слесарный инструмент"
        assert result["confidence"] == 0.9

    def test_determine_context_empty(self):
        """Тест с пустым списком кандидатов."""
        result = determine_context([])

        assert result["domain"] == "не определено"
        assert result["confidence"] == 0.0

    def test_determine_context_single_candidate(self):
        """Тест с одним кандидатом."""
        candidates = [
            {
                "concept_id": "c1",
                "term": "ключ гаечный",
                "domain": "слесарный инструмент",
                "similarity": 0.85,
                "parameters": [],
            },
        ]

        result = determine_context(candidates)

        assert result["domain"] == "слесарный инструмент"
        assert result["confidence"] == 0.85