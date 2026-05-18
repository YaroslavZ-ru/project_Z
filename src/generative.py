"""Модуль генеративного достраивания контекста для AI-Terminator.

Предоставляет класс GenerativeHelper для работы с языковыми моделями
при недостатке параметров из базы знаний.
"""

import logging
import re
import threading
import time
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class GenerativeHelper:
    """Обертка для генеративной модели с ленивой загрузкой и кэшированием.

    При инициализации принимает конфигурацию и загружает модель только при первом вызове.
    Если модель не загружается, система продолжает работать в обычном режиме.
    """

    # Встроенный маппинг русских ключевых слов на английские имена
    DEFAULT_MAPPING: Dict[str, str] = {
        "материал": "material",
        "размер": "size_mm",
        "тип": "type",
        "скорость": "speed",
        "мощность": "power",
        "вес": "weight",
        "длина": "length_mm",
        "цвет": "color",
        "напряжение": "voltage",
        "ёмкость": "capacity",
        "частота": "frequency",
        "температура": "temperature",
        "давление": "pressure",
        "форма": "shape",
        "покрытие": "coating",
        "привод": "drive",
        "источник энергии": "power_source",
    }

    def __init__(
        self,
        use_generative: bool = False,
        model_name: str = "rugpt3small_based_on_gpt2",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        max_new_params: int = 3,
        timeout_seconds: float = 2.0,
        keywords: Optional[List[str]] = None,
        mapping: Optional[Dict[str, str]] = None,
    ):
        """Инициализация генеративного помощника.

        Args:
            use_generative: Включить ли генеративный режим.
            model_name: Имя модели в Hugging Face Hub.
            max_new_tokens: Максимальное количество новых токенов.
            temperature: Температура генерации.
            max_new_params: Максимальное количество новых параметров.
            timeout_seconds: Таймаут на генерацию.
            keywords: Список ключевых слов для поиска параметров.
            mapping: Маппинг русских ключевых слов на английские имена.
        """
        self.use_generative = use_generative
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_new_params = max_new_params
        self.timeout_seconds = timeout_seconds
        self.keywords = keywords or [
            "материал", "размер", "тип", "скорость", "мощность", "вес",
            "длина", "цвет", "напряжение", "ёмкость", "частота", "температура",
            "давление", "форма", "покрытие", "привод", "источник энергии",
        ]
        self.mapping = mapping or self.DEFAULT_MAPPING

        # Состояние модели
        self._model = None
        self._tokenizer = None
        self._available = False
        self._load_attempted = False
        self._load_lock = threading.Lock()  # Для потокобезопасной загрузки

    def is_available(self) -> bool:
        """Проверить, доступна ли генеративная модель.

        Returns:
            True, если модель загружена и готова к работе.
        """
        if not self.use_generative:
            return False
        return self._available

    def _load_model(self) -> bool:
        """Загрузить генеративную модель.

        Returns:
            True при успешной загрузке, False иначе.
        """
        if self._load_attempted:
            return self._available

        with self._load_lock:
            if self._load_attempted:
                return self._available

            self._load_attempted = True

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
            except ImportError as e:
                logger.error(f"Не удалось импортировать transformers или torch: {e}")
                self._available = False
                return False

            try:
                logger.info(f"Загрузка генеративной модели: {self.model_name}")
                start_time = time.time()

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )

                # Переводим в режим оценки и на CPU
                self._model.eval()
                self._model.to("cpu")

                elapsed = time.time() - start_time
                if elapsed > 30:
                    logger.warning(f"Загрузка модели заняла {elapsed:.1f} секунд")

                self._available = True
                logger.info(f"Генеративная модель загружена: {self.model_name}")
                return True

            except Exception as e:
                logger.error(f"Ошибка загрузки генеративной модели: {e}")
                self._available = False
                return False

    def _clean_text(self, text: str) -> str:
        """Очистить текст от потенциально опасных символов.

        Args:
            text: Исходный текст.

        Returns:
            Очищенный текст.
        """
        # Разрешаем только буквы, цифры, пробелы, дефисы и точки
        cleaned = re.sub(r'[^\w\s\-\.]', '', text, flags=re.UNICODE)
        # Ограничиваем длину
        return cleaned[:500]

    def _build_prompt(self, term: str, hints: List[str]) -> str:
        """Сформировать промпт для генерации.

        Args:
            term: Термин.
            hints: Список подсказок.

        Returns:
            Строка промпта.
        """
        term_clean = self._clean_text(term)
        hints_clean = [self._clean_text(h) for h in hints if h]

        if hints_clean:
            hints_str = ", ".join(hints_clean)
            prompt = (
                f"Термин: {term_clean}. Контекст: {hints_str}. "
                f"Перечисли важные параметры, характеристики и ограничения "
                f"для формального описания этого понятия в виде короткого списка, "
                f"каждый пункт с новой строки. Не добавляй лишнего текста."
            )
        else:
            prompt = (
                f"Термин: {term_clean}. Перечисли важные параметры, характеристики "
                f"и ограничения для формального описания этого понятия в виде короткого списка, "
                f"каждый пункт с новой строки. Не добавляй лишнего текста."
            )

        # Ограничиваем длину промпта
        return prompt[:500]

    def _generate_text(self, prompt: str) -> List[str]:
        """Сгенерировать текст с использованием модели.

        Args:
            prompt: Промпт.

        Returns:
            Список сгенерированных текстов (до 3 вариантов).
        """
        if not self._available or not self._model or not self._tokenizer:
            return []

        try:
            # Токенизация
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

            # Генерация
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    num_return_sequences=3,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Декодирование
            generated_texts = []
            for i in range(3):
                text = self._tokenizer.decode(
                    outputs[i][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                if text:
                    generated_texts.append(text)

            logger.debug(f"Сгенерированные тексты: {generated_texts}")
            return generated_texts

        except Exception as e:
            logger.warning(f"Ошибка генерации текста: {e}")
            return []

    def _extract_parameters(self, generated_texts: List[str], existing_params: List[Dict]) -> List[Dict]:
        """Извлечь параметры из сгенерированного текста.

        Args:
            generated_texts: Список сгенерированных текстов.
            existing_params: Список уже существующих параметров.

        Returns:
            Список новых параметров.
        """
        new_params: List[Dict] = []
        existing_names = {p["name"].lower() for p in existing_params}

        for text in generated_texts:
            # Разбиваем на строки
            lines = re.split(r'[\n;]', text)
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Удаляем маркеры списка
                line = re.sub(r'^[-*•]\s*', '', line)
                line = re.sub(r'^\d+\.\s*', '', line)
                line = line.strip()

                # Проверяем наличие ключевого слова
                found_keyword = None
                for keyword in self.keywords:
                    if keyword.lower() in line.lower():
                        found_keyword = keyword
                        break

                if not found_keyword:
                    continue

                # Извлекаем описание (всё после ключевого слова)
                idx = line.lower().find(found_keyword.lower())
                if idx >= 0:
                    # Ищем разделитель после ключевого слова
                    rest = line[idx + len(found_keyword):].strip()
                    # Удаляем двоеточие, тире и т.д.
                    rest = re.sub(r'^[:\-\s]*', '', rest)
                    description = rest.strip() if rest else found_keyword
                else:
                    description = line

                # Определяем английское имя
                if found_keyword in self.mapping:
                    name = self.mapping[found_keyword]
                else:
                    # Транслитерация (упрощенная)
                    name = self._transliterate(found_keyword)
                    if not name:
                        name = f"param_{hash(description) % 10000}"

                # Определяем тип параметра
                param_type = self._determine_type(description)

                # Проверяем дубликаты
                if name.lower() in existing_names:
                    continue

                new_params.append({
                    "name": name,
                    "label_ru": found_keyword,
                    "type": param_type,
                    "description": description,
                    "confidence": 0.5,
                    "source": "generative",
                })

        # Отбираем лучшие параметры (по длине описания)
        new_params.sort(key=lambda x: len(x["description"]), reverse=True)
        return new_params[:self.max_new_params]

    def _transliterate(self, text: str) -> str:
        """Упрощенная транслитерация кириллицы в латиницу.

        Args:
            text: Текст на русском.

        Returns:
            Текст на латинице.
        """
        # Простой маппинг для часто встречающихся слов
        translit_map = {
            "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
            "е": "e", "ё": "yo", "ж": "zh", "з": "z", "и": "i",
            "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
            "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
            "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch",
            "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "",
            "э": "e", "ю": "yu", "я": "ya",
        }
        result = ""
        for char in text.lower():
            result += translit_map.get(char, char)
        return result

    def _determine_type(self, description: str) -> str:
        """Определить тип параметра по описанию.

        Args:
            description: Описание параметра.

        Returns:
            Тип: "string", "integer", "float", "boolean", или "enum".
        """
        desc_lower = description.lower()

        # Проверка на бинарный тип
        binary_indicators = ["да", "нет", "есть", "нет", "включено", "выключено", "истина", "ложь"]
        if any(ind in desc_lower for ind in binary_indicators):
            return "boolean"

        # Проверка на enum (перечисление через / или ,)
        if re.search(r'[\/,]\s*[а-яё]', desc_lower):
            return "enum"

        # Проверка на числовые единицы измерения
        unit_patterns = [
            r'\d+\s*(мм|см|м|км)',  # длина
            r'\d+\s*(г|кг|т)',  # вес
            r'\d+\s*(°C|°F|K)',  # температура
            r'\d+\s*(сек|мин|ч|ч)',  # время
            r'\d+\s*(В|кВ|мВ)',  # напряжение
            r'\d+\s*(А|мА)',  # ток
            r'\d+\s*(Вт|кВт)',  # мощность
            r'\d+\s*(Гц|МГц|кГц)',  # частота
            r'\d+\s*(Н·м|Н/м)',  # давление/момент
        ]
        if any(re.search(pattern, desc_lower) for pattern in unit_patterns):
            return "float"

        return "string"

    def _extract_refinements(self, generated_texts: List[str]) -> List[str]:
        """Извлечь рекомендации из сгенерированного текста.

        Args:
            generated_texts: Список сгенерированных текстов.

        Returns:
            Список рекомендаций (до 3 штук).
        """
        refinement_keywords = [
            "рекоменду", "желательно", "следует", "необходимо",
            "уточните", "обратите внимание", "можно добавить"
        ]

        refinements: List[str] = []
        for text in generated_texts:
            # Разбиваем на предложения
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Проверяем наличие ключевых слов
                for keyword in refinement_keywords:
                    if keyword in sentence.lower():
                        # Очищаем от маркеров списка
                        sentence_clean = re.sub(r'^[-*•]\s*', '', sentence)
                        sentence_clean = re.sub(r'^\d+\.\s*', '', sentence_clean)
                        sentence_clean = sentence_clean.strip()
                        if sentence_clean and len(sentence_clean) > 10:
                            if sentence_clean not in refinements:
                                refinements.append(sentence_clean)
                            break

        return refinements[:3]

    def generate_suggestions(
        self, term: str, hints: List[str], existing_params: List[Dict]
    ) -> Tuple[List[Dict], List[str]]:
        """Сгенерировать параметры и рекомендации.

        Args:
            term: Термин.
            hints: Список подсказок.
            existing_params: Список уже существующих параметров.

        Returns:
            Кортеж (список новых параметров, список рекомендаций).
        """
        if not self.use_generative:
            return [], []

        # Загружаем модель при необходимости
        if not self._available:
            self._load_model()

        if not self._available:
            logger.debug("Генеративная модель недоступна")
            return [], []

        # Формируем промпт
        prompt = self._build_prompt(term, hints)
        logger.debug(f"Промпт: {prompt[:200]}...")

        # Генерируем текст
        generated_texts = self._generate_text(prompt)
        if not generated_texts:
            logger.debug("Генерация не вернула текстов")
            return [], []

        # Извлекаем параметры
        new_params = self._extract_parameters(generated_texts, existing_params)

        # Извлекаем рекомендации
        refinements = self._extract_refinements(generated_texts)

        logger.info(
            f"Генерация: добавлено {len(new_params)} параметров, "
            f"{len(refinements)} рекомендаций"
        )

        return new_params, refinements
