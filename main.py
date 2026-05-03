#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import spacy
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import uuid
import warnings
import time
from sentence_transformers import SentenceTransformer
import requests
import random

# Подавляем предупреждения spaCy о глоссарии
warnings.filterwarnings("ignore", message=".*Term.*not found in glossary.*")
warnings.filterwarnings("ignore", message=".*Failed to obtain server version.*")

# Пробуем импортировать Qdrant с обработкой ошибок
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️ Qdrant client не установлен. Установите: pip install qdrant-client")

class TermProcessor:
    """
    Класс для обработки терминов в проекте AI-Terminator с интеграцией Qdrant
    """
    def __init__(self, use_qdrant: bool = True, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        # Инициализация spaCy
        try:
            self.nlp = spacy.load("ru_core_news_sm")
            print("✅ Модель spaCy загружена успешно")
        except OSError:
            print("📥 Модель не найдена. Скачиваем...")
            spacy.cli.download("ru_core_news_sm")
            self.nlp = spacy.load("ru_core_news_sm")
            print("✅ Модель загружена")
        
        # Инициализация sentence-transformers для создания эмбеддингов
        print("📥 Загрузка модели для эмбеддингов...")
        try:
            # Используем более легкую модель для начала
            self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ Модель эмбеддингов загружена")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки модели: {e}")
            print("⚠️ Пытаемся загрузить альтернативную модель...")
            try:
                self.embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')
                print("✅ Альтернативная модель загружена")
            except:
                print("❌ Не удалось загрузить модель эмбеддингов")
                self.embedder = None
        
        # Инициализация Qdrant
        self.use_qdrant = use_qdrant and QDRANT_AVAILABLE and self.embedder is not None
        
        if self.use_qdrant:
            self._init_qdrant(qdrant_host, qdrant_port)
        else:
            print("⚠️ Работаем без векторного поиска")
    
    def _init_qdrant(self, host: str, port: int):
        """
        Инициализирует подключение к Qdrant
        """
        try:
            self.qdrant = QdrantClient(host=host, port=port, timeout=5.0)
            # Проверяем подключение
            self.qdrant.get_collections()
            print(f"✅ Подключено к Qdrant на {host}:{port}")
            
            # Создаем коллекцию для терминов, если её нет
            self._init_collection()
        except Exception as e:
            print(f"⚠️ Не удалось подключиться к Qdrant: {e}")
            print("⚠️ Убедитесь что Qdrant запущен:")
            print("   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
            print("⚠️ Работаем без векторного поиска")
            self.use_qdrant = False
    
    def _init_collection(self, collection_name: str = "terms"):
        """
        Инициализирует коллекцию в Qdrant
        """
        try:
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=384,  # Размер вектора для paraphrase-multilingual-MiniLM-L12-v2
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Создана коллекция '{collection_name}' в Qdrant")
            else:
                print(f"✅ Коллекция '{collection_name}' уже существует")
        except Exception as e:
            print(f"⚠️ Ошибка при создании коллекции: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Получает эмбеддинг текста
        """
        if self.embedder is None:
            return [0.0] * 384  # Возвращаем нулевой вектор если модель не загружена
        
        try:
            embedding = self.embedder.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"⚠️ Ошибка получения эмбеддинга: {e}")
            return [0.0] * 384
    
    def _safe_get_morphology(self, token):
        """
        Безопасно получает морфологические признаки токена
        """
        morphology = {}
        
        if not token.morph:
            return morphology
        
        try:
            if hasattr(token.morph, 'to_dict'):
                morphology = token.morph.to_dict()
            elif hasattr(token.morph, 'items'):
                morphology = dict(token.morph)
            else:
                morph_str = str(token.morph)
                if morph_str and morph_str != 'None':
                    for item in morph_str.split('|'):
                        if '=' in item:
                            key, value = item.split('=', 1)
                            morphology[key.strip()] = value.strip()
        except Exception as e:
            print(f"⚠️ Не удалось получить морфологию: {e}")
        
        return morphology
    
    def save_to_qdrant(self, term_data: Dict[str, Any]) -> bool:
        """
        Сохраняет термин и его эмбеддинг в Qdrant
        """
        if not self.use_qdrant:
            return False
        
        try:
            # Создаем текст для эмбеддинга
            text_for_embedding = term_data['original']
            if term_data.get('usages') and len(term_data['usages']) > 0:
                text_for_embedding += " " + term_data['usages'][0]['sentence']
            
            # Получаем эмбеддинг
            embedding = self._get_embedding(text_for_embedding)
            
            # Подготавливаем payload (метаданные)
            payload = {
                "original": term_data['original'],
                "lemma": term_data['lemma'],
                "pos": term_data['pos'],
                "is_stop": term_data['is_stop'],
                "has_context": len(term_data.get('usages', [])) > 0,
                "timestamp": str(time.time())
            }
            
            # Добавляем морфологию если есть
            if term_data.get('morphology'):
                payload["morphology"] = json.dumps(term_data['morphology'], ensure_ascii=False)
            
            # Добавляем связанные термины, если есть
            if term_data.get('related_terms'):
                related_lemmas = [r['lemma'] for r in term_data['related_terms'][:3]]
                payload["related"] = json.dumps(related_lemmas, ensure_ascii=False)
            
            # Создаем точку (универсальный формат)
            point = {
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": payload
            }
            
            # Загружаем в Qdrant
            self.qdrant.upsert(
                collection_name="terms",
                points=[point],
                wait=True
            )
            
            print(f"✅ Термин '{term_data['original']}' сохранен в Qdrant")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при сохранении в Qdrant: {e}")
            return False
    
    def find_similar_terms(self, term: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Находит семантически похожие термины в Qdrant (с автоопределением метода)
        """
        if not self.use_qdrant:
            return []
        
        try:
            # Получаем эмбеддинг для поискового запроса
            query_embedding = self._get_embedding(term)
            
            similar_terms = []
            results = []
            
            # ВАРИАНТ 3: Универсальное решение - пробуем разные методы
            # Пробуем новый метод query_points (современные версии Qdrant)
            if hasattr(self.qdrant, 'query_points'):
                try:
                    search_result = self.qdrant.query_points(
                        collection_name="terms",
                        query=query_embedding,
                        limit=limit
                    )
                    # В новых версиях результат в .points
                    results = search_result.points if hasattr(search_result, 'points') else []
                    print(f"🔍 Использую метод query_points (современный API)")
                except Exception as e:
                    print(f"⚠️ Ошибка при использовании query_points: {e}")
            
            # Пробуем старый метод search (старые версии Qdrant)
            elif hasattr(self.qdrant, 'search'):
                try:
                    search_result = self.qdrant.search(
                        collection_name="terms",
                        query_vector=query_embedding,
                        limit=limit
                    )
                    # В старых версиях результат - это список точек напрямую
                    results = search_result if isinstance(search_result, list) else []
                    print(f"🔍 Использую метод search (старый API)")
                except Exception as e:
                    print(f"⚠️ Ошибка при использовании search: {e}")
            
            # Пробуем альтернативный метод search_points
            elif hasattr(self.qdrant, 'search_points'):
                try:
                    search_result = self.qdrant.search_points(
                        collection_name="terms",
                        vector=query_embedding,
                        limit=limit
                    )
                    results = search_result if isinstance(search_result, list) else []
                    print(f"🔍 Использую метод search_points")
                except Exception as e:
                    print(f"⚠️ Ошибка при использовании search_points: {e}")
            
            else:
                print("❌ Не найден подходящий метод для поиска в Qdrant")
                print("   Доступные методы:", [m for m in dir(self.qdrant) if 'search' in m or 'query' in m])
                return []
            
            # Обрабатываем результаты
            for result in results:
                # Получаем payload (метаданные)
                payload = {}
                if hasattr(result, 'payload'):
                    payload = result.payload
                elif isinstance(result, dict) and 'payload' in result:
                    payload = result['payload']
                
                # Получаем score (оценку сходства)
                score = 0.0
                if hasattr(result, 'score'):
                    score = float(result.score)
                elif isinstance(result, dict) and 'score' in result:
                    score = float(result['score'])
                
                term_info = {
                    "term": payload.get("original", ""),
                    "lemma": payload.get("lemma", ""),
                    "similarity": score,
                    "pos": payload.get("pos", "")
                }
                
                # Добавляем related если есть
                if payload.get("related"):
                    try:
                        term_info["related"] = json.loads(payload["related"])
                    except:
                        pass
                
                # Не включаем сам запрос в результаты (опционально)
                if term_info["term"].lower() != term.lower():
                    similar_terms.append(term_info)
            
            return similar_terms[:limit]
            
        except Exception as e:
            print(f"❌ Ошибка при поиске в Qdrant: {e}")
            return []
    
    def extract_term_features(self, term: str, context: Optional[str] = None, 
                            find_similar: bool = True) -> Dict[str, Any]:
        """
        Извлекает все характеристики термина для базы знаний
        """
        # Анализируем термин
        term_doc = self.nlp(term)
        
        # Проверяем, что термин не пустой
        if not term_doc:
            return {"error": "Пустой термин"}
        
        term_token = term_doc[0]
        
        # Безопасно получаем морфологию
        morphology = self._safe_get_morphology(term_token)
        
        # Базовая структура термина
        term_data = {
            "id": str(uuid.uuid4()),
            "original": term,
            "lemma": term_token.lemma_,
            "pos": term_token.pos_,
            "pos_description": spacy.explain(term_token.pos_) if term_token.pos_ else "",
            "tag": term_token.tag_,
            "morphology": morphology,
            "length": len(term),
            "word_count": len(term_doc),
            "is_alpha": term_token.is_alpha,
            "is_digit": term_token.like_num,
            "is_punct": term_token.is_punct,
            "is_stop": term_token.is_stop,
            "shape": term_token.shape_,
        }
        
        # Если есть контекст, анализируем его
        if context and context.strip():
            context_doc = self.nlp(context)
            
            # Ищем термин в контексте
            usages = []
            related_terms = []
            seen_lemmas = set()
            
            for token in context_doc:
                # Проверяем, является ли токен нашим термином
                if token.lemma_.lower() == term_token.lemma_.lower() or token.text.lower() == term.lower():
                    usage = {
                        "sentence": token.sent.text,
                        "position": token.i,
                        "syntactic_role": token.dep_,
                        "head_word": token.head.text,
                        "head_pos": token.head.pos_,
                        "dependents": [child.text for child in token.children][:3]
                    }
                    usages.append(usage)
                
                # Собираем связанные термины (существительные и прилагательные)
                elif token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and len(token.text) > 2:
                    lemma_lower = token.lemma_.lower()
                    if lemma_lower not in seen_lemmas and lemma_lower != term_token.lemma_.lower():
                        seen_lemmas.add(lemma_lower)
                        related_terms.append({
                            "term": token.text,
                            "lemma": token.lemma_,
                            "pos": token.pos_
                        })
            
            term_data["usages"] = usages
            term_data["related_terms"] = related_terms[:5]
            
            # Добавляем информацию о контексте
            term_data["context_metrics"] = {
                "total_sentences": len(list(context_doc.sents)),
                "total_tokens": len(context_doc),
                "mentions_count": len(usages)
            }
        else:
            term_data["usages"] = []
            term_data["related_terms"] = []
        
        # Сохраняем в Qdrant
        if self.use_qdrant:
            self.save_to_qdrant(term_data)
        
        # Ищем похожие термины
        if find_similar and self.use_qdrant:
            similar = self.find_similar_terms(term)
            if similar:
                term_data["similar_terms"] = similar
        
        return term_data
    
    def batch_process_terms(self, terms: List[Tuple[str, Optional[str]]]) -> List[Dict[str, Any]]:
        """
        Пакетная обработка терминов
        """
        results = []
        for i, (term, context) in enumerate(terms):
            print(f"\n🔄 Обработка {i+1}/{len(terms)}: '{term}'")
            term_data = self.extract_term_features(term, context)
            results.append(term_data)
        return results
    
    def search_by_semantic(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Поиск терминов по семантическому сходству
        """
        return self.find_similar_terms(query, limit)
    
    def print_term_info(self, term_data: Dict[str, Any]):
        """
        Красивый вывод информации о термине
        """
        print("\n" + "="*70)
        print(f"📌 ТЕРМИН: {term_data.get('original', '')}")
        print("="*70)
        
        print(f"\n📝 Основная информация:")
        print(f"  • ID: {term_data.get('id', '')}")
        print(f"  • Лемма: {term_data.get('lemma', '')}")
        print(f"  • Часть речи: {term_data.get('pos', '')} - {term_data.get('pos_description', '')}")
        print(f"  • Тег: {term_data.get('tag', '')}")
        
        if term_data.get('morphology'):
            print(f"\n🔤 Морфологические признаки:")
            for key, value in term_data['morphology'].items():
                print(f"  • {key}: {value}")
        
        print(f"\n🎯 Свойства:")
        print(f"  • Буквенный: {'Да' if term_data.get('is_alpha') else 'Нет'}")
        print(f"  • Число: {'Да' if term_data.get('is_digit') else 'Нет'}")
        print(f"  • Стоп-слово: {'Да' if term_data.get('is_stop') else 'Нет'}")
        
        if term_data.get('usages'):
            print(f"\n📖 Контексты использования ({len(term_data['usages'])}):")
            for i, usage in enumerate(term_data['usages'][:3], 1):  # Показываем первые 3
                print(f"\n  {i}. Предложение: {usage.get('sentence', '')}")
                print(f"     Роль: {usage.get('syntactic_role', '')}")
        
        if term_data.get('related_terms'):
            print(f"\n🔗 Связанные термины (из контекста):")
            for rel in term_data['related_terms'][:5]:
                print(f"  • {rel.get('term', '')} ({rel.get('pos', '')})")
        
        if term_data.get('similar_terms'):
            print(f"\n🎯 Похожие термины (векторный поиск):")
            for sim in term_data['similar_terms']:
                similarity = sim.get('similarity', 0) * 100
                print(f"  • {sim.get('term', '')} ({sim.get('pos', '')}) - сходство: {similarity:.1f}%")
        
        if term_data.get('context_metrics'):
            print(f"\n📊 Метрики контекста:")
            print(f"  • Предложений: {term_data['context_metrics'].get('total_sentences', 0)}")
            print(f"  • Упоминаний термина: {term_data['context_metrics'].get('mentions_count', 0)}")


class TermDictionaryBuilder:
    """
    Класс для построения и пополнения словаря терминов в Qdrant
    """
    def __init__(self, processor):
        self.processor = processor
        self.nlp = processor.nlp
        
        # Базовые термины для старта
        self.base_terms = [
            "искусственный интеллект", "машинное обучение", "нейронная сеть",
            "большие данные", "облачные вычисления", "интернет вещей",
            "блокчейн", "криптовалюта", "кибербезопасность",
            "робототехника", "компьютерное зрение", "обработка естественного языка",
            "рекомендательная система", "генеративный дизайн", "квантовые вычисления",
            "нейросеть", "алгоритм", "программирование", "база данных",
            "информационная безопасность", "автоматизация", "оптимизация",
            "прогнозирование", "кластеризация", "классификация", "регрессия",
            "глубокое обучение", "обучение с подкреплением", "трансформер",
            "компьютер", "сервер", "процессор", "память", "хранилище данных"
        ]
    
    def add_base_terms(self):
        """
        Добавляет базовые термины с контекстом
        """
        print("\n📌 Добавление базовых терминов...")
        count = 0
        for term in self.base_terms:
            context = f"Термин {term} используется в области современных технологий и информатики. {term} является важным понятием в IT сфере."
            self.processor.extract_term_features(term, context, find_similar=False)
            count += 1
            if count % 10 == 0:
                print(f"   Добавлено {count}/{len(self.base_terms)} терминов")
            time.sleep(0.1)
        print(f"✅ Добавлено {len(self.base_terms)} базовых терминов")
    
    def add_from_wikipedia_category(self, category: str, max_terms: int = 20):
        """
        Добавляет термины из категории Википедии
        """
        print(f"\n📌 Импорт из категории '{category}'...")
        
        url = "https://ru.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Категория:{category}",
            "cmlimit": max_terms,
            "format": "json"
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            count = 0
            for member in data.get("query", {}).get("categorymembers", []):
                title = member["title"]
                if "Категория:" not in title and "Список" not in title and ":" not in title:
                    # Получаем краткое описание
                    summary = self._get_wiki_summary(title)
                    if summary:
                        self.processor.extract_term_features(title, summary, find_similar=False)
                        count += 1
                        print(f"   Добавлен: {title}")
                        time.sleep(0.3)
                    
            print(f"✅ Добавлено {count} терминов из категории '{category}'")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    def _get_wiki_summary(self, title: str) -> str:
        """Получает краткое описание статьи"""
        url = "https://ru.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json"
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    return extract[:300]  # Первые 300 символов
        except:
            pass
        return ""
    
    def generate_semantic_variations(self, seed_terms: List[str] = None, variations_per_term: int = 3):
        """
        Генерирует семантические вариации терминов
        """
        if seed_terms is None:
            seed_terms = self.base_terms[:10]  # Берем первые 10 базовых терминов
        
        print(f"\n📌 Генерация вариаций на основе {len(seed_terms)} терминов...")
        
        templates = [
            "современный {}",
            "технология {}",
            "система {}",
            "применение {}",
            "разработка {}",
            "{} в промышленности",
            "{} для бизнеса",
            "метод {}",
            "алгоритм {}",
            "модель {}"
        ]
        
        new_terms = set()
        
        # Генерируем вариации для каждого seed термина
        for term in seed_terms:
            for template in templates:
                variation = template.format(term)
                new_terms.add(variation)
        
        # Добавляем первые 30 вариаций
        count = 0
        for term in list(new_terms)[:30]:
            context = f"Термин {term} связан с областью {random.choice(seed_terms)}. Это важное понятие в современных технологиях."
            self.processor.extract_term_features(term, context, find_similar=False)
            count += 1
            if count % 10 == 0:
                print(f"   Добавлено {count} вариаций")
        
        print(f"✅ Добавлено {count} семантических вариаций")
    
    def build_full_dictionary(self):
        """
        Полное построение словаря
        """
        print("\n" + "="*70)
        print("🚀 ЗАПУСК ПОСТРОЕНИЯ СЛОВАРЯ ТЕРМИНОВ")
        print("="*70)
        
        # 1. Базовые термины
        self.add_base_terms()
        
        # 2. Термины из Википедии (IT категории)
        categories = [
            "Информационные_технологии",
            "Искусственный_интеллект",
            "Программирование",
            "Компьютерные_науки",
            "Алгоритмы",
            "Базы_данных"
        ]
        for category in categories:
            self.add_from_wikipedia_category(category, max_terms=10)
        
        # 3. Семантические вариации
        self.generate_semantic_variations()
        
        # Проверяем результат
        print("\n" + "="*70)
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print("="*70)
        
        test_queries = ["искусственный интеллект", "нейросеть", "компьютер", "программирование", "алгоритм"]
        for query in test_queries:
            results = self.processor.search_by_semantic(query, limit=5)
            print(f"\n🔍 По запросу '{query}':")
            if results:
                for i, r in enumerate(results, 1):
                    print(f"  {i}. {r['term']} ({r['similarity']*100:.1f}%)")
            else:
                print("  Ничего не найдено")
        
        print("\n" + "="*70)
        print("✅ Построение словаря завершено!")
        return True


# Основная программа
if __name__ == "__main__":
    print("🔍 ЗАПУСК AI-TERMINATOR С УНИВЕРСАЛЬНЫМ QDRANT КЛИЕНТОМ")
    print("="*70)
    
    # Создаем процессор
    processor = TermProcessor(use_qdrant=True)
    
    if processor.use_qdrant:
        print(f"\n✅ Qdrant подключен")
    
    # Создаем строитель словаря
    builder = TermDictionaryBuilder(processor)
    
    # Запускаем построение словаря
    builder.build_full_dictionary()
    
    # Дополнительное тестирование поиска
    print("\n" + "="*70)
    print("🔍 ТЕСТИРОВАНИЕ ПОИСКА")
    print("="*70)
    
    test_queries = [
        "транспортное средство",
        "электронное устройство", 
        "программный продукт",
        "метод обучения",
        "система хранения"
    ]
    
    for query in test_queries:
        print(f"\n🔍 По запросу '{query}':")
        results = processor.search_by_semantic(query, limit=3)
        if results:
            for r in results:
                print(f"  • {r['term']} (сходство: {r['similarity']*100:.1f}%)")
        else:
            print("  Ничего не найдено")
    
    print("\n" + "="*70)
    print("✅ Все тесты завершены!")
