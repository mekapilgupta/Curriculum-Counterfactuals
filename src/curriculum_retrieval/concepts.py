"""
Bilingual concept extraction, document representation variants (V0-V4),
and query representation variants (Q0-Q4).
"""

import abc
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import httpx
from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_exponential
from curriculum_retrieval.provenance import compute_text_hash
from curriculum_retrieval.schemas import (
    BilingualConcept,
    ConceptRecord,
    QueryConceptRecord,
    QueryRecord,
    SourceDocumentRecord,
    TranslationRecord,
)

load_dotenv()
console = Console()


class BaseConceptGenerator(abc.ABC):
    """Abstract base class for bilingual concept extraction."""

    @abc.abstractmethod
    def extract_concepts(self, text: str, is_query: bool = False) -> List[BilingualConcept]:
        """Extract 3-8 bilingual scientific concepts from the provided text."""
        pass

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        pass


class HeuristicConceptGenerator(BaseConceptGenerator):
    """
    Deterministic rule-based concept extractor based on NLP heuristics and domain metadata.
    Marked explicitly as generator_provider: heuristic.
    """

    def __init__(self, model_name: str = "heuristic-v1"):
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return "heuristic"

    @property
    def model_name(self) -> str:
        return self._model_name

    def extract_concepts(self, text: str, is_query: bool = False) -> List[BilingualConcept]:
        words = re.findall(r"\b[A-Za-z]{4,}\b", text)
        stop_words = {
            "this", "that", "these", "those", "what", "which", "where", "when",
            "there", "their", "about", "could", "should", "would", "other",
            "because", "between", "through", "during", "before", "after",
            "above", "below", "following", "correct", "statement", "answer",
            "question", "example", "figure", "table", "shown", "below"
        }
        candidate_words = [w.capitalize() for w in words if w.lower() not in stop_words]
        
        # Deduplicate preserving order
        unique_words = list(dict.fromkeys(candidate_words))[:6]
        if not unique_words:
            unique_words = ["Science Concept", "Educational Principle"]

        concepts = []
        for i, word in enumerate(unique_words):
            cid = f"c_heur_{compute_text_hash(word)[:8]}"
            concepts.append(
                BilingualConcept(
                    concept_id=cid,
                    label_en=word,
                    label_hi=f"{word} (संकल्पना)",
                    aliases_en=[word.lower()],
                    aliases_hi=[f"{word.lower()} अवधारणा"],
                    evidence_span_en=word,
                    evidence_span_hi=f"{word} (संकल्पना)",
                    confidence=1.0,
                )
            )
        return concepts


class OpenRouterConceptGenerator(BaseConceptGenerator):
    """Concept generator powered by OpenRouter LLMs."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._model_name = (
            model_name
            or os.getenv("OPENROUTER_CONCEPT_MODEL", "google/gemini-2.0-flash-001")
        )
        self.temperature = temperature

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model_name(self) -> str:
        return self._model_name

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def extract_concepts(self, text: str, is_query: bool = False) -> List[BilingualConcept]:
        if not self._api_key:
            # Fallback to heuristic generator if API key is missing
            fallback = HeuristicConceptGenerator()
            return fallback.extract_concepts(text, is_query=is_query)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mekapilgupta/Curriculum-Counterfactuals",
            "X-Title": "Curriculum Counterfactuals Pipeline",
        }

        system_prompt = (
            "You are an expert bilingual curriculum annotator for K-12 science. "
            "Extract 3 to 8 scientific entities, processes, relations, principles, or quantities "
            "explicitly supported by the given text.\n"
            "Output JSON with format:\n"
            "{\n"
            '  "concepts": [\n'
            '    {\n'
            '      "label_en": "English Concept Name",\n'
            '      "label_hi": "Hindi Concept Name (Devanagari)",\n'
            '      "aliases_en": ["synonym1", "synonym2"],\n'
            '      "aliases_hi": ["पर्याय1", "पर्याय2"],\n'
            '      "evidence_span_en": "exact span from input",\n'
            '      "evidence_span_hi": "translated span"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self.temperature,
            "max_tokens": 8192,
        }

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            raw_content = message.get("content") or ""
            if not raw_content and message.get("reasoning"):
                raw_content = message.get("reasoning")

            if not raw_content:
                return HeuristicConceptGenerator().extract_concepts(text, is_query=is_query)

            cleaned = raw_content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)

            try:
                parsed = json.loads(cleaned)
            except Exception:
                # Fallback to heuristic if json parsing fails
                return HeuristicConceptGenerator().extract_concepts(text, is_query=is_query)
            
            raw_concepts = parsed.get("concepts", []) if isinstance(parsed, dict) else []
            if not raw_concepts:
                return HeuristicConceptGenerator().extract_concepts(text, is_query=is_query)

            results = []
            for item in raw_concepts:
                label_en = str(item.get("label_en", "")).strip()
                label_hi = str(item.get("label_hi", "")).strip()
                if not label_en:
                    continue
                cid = f"c_{compute_text_hash(label_en)[:8]}"
                results.append(
                    BilingualConcept(
                        concept_id=cid,
                        label_en=label_en,
                        label_hi=label_hi or label_en,
                        aliases_en=item.get("aliases_en", []),
                        aliases_hi=item.get("aliases_hi", []),
                        evidence_span_en=item.get("evidence_span_en", ""),
                        evidence_span_hi=item.get("evidence_span_hi", ""),
                        confidence=1.0,
                    )
                )
            return results


class ConceptManager:
    """Manages document and query concept generation and disk caching."""

    def __init__(self, cache_dir: str | Path = "data/concepts"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.doc_cache_file = self.cache_dir / "document_concepts.jsonl"
        self.query_cache_file = self.cache_dir / "query_concepts.jsonl"
        self._doc_cache: Dict[str, ConceptRecord] = {}
        self._query_cache: Dict[str, QueryConceptRecord] = {}
        self._load_cache()

    def _load_cache(self):
        if self.doc_cache_file.exists():
            with open(self.doc_cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        self._doc_cache[d["document_id"]] = ConceptRecord(**d)

        if self.query_cache_file.exists():
            with open(self.query_cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        self._query_cache[d["query_id"]] = QueryConceptRecord(**d)

    def get_or_generate_doc_concepts(
        self,
        doc: SourceDocumentRecord,
        generator: BaseConceptGenerator,
    ) -> ConceptRecord:
        if doc.document_id in self._doc_cache:
            return self._doc_cache[doc.document_id]

        concepts = generator.extract_concepts(doc.lecture, is_query=False)
        record = ConceptRecord(
            document_id=doc.document_id,
            source_text_hash=doc.source_text_hash,
            concept_schema_version="v1",
            generator_provider=generator.provider_name,  # type: ignore
            generator_model=generator.model_name,
            concepts=concepts,
        )
        self._doc_cache[doc.document_id] = record
        with open(self.doc_cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump()) + "\n")
        return record

    def get_or_generate_query_concepts(
        self,
        query: QueryRecord,
        generator: BaseConceptGenerator,
    ) -> QueryConceptRecord:
        if query.query_id in self._query_cache:
            return self._query_cache[query.query_id]

        concepts = generator.extract_concepts(query.question_text, is_query=True)
        record = QueryConceptRecord(
            query_id=query.query_id,
            source_text_hash=compute_text_hash(query.question_text),
            concept_schema_version="v1",
            generator_provider=generator.provider_name,  # type: ignore
            generator_model=generator.model_name,
            concepts=concepts,
        )
        self._query_cache[query.query_id] = record
        with open(self.query_cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump()) + "\n")
        return record


# Document representation builders (V0 - V4)
def build_document_variant(
    variant: str,
    translation: TranslationRecord,
    concept_record: Optional[ConceptRecord],
    source_doc: SourceDocumentRecord,
) -> str:
    """
    Build document representation:
    - V0: Hindi translated lecture only
    - V1: Hindi lecture + English concepts
    - V2: Hindi lecture + Hindi concepts
    - V3: Hindi lecture + bilingual English-Hindi concepts
    - V4: Hindi lecture + bilingual concepts + ScienceQA metadata
    """
    hindi_text = translation.translated_text
    if variant == "V0":
        return hindi_text

    concepts = concept_record.concepts if concept_record else []
    en_concepts = " | ".join([c.label_en for c in concepts if c.label_en])
    hi_concepts = " | ".join([c.label_hi for c in concepts if c.label_hi])
    bilingual_concepts = " | ".join(
        [f"{c.label_en} ({c.label_hi})" for c in concepts if c.label_en]
    )

    if variant == "V1":
        return f"{hindi_text}\n\n[Concepts EN]: {en_concepts}" if en_concepts else hindi_text
    elif variant == "V2":
        return f"{hindi_text}\n\n[संकल्पनाएँ]: {hi_concepts}" if hi_concepts else hindi_text
    elif variant == "V3":
        return f"{hindi_text}\n\n[Bilingual Concepts]: {bilingual_concepts}" if bilingual_concepts else hindi_text
    elif variant == "V4":
        meta_parts = []
        if source_doc.subject:
            meta_parts.append(f"Subject: {source_doc.subject}")
        if source_doc.topic:
            meta_parts.append(f"Topic: {source_doc.topic}")
        if source_doc.category:
            meta_parts.append(f"Category: {source_doc.category}")
        if source_doc.skill:
            meta_parts.append(f"Skill: {source_doc.skill}")
        if source_doc.grade:
            meta_parts.append(f"Grade: {source_doc.grade}")
        meta_str = " | ".join(meta_parts)
        return (
            f"{hindi_text}\n\n[Bilingual Concepts]: {bilingual_concepts}\n\n[Curriculum Metadata]: {meta_str}"
        )
    else:
        raise ValueError(f"Unknown document variant: {variant}")


# Query representation builders (Q0 - Q4)
def build_query_variant(
    variant: str,
    query: QueryRecord,
    query_translation: Optional[TranslationRecord],
    query_concept_record: Optional[QueryConceptRecord],
) -> str:
    """
    Build query representation:
    - Q0: original English question
    - Q1: English question + English query concepts
    - Q2: Hindi-translated question
    - Q3: English question + Hindi query concepts
    - Q4: English question + bilingual query concepts
    """
    q_en = query.question_text
    if variant == "Q0":
        return q_en

    concepts = query_concept_record.concepts if query_concept_record else []
    en_concepts = " | ".join([c.label_en for c in concepts if c.label_en])
    hi_concepts = " | ".join([c.label_hi for c in concepts if c.label_hi])
    bilingual_concepts = " | ".join(
        [f"{c.label_en} ({c.label_hi})" for c in concepts if c.label_en]
    )

    if variant == "Q1":
        return f"{q_en} [Concepts]: {en_concepts}" if en_concepts else q_en
    elif variant == "Q2":
        if query_translation and query_translation.translated_text:
            return query_translation.translated_text
        return q_en
    elif variant == "Q3":
        return f"{q_en} [संकल्पनाएँ]: {hi_concepts}" if hi_concepts else q_en
    elif variant == "Q4":
        return f"{q_en} [Bilingual Concepts]: {bilingual_concepts}" if bilingual_concepts else q_en
    else:
        raise ValueError(f"Unknown query variant: {variant}")
