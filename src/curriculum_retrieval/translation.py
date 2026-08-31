"""
Translation interfaces and providers for English to Hindi educational text.
Supports IndicTrans2 offline, OpenRouter API, and deterministic mock translation.
"""

import abc
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import httpx
from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_exponential
from curriculum_retrieval.provenance import compute_raw_hash, compute_text_hash
from curriculum_retrieval.schemas import TranslationRecord

load_dotenv()
console = Console()


class BaseTranslationProvider(abc.ABC):
    """Abstract base class for translation providers."""

    @abc.abstractmethod
    def translate_texts(
        self,
        texts: List[str],
        source_lang: str = "en",
        target_lang: str = "hi",
    ) -> List[str]:
        """Translate a batch of texts from source to target language."""
        pass

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Identifier name of the provider."""
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Name or Hugging Face / OpenRouter identifier of the model."""
        pass


class MockTranslationProvider(BaseTranslationProvider):
    """Deterministic mock translator for smoke tests and unit testing."""

    def __init__(self, model_name: str = "mock-indic-v1"):
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    def translate_texts(
        self,
        texts: List[str],
        source_lang: str = "en",
        target_lang: str = "hi",
    ) -> List[str]:
        results = []
        for text in texts:
            # Deterministic pseudo-translation with Hindi marker
            clean = " ".join(text.strip().split())
            results.append(f"[अनुवाद] {clean}")
        return results


class IndicTrans2TranslationProvider(BaseTranslationProvider):
    """Offline IndicTrans2 Hugging Face translation provider."""

    def __init__(
        self,
        model_name: str = "ai4bharat/indictrans2-en-indic-1B",
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
    ):
        self._model_name = model_name
        self.device_str = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self._initialized = False

    @property
    def provider_name(self) -> str:
        return "indictrans2"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _lazy_init(self):
        if self._initialized:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            console.print(f"[bold cyan]Loading IndicTrans2 model {self._model_name}...[/bold cyan]")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=True,
            )
            device = "cuda" if torch.cuda.is_available() and self.device_str == "auto" else "cpu"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self._model_name,
                trust_remote_code=True,
            ).to(device)
            self._initialized = True
        except Exception as e:
            console.print(
                f"[bold red]IndicTrans2 load error:[/bold red] {e}. "
                "Ensure `transformers`, `torch`, and `IndicTransToolkit` (if required) are installed."
            )
            raise

    def translate_texts(
        self,
        texts: List[str],
        source_lang: str = "eng_Latn",
        target_lang: str = "hin_Deva",
    ) -> List[str]:
        self._lazy_init()
        import torch

        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=4,
                )
            decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            results.extend(decoded)
        return results


class OpenRouterTranslationProvider(BaseTranslationProvider):
    """OpenRouter external translation provider."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._model_name = (
            model_name
            or os.getenv("OPENROUTER_TRANSLATION_MODEL", "google/gemini-2.0-flash-001")
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model_name(self) -> str:
        return self._model_name

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_openrouter_single(self, text: str) -> str:
        if not self._api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Set it in .env or pass it to the provider."
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mekapilgupta/Curriculum-Counterfactuals",
            "X-Title": "Curriculum Counterfactuals Pipeline",
        }

        system_prompt = (
            "You are a professional scientific and educational translator. "
            "Translate the following English educational text into clear, accurate Hindi (Devanagari script). "
            "Rules:\n"
            "1. Output ONLY a valid JSON object with a single key 'translated_text'.\n"
            "2. Do not add any explanation, commentary, or answers.\n"
            "3. Do not summarize or omit scientific terms.\n"
            "4. Preserve all mathematical equations, variables, and named entities."
        )

        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self.temperature,
            "max_tokens": max(self.max_tokens, 8192),
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
                return text

            cleaned = raw_content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)

            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "translated_text" in parsed:
                    return str(parsed["translated_text"]).strip()
                return cleaned
            except Exception:
                match = re.search(r'"translated_text"\s*:\s*"([^"]+)"', cleaned)
                if match:
                    return match.group(1).strip()
                return cleaned

    def translate_texts(
        self,
        texts: List[str],
        source_lang: str = "en",
        target_lang: str = "hi",
    ) -> List[str]:
        results = []
        for text in texts:
            translated = self._call_openrouter_single(text)
            results.append(translated)
        return results


import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class TranslationManager:
    """Manages thread-safe translation caching and parallel disk persistence."""

    def __init__(
        self,
        cache_dir: str | Path = "data/translations",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "translation_cache.jsonl"
        self._cache: Dict[str, TranslationRecord] = {}
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record_dict = json.loads(line)
                        key = f"{record_dict['source_text_hash']}_{record_dict['translation_provider']}_{record_dict['translation_model']}"
                        self._cache[key] = TranslationRecord(**record_dict)

    def _save_record(self, record: TranslationRecord):
        with self._lock:
            key = f"{record.source_text_hash}_{record.translation_provider}_{record.translation_model}"
            self._cache[key] = record
            with open(self.cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.model_dump()) + "\n")

    def get_or_translate(
        self,
        document_id: str,
        text: str,
        provider: BaseTranslationProvider,
        prompt_version: str = "v1",
    ) -> TranslationRecord:
        text_hash = compute_text_hash(text)
        key = f"{text_hash}_{provider.provider_name}_{provider.model_name}"

        with self._lock:
            if key in self._cache:
                return self._cache[key]

        translated_texts = provider.translate_texts([text])
        trans_text = translated_texts[0]
        trans_hash = compute_text_hash(trans_text)
        trans_id = f"trans_{trans_hash[:16]}"

        record = TranslationRecord(
            document_id=document_id,
            translation_id=trans_id,
            source_text_hash=text_hash,
            target_language="hi",
            translation_provider=provider.provider_name,  # type: ignore
            translation_model=provider.model_name,
            prompt_version=prompt_version,
            translated_text=trans_text,
            translated_text_hash=trans_hash,
            translation_status="success" if trans_text else "empty",
        )
        self._save_record(record)
        return record

    def translate_documents_parallel(
        self,
        documents: List[Any],
        provider: BaseTranslationProvider,
        max_workers: int = 10,
    ) -> Dict[str, TranslationRecord]:
        """Translate documents concurrently using a thread pool with live progress bar."""
        results: Dict[str, TranslationRecord] = {}
        pending_docs = []

        for doc in documents:
            thash = compute_text_hash(doc.lecture)
            key = f"{thash}_{provider.provider_name}_{provider.model_name}"
            if key in self._cache:
                results[doc.document_id] = self._cache[key]
            else:
                pending_docs.append(doc)

        console.print(f"[bold cyan]Translating {len(pending_docs)} documents ({len(results)} cached) using {max_workers} workers...[/bold cyan]")
        if not pending_docs:
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {
                executor.submit(self.get_or_translate, doc.document_id, doc.lecture, provider): doc
                for doc in pending_docs
            }
            with tqdm(total=len(pending_docs), desc="Translating", unit="doc") as pbar:
                for future in as_completed(future_to_doc):
                    doc = future_to_doc[future]
                    try:
                        rec = future.result()
                        results[doc.document_id] = rec
                    except Exception as e:
                        console.print(f"[bold red]Error translating doc {doc.document_id}:[/bold red] {e}")
                    pbar.update(1)

        return results
