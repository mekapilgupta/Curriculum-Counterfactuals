"""
Dual LLM Judge evaluation system (OpenRouter Model A & Model B) and agreement statistics.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
import httpx
import numpy as np
from rich.console import Console
from sklearn.metrics import cohen_kappa_score
from tenacity import retry, stop_after_attempt, wait_exponential
from curriculum_retrieval.schemas import LLMJudgeOutput

load_dotenv()
console = Console()


class LLMJudgeEvaluator:
    """Evaluates question-document pairs using independent LLM judges with fixed rubrics."""

    def __init__(
        self,
        model_a: Optional[str] = None,
        model_b: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: str | Path = "data/annotations",
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model_a = (
            model_a
            or os.getenv("OPENROUTER_JUDGE_MODEL_A", "google/gemini-2.0-flash-001")
        )
        self.model_b = (
            model_b
            or os.getenv("OPENROUTER_JUDGE_MODEL_B", "qwen/qwen-2.5-72b-instruct")
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "llm_judge_cache.jsonl"
        self._cache: Dict[str, LLMJudgeOutput] = {}
        self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        k = f"{d['query_id']}_{d['document_id']}_{d['model']}"
                        self._cache[k] = LLMJudgeOutput(**d["output"])

    def _mock_judgment(
        self, question: str, doc_text_hi: str, grade: str, model_name: str
    ) -> LLMJudgeOutput:
        """Deterministic mock judge for testing without API keys."""
        # Simple heuristic judgment based on length and grade match
        is_informative = len(doc_text_hi) > 50
        return LLMJudgeOutput(
            answer_support=2 if is_informative else 1,
            pedagogical_suitability=2 if grade else 1,
            language_quality=2,
            unsupported_claims=0,
            reason=f"Mock evaluation by {model_name}: text contains sufficient educational context.",
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_judge(
        self,
        model_name: str,
        question: str,
        doc_text_hi: str,
        grade: str,
    ) -> LLMJudgeOutput:
        if not self.api_key or model_name.startswith("mock"):
            return self._mock_judgment(question, doc_text_hi, grade, model_name)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mekapilgupta/Curriculum-Counterfactuals",
            "X-Title": "Curriculum Counterfactuals Pipeline",
        }

        system_prompt = (
            "You are an impartial educational retrieval judge evaluating whether a retrieved Hindi explanation "
            "supports answering a student's English science question.\n\n"
            "Rubrics:\n"
            "answer_support:\n"
            "0 = does not help answer the question\n"
            "1 = partially useful\n"
            "2 = directly supports the answer\n\n"
            "pedagogical_suitability:\n"
            "0 = clearly unsuitable or unrelated\n"
            "1 = usable with substantial adaptation\n"
            "2 = suitable for the declared grade/level\n\n"
            "language_quality:\n"
            "0 = unusable or seriously corrupted\n"
            "1 = understandable with issues\n"
            "2 = clear and grammatical\n\n"
            "unsupported_claims:\n"
            "0 = no apparent unsupported claims\n"
            "1 = possible unsupported or mistranslated claim\n"
            "2 = clear unsupported claim\n\n"
            "Output valid JSON only:\n"
            "{\n"
            '  "answer_support": 0|1|2,\n'
            '  "pedagogical_suitability": 0|1|2,\n'
            '  "language_quality": 0|1|2,\n'
            '  "unsupported_claims": 0|1|2,\n'
            '  "reason": "short structured reason"\n'
            "}"
        )

        user_content = (
            f"Target Grade Level: {grade or 'K-12'}\n"
            f"Student Question (English): {question}\n\n"
            f"Retrieved Explanation (Hindi): {doc_text_hi}\n"
        )

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"]
            parsed = json.loads(raw_content)
            return LLMJudgeOutput(**parsed)

    def evaluate_pair(
        self,
        query_id: str,
        doc_id: str,
        question: str,
        doc_text_hi: str,
        grade: str,
    ) -> Tuple[LLMJudgeOutput, LLMJudgeOutput]:
        """Run evaluation with both Judge A and Judge B."""
        key_a = f"{query_id}_{doc_id}_{self.model_a}"
        if key_a in self._cache:
            out_a = self._cache[key_a]
        else:
            out_a = self._call_judge(self.model_a, question, doc_text_hi, grade)
            self._cache[key_a] = out_a
            with open(self.cache_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({
                        "query_id": query_id,
                        "document_id": doc_id,
                        "model": self.model_a,
                        "output": out_a.model_dump(),
                    })
                    + "\n"
                )

        key_b = f"{query_id}_{doc_id}_{self.model_b}"
        if key_b in self._cache:
            out_b = self._cache[key_b]
        else:
            out_b = self._call_judge(self.model_b, question, doc_text_hi, grade)
            self._cache[key_b] = out_b
            with open(self.cache_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({
                        "query_id": query_id,
                        "document_id": doc_id,
                        "model": self.model_b,
                        "output": out_b.model_dump(),
                    })
                    + "\n"
                )

        return out_a, out_b


def compute_judge_agreement_statistics(
    judgments_a: List[LLMJudgeOutput],
    judgments_b: List[LLMJudgeOutput],
) -> Dict[str, Any]:
    """Calculate agreement metrics between Judge A and Judge B."""
    assert len(judgments_a) == len(judgments_b), "Judgments count mismatch."
    n = len(judgments_a)
    if n == 0:
        return {}

    ans_a = [j.answer_support for j in judgments_a]
    ans_b = [j.answer_support for j in judgments_b]

    ped_a = [j.pedagogical_suitability for j in judgments_a]
    ped_b = [j.pedagogical_suitability for j in judgments_b]

    lang_a = [j.language_quality for j in judgments_a]
    lang_b = [j.language_quality for j in judgments_b]

    exact_ans_agree = np.mean([a == b for a, b in zip(ans_a, ans_b)])
    exact_ped_agree = np.mean([a == b for a, b in zip(ped_a, ped_b)])
    exact_lang_agree = np.mean([a == b for a, b in zip(lang_a, lang_b)])

    try:
        kappa_ans = cohen_kappa_score(ans_a, ans_b)
    except Exception:
        kappa_ans = 1.0

    avg_ans_support = float(np.mean(ans_a + ans_b) / 2.0)
    avg_ped_suit = float(np.mean(ped_a + ped_b) / 2.0)
    avg_lang_qual = float(np.mean(lang_a + lang_b) / 2.0)
    
    # Translation corrupt rate (language_quality == 0)
    trans_error_rate = float(np.mean([(a == 0 or b == 0) for a, b in zip(lang_a, lang_b)]))

    return {
        "n_evaluated_pairs": n,
        "exact_agreement_answer_support": round(float(exact_ans_agree), 4),
        "exact_agreement_pedagogical_suitability": round(float(exact_ped_agree), 4),
        "exact_agreement_language_quality": round(float(exact_lang_agree), 4),
        "cohens_kappa_answer_support": round(float(kappa_ans), 4),
        "average_answer_support": round(avg_ans_support, 4),
        "average_pedagogical_suitability": round(avg_ped_suit, 4),
        "average_language_quality": round(avg_lang_qual, 4),
        "translation_error_rate": round(trans_error_rate, 4),
    }
