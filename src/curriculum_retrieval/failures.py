"""
Automated failure mode detection, categorization, and reporting across 14 failure classes.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from rich.console import Console

console = Console()

FAILURE_CATEGORIES = [
    "1. Raw text succeeds and concepts fail",
    "2. Concepts improve rank",
    "3. Hindi concepts help while English concepts do not",
    "4. English concepts help while Hindi concepts do not",
    "5. Translation provider disagreement",
    "6. Encoder disagreement",
    "7. Correct document missing from concept candidate pool",
    "8. BM25 succeeds while dense retrieval fails",
    "9. Dense succeeds while BM25 fails",
    "10. Grade metadata improves suitability but hurts retrieval",
    "11. Concept generator produces unsupported concepts",
    "12. Hindi translation changes scientific meaning",
    "13. LLM judges disagree",
    "14. Correct document ranked below semantically similar distractor",
]


class FailureAnalyzer:
    """Detects and categorizes failure cases across retrieval ablations and judge evaluations."""

    def __init__(self, output_dir: str | Path = "outputs/failures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.failure_cases: List[Dict[str, Any]] = []

    def analyze_query_runs(
        self,
        query_id: str,
        target_doc_id: str,
        question: str,
        results_by_system: Dict[str, List[str]],  # system_id -> ranked_doc_ids
        concept_candidates: List[str],
        judge_outputs: Optional[Dict[str, Any]] = None,
    ):
        """Analyze multi-system ranking for a single query to identify triggered failure modes."""
        # Rank of target doc in each system
        def get_rank(sys_id: str) -> int:
            ranked = results_by_system.get(sys_id, [])
            return (ranked.index(target_doc_id) + 1) if target_doc_id in ranked else 999

        r_raw = get_rank("R0_raw")
        r_concept = get_rank("R5_concept")
        r_bm25 = get_rank("R1_bm25")
        r_dense = get_rank("R0_dense")
        r_grade = get_rank("R3_grade")
        r_en_c = get_rank("R5_en_concept")
        r_hi_c = get_rank("R5_hi_concept")
        r_ind = get_rank("R0_indictrans2")
        r_openr = get_rank("R0_openrouter")
        r_e5 = get_rank("R0_e5")
        r_bge = get_rank("R0_bge")

        detected = []

        # 1. Raw text succeeds (top-3) and concepts fail (> top-5)
        if r_raw <= 3 and r_concept > 5:
            detected.append("1. Raw text succeeds and concepts fail")

        # 2. Concepts improve rank
        if r_concept < r_raw and r_concept <= 5:
            detected.append("2. Concepts improve rank")

        # 3. Hindi concepts help while English do not
        if r_hi_c < r_en_c and r_hi_c <= 5:
            detected.append("3. Hindi concepts help while English concepts do not")

        # 4. English concepts help while Hindi do not
        if r_en_c < r_hi_c and r_en_c <= 5:
            detected.append("4. English concepts help while Hindi concepts do not")

        # 5. Translation provider disagreement
        if abs(r_ind - r_openr) >= 5 and (r_ind <= 10 or r_openr <= 10):
            detected.append("5. Translation provider disagreement")

        # 6. Encoder disagreement
        if abs(r_e5 - r_bge) >= 5 and (r_e5 <= 10 or r_bge <= 10):
            detected.append("6. Encoder disagreement")

        # 7. Correct doc missing from concept candidate pool
        if concept_candidates and target_doc_id not in concept_candidates:
            detected.append("7. Correct document missing from concept candidate pool")

        # 8. BM25 succeeds while dense fails
        if r_bm25 <= 3 and r_dense > 10:
            detected.append("8. BM25 succeeds while dense retrieval fails")

        # 9. Dense succeeds while BM25 fails
        if r_dense <= 3 and r_bm25 > 10:
            detected.append("9. Dense succeeds while BM25 fails")

        # 10. Grade metadata hurts retrieval
        if r_grade > r_dense and r_dense <= 5:
            detected.append("10. Grade metadata improves suitability but hurts retrieval")

        # 13. LLM judges disagree
        if judge_outputs:
            ja = judge_outputs.get("judge_a", {})
            jb = judge_outputs.get("judge_b", {})
            if ja and jb and ja.get("answer_support") != jb.get("answer_support"):
                detected.append("13. LLM judges disagree")

        # 14. Target ranked below distractor
        if r_dense > 1 and r_dense <= 5:
            detected.append("14. Correct document ranked below semantically similar distractor")

        for mode in detected:
            self.failure_cases.append({
                "query_id": query_id,
                "target_document_id": target_doc_id,
                "question": question,
                "failure_category": mode,
                "ranks": {
                    "raw": r_raw,
                    "concept": r_concept,
                    "bm25": r_bm25,
                    "dense": r_dense,
                    "grade": r_grade,
                },
            })

    def save_reports(self) -> Tuple[Path, Path]:
        """Save failure cases JSONL and failure summary CSV."""
        jsonl_path = self.output_dir / "failure_cases.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in self.failure_cases:
                f.write(json.dumps(item) + "\n")

        # Aggregate counts
        counts = defaultdict(int)
        for item in self.failure_cases:
            counts[item["failure_category"]] += 1

        summary_rows = []
        for cat in FAILURE_CATEGORIES:
            summary_rows.append({
                "failure_category": cat,
                "occurrence_count": counts[cat],
                "percentage_of_failures": (
                    round((counts[cat] / len(self.failure_cases) * 100.0), 2)
                    if self.failure_cases
                    else 0.0
                ),
            })

        df_summary = pd.DataFrame(summary_rows)
        csv_path = self.output_dir / "failure_summary.csv"
        df_summary.to_csv(csv_path, index=False, encoding="utf-8")

        console.print(
            f"[bold green]Saved {len(self.failure_cases)} failure cases[/bold green] to {jsonl_path} and {csv_path}"
        )
        return jsonl_path, csv_path
