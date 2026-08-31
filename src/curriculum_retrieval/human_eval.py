"""
Human evaluation sampling, stratified export (JSONL & CSV), and evaluation analysis.
"""

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from rich.console import Console
from curriculum_retrieval.schemas import HumanEvalRecord

console = Console()


def export_stratified_human_eval_sample(
    eval_records: List[Dict[str, Any]],
    n_samples: int = 200,
    seed: int = 42,
    output_jsonl: str | Path = "data/annotations/human_eval.jsonl",
    output_csv: str | Path = "data/annotations/human_eval.csv",
) -> List[HumanEvalRecord]:
    """
    Produce a stratified sample of retrieval pairs across system, provider, and grade.
    Exports both JSONL and CSV for manual annotator inspection.
    """
    rng = random.Random(seed)
    n_target = min(n_samples, len(eval_records))

    # Stratified grouping
    strata = {}
    for r in eval_records:
        key = (r.get("system_id", "R0"), r.get("translation_provider", "offline"), r.get("target_grade", ""))
        if key not in strata:
            strata[key] = []
        strata[key].append(r)

    selected = []
    keys = list(strata.keys())
    rng.shuffle(keys)

    # Round-robin selection across strata
    while len(selected) < n_target and any(len(strata[k]) > 0 for k in keys):
        for k in keys:
            if strata[k] and len(selected) < n_target:
                idx = rng.randint(0, len(strata[k]) - 1)
                selected.append(strata[k].pop(idx))

    human_records = []
    for i, item in enumerate(selected, start=1):
        rec = HumanEvalRecord(
            sample_id=f"human_sample_{i:04d}",
            query_id=item["query_id"],
            document_id=item["document_id"],
            translation_provider=item.get("translation_provider", ""),
            system_id=item.get("system_id", "R0"),
            rank=item.get("rank", 1),
            question=item.get("question", ""),
            document_text_hi=item.get("document_text_hi", ""),
            target_grade=item.get("target_grade", ""),
            llm_judge_a=item.get("llm_judge_a", {}),
            llm_judge_b=item.get("llm_judge_b", {}),
            human_answer_support=None,
            human_pedagogical_suitability=None,
            human_translation_quality=None,
            human_concept_correctness=None,
            human_pass=None,
            human_notes="",
            annotator_id="",
        )
        human_records.append(rec)

    # Save JSONL
    p_jsonl = Path(output_jsonl)
    p_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(p_jsonl, "w", encoding="utf-8") as f:
        for r in human_records:
            f.write(json.dumps(r.model_dump()) + "\n")

    # Save CSV
    p_csv = Path(output_csv)
    p_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for r in human_records:
        row_d = r.model_dump()
        row_d["llm_judge_a"] = json.dumps(row_d["llm_judge_a"])
        row_d["llm_judge_b"] = json.dumps(row_d["llm_judge_b"])
        csv_rows.append(row_d)

    if csv_rows:
        df = pd.DataFrame(csv_rows)
        df.to_csv(p_csv, index=False, encoding="utf-8")

    console.print(f"[bold green]Exported {len(human_records)} human evaluation samples[/bold green] to {p_jsonl} and {p_csv}")
    return human_records


def evaluate_human_annotations(input_path: str | Path = "data/annotations/human_eval.jsonl") -> Dict[str, Any]:
    """Parse filled human annotation file and calculate summary agreement and pass rates."""
    p = Path(input_path)
    if not p.exists():
        return {"status": "missing_file", "message": f"Human eval file not found at {input_path}"}

    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(HumanEvalRecord(**json.loads(line)))

    annotated = [r for r in records if r.human_answer_support is not None or r.human_pass is not None]

    if not annotated:
        return {
            "status": "empty_annotations",
            "total_samples": len(records),
            "annotated_samples": 0,
            "message": "Human evaluation file contains no completed annotations. Human validation claim withheld.",
        }

    pass_count = sum(1 for r in annotated if r.human_pass is True or (r.human_answer_support or 0) >= 1)
    pass_rate = pass_count / float(len(annotated))

    avg_ans_support = float(pd.Series([r.human_answer_support for r in annotated if r.human_answer_support is not None]).mean())
    avg_trans_qual = float(pd.Series([r.human_translation_quality for r in annotated if r.human_translation_quality is not None]).mean())

    return {
        "status": "completed",
        "total_samples": len(records),
        "annotated_samples": len(annotated),
        "human_pass_rate": round(pass_rate, 4),
        "average_human_answer_support": round(avg_ans_support, 4),
        "average_human_translation_quality": round(avg_trans_qual, 4),
    }
