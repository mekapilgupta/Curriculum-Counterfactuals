"""
Dataset inspection, strict leakage filtering, deduplication, and ingestion pipeline.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset
from rich.console import Console
from curriculum_retrieval.provenance import compute_text_hash
from curriculum_retrieval.schemas import DatasetManifest, QueryRecord, SourceDocumentRecord

console = Console()

LEAKAGE_FORBIDDEN_FIELDS = {"solution", "answer", "choices", "hint", "gold_answer", "gold answer"}


def inspect_dataset_schema(dataset_name: str = "derek-thomas/ScienceQA") -> Dict[str, Any]:
    """Inspect dataset schema and return column mapping and summary statistics."""
    console.print(f"[bold blue]Inspecting dataset:[/bold blue] {dataset_name}")
    try:
        ds = load_dataset(dataset_name, split="train", streaming=True)
        sample = next(iter(ds))
        detected_keys = list(sample.keys())
        console.print(f"Detected columns: {detected_keys}")
        return {
            "dataset_name": dataset_name,
            "detected_columns": detected_keys,
            "has_lecture": "lecture" in detected_keys,
            "has_question": "question" in detected_keys,
            "sample_keys": detected_keys,
        }
    except Exception as e:
        console.print(f"[bold red]Failed to stream dataset {dataset_name}:[/bold red] {e}")
        return {"error": str(e)}


def prepare_scienceqa_data(
    dataset_name: str = "derek-thomas/ScienceQA",
    max_documents: Optional[int] = 5000,
    max_queries: Optional[int] = 1000,
    min_lecture_chars: int = 100,
    min_question_chars: int = 10,
    output_dir: str | Path = "data",
) -> Tuple[List[SourceDocumentRecord], List[QueryRecord], DatasetManifest]:
    """Load, filter, deduplicate ScienceQA and produce clean document and query sets."""
    out_path = Path(output_dir)
    processed_dir = out_path / "processed"
    manifests_dir = out_path / "manifests"
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold green]Loading dataset {dataset_name}...[/bold green]")
    raw_splits = ["train", "validation", "test"]
    loaded_splits = {}
    for split in raw_splits:
        try:
            loaded_splits[split] = load_dataset(dataset_name, split=split)
        except Exception:
            # Fallback if split names differ
            pass

    if not loaded_splits:
        # Load default
        ds_all = load_dataset(dataset_name)
        for split_name in ds_all.keys():
            loaded_splits[split_name] = ds_all[split_name]

    total_raw_rows = sum(len(ds) for ds in loaded_splits.values())
    console.print(f"Total raw rows across all splits: {total_raw_rows}")

    lecture_to_doc: Dict[str, SourceDocumentRecord] = {}
    lecture_hash_to_qids = defaultdict(list)
    query_records: List[QueryRecord] = []
    duplicate_map = []
    usable_rows = 0

    row_counter = 0
    for split_name, ds in loaded_splits.items():
        for row in ds:
            row_counter += 1
            question = str(row.get("question", "") or "").strip()
            lecture = str(row.get("lecture", "") or "").strip()
            
            # Filtering criteria
            if not question or not lecture:
                continue
            if len(lecture) < min_lecture_chars:
                continue
            if len(question) < min_question_chars:
                continue

            usable_rows += 1
            row_id = str(row.get("id", f"sqa_{row_counter}"))
            subject = str(row.get("subject", "") or "")
            topic = str(row.get("topic", "") or "")
            category = str(row.get("category", "") or "")
            skill = str(row.get("skill", "") or "")
            grade = str(row.get("grade", "") or "")

            lecture_hash = compute_text_hash(lecture)
            doc_id = f"doc_{lecture_hash[:16]}"
            q_id = f"q_{split_name}_{row_id}"

            lecture_hash_to_qids[lecture_hash].append(q_id)

            if lecture_hash not in lecture_to_doc:
                doc_record = SourceDocumentRecord(
                    document_id=doc_id,
                    source_dataset=dataset_name,
                    source_row_id=row_id,
                    source_split=split_name,
                    source_text_hash=lecture_hash,
                    question_ids=[],
                    subject=subject,
                    topic=topic,
                    category=category,
                    skill=skill,
                    grade=grade,
                    lecture_length_chars=len(lecture),
                    lecture=lecture,
                )
                lecture_to_doc[lecture_hash] = doc_record
            else:
                duplicate_map.append({
                    "original_doc_id": lecture_to_doc[lecture_hash].document_id,
                    "duplicate_row_id": row_id,
                    "split": split_name,
                    "lecture_hash": lecture_hash,
                })

            q_record = QueryRecord(
                query_id=q_id,
                question_text=question,
                source_row_id=row_id,
                source_split=split_name,
                target_document_id=doc_id,
                grade=grade,
                subject=subject,
                topic=topic,
                category=category,
                skill=skill,
            )
            query_records.append(q_record)

    # Attach all mapped question IDs to each document record
    for lecture_hash, doc in lecture_to_doc.items():
        doc.question_ids = lecture_hash_to_qids[lecture_hash]

    documents = list(lecture_to_doc.values())

    # Limit to max_documents and max_queries if specified
    if max_documents and len(documents) > max_documents:
        documents = documents[:max_documents]
        valid_doc_ids = {d.document_id for d in documents}
        query_records = [q for q in query_records if q.target_document_id in valid_doc_ids]

    if max_queries and len(query_records) > max_queries:
        query_records = query_records[:max_queries]

    console.print(f"Filtered to [bold]{len(documents)}[/bold] unique documents and [bold]{len(query_records)}[/bold] queries.")

    # Save to disk
    docs_file = processed_dir / "scienceqa_documents.jsonl"
    with open(docs_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc.model_dump(), ensure_ascii=False) + "\n")

    queries_file = processed_dir / "scienceqa_queries.jsonl"
    with open(queries_file, "w", encoding="utf-8") as f:
        for q in query_records:
            f.write(json.dumps(q.model_dump(), ensure_ascii=False) + "\n")

    dup_file = processed_dir / "duplicate_map.jsonl"
    with open(dup_file, "w", encoding="utf-8") as f:
        for item in duplicate_map:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = DatasetManifest(
        total_raw_rows=total_raw_rows,
        usable_rows=usable_rows,
        unique_documents=len(documents),
        unique_queries=len(query_records),
        min_lecture_chars=min_lecture_chars,
        min_question_chars=min_question_chars,
        source_dataset=dataset_name,
        schema_detected={
            "documents_file": str(docs_file),
            "queries_file": str(queries_file),
            "duplicate_map_file": str(dup_file),
        },
    )

    manifest_file = manifests_dir / "dataset_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)

    console.print(f"[bold green]Data preparation complete![/bold green] Saved to {processed_dir}")
    return documents, query_records, manifest


def validate_data_leakage(
    documents: List[SourceDocumentRecord],
    queries: List[QueryRecord],
) -> Dict[str, Any]:
    """Audit data objects to guarantee no forbidden fields or leakage exist."""
    errors = []
    
    for doc in documents:
        # Check that forbidden fields are not in document serialization
        doc_dict = doc.model_dump()
        for forbidden in LEAKAGE_FORBIDDEN_FIELDS:
            if forbidden in doc_dict:
                errors.append(f"Leakage in doc {doc.document_id}: contains forbidden key {forbidden}")
        
        # Verify length
        if len(doc.lecture) < 10:
            errors.append(f"Invalid lecture length in doc {doc.document_id}")

    for q in queries:
        q_dict = q.model_dump()
        for forbidden in LEAKAGE_FORBIDDEN_FIELDS:
            if forbidden in q_dict:
                errors.append(f"Leakage in query {q.query_id}: contains forbidden key {forbidden}")
        if not q.target_document_id:
            errors.append(f"Query {q.query_id} has no target_document_id")

    return {
        "valid": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors[:20],
        "total_documents_checked": len(documents),
        "total_queries_checked": len(queries),
    }
