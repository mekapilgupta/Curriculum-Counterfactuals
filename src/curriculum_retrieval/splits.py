"""
Grouped data splitting and zero-leakage partition management.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from curriculum_retrieval.schemas import QueryRecord, SourceDocumentRecord, SplitManifest


def create_grouped_splits(
    documents: List[SourceDocumentRecord],
    queries: List[QueryRecord],
    train_fraction: float = 0.70,
    dev_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    output_path: Optional[str | Path] = "data/manifests/split_manifest.json",
) -> Tuple[Dict[str, List[QueryRecord]], SplitManifest]:
    """
    Split queries into train, dev, and test sets strictly grouped by document ID / lecture hash
    so no document is shared across splits.
    """
    rng = random.Random(seed)
    
    # Group queries by target_document_id
    doc_to_queries = {}
    for doc in documents:
        doc_to_queries[doc.document_id] = []
    for q in queries:
        if q.target_document_id in doc_to_queries:
            doc_to_queries[q.target_document_id].append(q)

    doc_ids = list(doc_to_queries.keys())
    rng.shuffle(doc_ids)

    n_docs = len(doc_ids)
    n_train = int(n_docs * train_fraction)
    n_dev = int(n_docs * dev_fraction)

    train_doc_ids = set(doc_ids[:n_train])
    dev_doc_ids = set(doc_ids[n_train:n_train + n_dev])
    test_doc_ids = set(doc_ids[n_train + n_dev:])

    split_queries: Dict[str, List[QueryRecord]] = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for doc_id, q_list in doc_to_queries.items():
        if doc_id in train_doc_ids:
            split_queries["train"].extend(q_list)
        elif doc_id in dev_doc_ids:
            split_queries["dev"].extend(q_list)
        else:
            split_queries["test"].extend(q_list)

    manifest = SplitManifest(
        strategy="grouped_by_lecture_hash",
        seed=seed,
        train_ids=[q.query_id for q in split_queries["train"]],
        dev_ids=[q.query_id for q in split_queries["dev"]],
        test_ids=[q.query_id for q in split_queries["test"]],
        grouping_field="target_document_id",
    )

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

    return split_queries, manifest


def verify_split_leakage(
    documents: List[SourceDocumentRecord],
    split_queries: Dict[str, List[QueryRecord]],
) -> Dict[str, Any]:
    """Verify that there is zero overlap of document IDs or lecture hashes across splits."""
    doc_split_map = {}
    leakages = []

    for split_name, q_list in split_queries.items():
        for q in q_list:
            doc_id = q.target_document_id
            if doc_id in doc_split_map and doc_split_map[doc_id] != split_name:
                leakages.append(
                    f"Document {doc_id} present in both '{doc_split_map[doc_id]}' and '{split_name}'"
                )
            else:
                doc_split_map[doc_id] = split_name

    return {
        "is_leakage_free": len(leakages) == 0,
        "leakage_count": len(leakages),
        "leakages": leakages[:10],
        "train_count": len(split_queries.get("train", [])),
        "dev_count": len(split_queries.get("dev", [])),
        "test_count": len(split_queries.get("test", [])),
    }
