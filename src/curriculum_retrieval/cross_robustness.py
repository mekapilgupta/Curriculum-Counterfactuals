"""
Cross-Model Robustness Matrix: Multi-Encoder & Translation System Invariance.

Evaluates:
- Encoders: `intfloat/multilingual-e5-base` vs `BAAI/bge-m3`
- Translation Invariance: Compares retrieval performance and concept grounding across translation systems.
"""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from rich.console import Console

from curriculum_retrieval.audit import FairResearchAuditor

console = Console()

def run_cross_model_robustness_grid():
    encoders = [
        ("multilingual-e5-base", "intfloat/multilingual-e5-base"),
        ("bge-m3", "BAAI/bge-m3"),
    ]
    
    summary_rows = []
    
    for enc_label, enc_model in encoders:
        console.print(f"[bold cyan]Auditing encoder: {enc_label} ({enc_model})...[/bold cyan]")
        auditor = FairResearchAuditor()
        results = auditor.run_comprehensive_matrix(encoder_name=enc_model)
        
        raw_mrr = results.get("2a_hi_raw_r0", {}).get("mrr", 0.0)
        meta_mrr = results.get("2b_hi_fair_predicted_metadata", {}).get("mrr", 0.0)
        concept_mrr = results.get("2c_hi_pure_bilingual_concepts_r5", {}).get("mrr", 0.0)
        en_ref_mrr = results.get("1b_en_matched_bilingual", {}).get("mrr", 0.0)
        
        cis = results.get("bootstrap_cis", {})
        concept_ci = cis.get("2c_hi_pure_bilingual_concepts_r5", {})
        ci_str = f"[{concept_ci.get('ci_lower', 0.0):+.3f}, {concept_ci.get('ci_upper', 0.0):+.3f}]"
        
        gap_recovered = (concept_mrr - raw_mrr) / max(en_ref_mrr - raw_mrr, 1e-6) * 100.0
        
        summary_rows.append({
            "Encoder_Model": enc_label,
            "HuggingFace_ID": enc_model,
            "Hindi_Raw_R0": round(raw_mrr, 4),
            "Fair_Predicted_Metadata_R4": round(meta_mrr, 4),
            "Pure_Bilingual_Concepts_R5": round(concept_mrr, 4),
            "Relative_Gain_over_Raw": f"{(concept_mrr - raw_mrr) / raw_mrr * 100.0:+.1f}%",
            "Paired_Grouped_95_CI": ci_str,
            "Matched_English_Reference": round(en_ref_mrr, 4),
            "Gap_Recovery_Rate": f"{gap_recovered:.1f}%",
        })
        
    df = pd.DataFrame(summary_rows)
    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "table_cross_encoder_robustness.csv", index=False)
    with open(out_dir / "table_cross_encoder_robustness.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
        
    console.print(f"[bold green]Saved cross-encoder robustness matrix to {out_dir}/table_cross_encoder_robustness.md[/bold green]")
    return df

if __name__ == "__main__":
    run_cross_model_robustness_grid()
