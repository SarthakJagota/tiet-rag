"""
Compare Vanilla RAG vs Agentic RAG across all metrics.
Run after both run_all.py and run_all_agentic.py have completed.

Usage:
    python evaluation/compare_agentic.py
"""
import os, sys
import pandas as pd

BASE = os.path.dirname(__file__)

def pct(filename, col):
    path = os.path.join(BASE, filename)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if col not in df.columns:
            return None
        return df[col].mean() * 100
    except:
        return None

METRICS = [
    ("Correctness",         "correctness_scores.csv",          "correctness_agentic.csv",          "correctness"),
    ("Relevance",           "relevance_scores.csv",             "relevance_agentic.csv",             "relevance"),
    ("Groundedness",        "groundedness_scores.csv",          "groundedness_agentic.csv",          "groundedness"),
    ("Retrieval Relevance", "retrieval_relevance_scores.csv",   "retrieval_relevance_agentic.csv",   "retrieval_relevance"),
    ("Specificity",         "generation_quality_scores.csv",    "generation_quality_agentic.csv",    "specificity"),
    ("Conciseness",         "generation_quality_scores.csv",    "generation_quality_agentic.csv",    "conciseness"),
    ("Completeness",        "generation_quality_scores.csv",    "generation_quality_agentic.csv",    "completeness"),
]

RECALL = [
    ("Recall@5",  "hit@5"),
    ("Recall@7",  "hit@7"),
    ("Recall@10", "hit@10"),
    ("Recall@20", "hit@20"),
]

def recall_pct(filename, col):
    path = os.path.join(BASE, filename)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df[col].sum() / len(df) * 100
    except:
        return None

print("\n" + "="*72)
print("          VANILLA RAG  vs  AGENTIC RAG  —  All Metrics")
print("="*72)
print(f"{'Metric':<24} {'Vanilla RAG':>12} {'Agentic RAG':>12} {'Delta':>10}")
print("-"*72)

for name, vanilla_file, agentic_file, col in METRICS:
    v = pct(vanilla_file, col)
    a = pct(agentic_file, col)
    vs  = f"{v:.2f}%" if v is not None else "N/A"
    as_ = f"{a:.2f}%" if a is not None else "N/A"
    if v is not None and a is not None:
        delta = a - v
        ds = f"{'+' if delta>=0 else ''}{delta:.2f}%"
        winner = " <-- agentic wins" if delta > 1 else (" <-- vanilla wins" if delta < -1 else "")
    else:
        ds, winner = "N/A", ""
    print(f"{name:<24} {vs:>12} {as_:>12} {ds:>10}{winner}")

print("-"*72)
for name, col in RECALL:
    v = recall_pct("recall_at_k_results.csv", col)
    a = recall_pct("recall_at_k_agentic.csv", col)
    vs  = f"{v:.2f}%" if v is not None else "N/A"
    as_ = f"{a:.2f}%" if a is not None else "N/A"
    if v is not None and a is not None:
        delta = a - v
        ds = f"{'+' if delta>=0 else ''}{delta:.2f}%"
    else:
        ds = "N/A"
    print(f"{name:<24} {vs:>12} {as_:>12} {ds:>10}")

print("="*72)

# Per hop-type breakdown
print("\nCorrectness by hop type (Agentic RAG):")
try:
    ag = pd.read_csv(os.path.join(BASE, "generated_answers_agentic.csv"))
    co = pd.read_csv(os.path.join(BASE, "correctness_agentic.csv"))
    ag["correctness"] = co["correctness"].values

    vn = pd.read_csv(os.path.join(BASE, "generated_answers.csv"))
    cv = pd.read_csv(os.path.join(BASE, "correctness_scores.csv"))
    vn["correctness"] = cv["correctness"].values

    hop_types = ag["hop_type"].unique()
    print(f"{'hop_type':<22} {'Vanilla':>8} {'Agentic':>8} {'Delta':>8}")
    print("-"*50)
    for ht in sorted(hop_types):
        ag_grp = ag[ag["hop_type"] == ht]
        vn_grp = vn[vn.get("hop_type", pd.Series(dtype=str)) == ht] if "hop_type" in vn.columns else pd.DataFrame()
        ag_acc = ag_grp["correctness"].mean() * 100
        vn_acc = vn_grp["correctness"].mean() * 100 if len(vn_grp) > 0 else None
        vs = f"{vn_acc:.1f}%" if vn_acc is not None else "N/A"
        delta = f"{ag_acc-vn_acc:+.1f}%" if vn_acc is not None else "N/A"
        print(f"{ht:<22} {vs:>8} {ag_acc:>7.1f}% {delta:>8}  (n={len(ag_grp)})")
except Exception as e:
    print(f"  (Could not compute breakdown: {e})")

print("="*72)
