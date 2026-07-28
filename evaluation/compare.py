"""
Prints a side-by-side comparison of WITH vs WITHOUT reranking results.

Requires both sets of score CSVs to exist:
  WITH reranking    : correctness_scores.csv, relevance_scores.csv, etc.
  WITHOUT reranking : correctness_no_rerank.csv, relevance_no_rerank.csv, etc.

Usage:
    python evaluation/compare.py
"""
import os
import pandas as pd

BASE = os.path.dirname(__file__)


def pct(df, col, val=1):
    return (df[col] == val).sum() / len(df) * 100


def load(filename):
    path = os.path.join(BASE, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── Load WITH reranking ────────────────────────────────────────────────────────
c_r   = load("correctness_scores.csv")
rel_r = load("relevance_scores.csv")
gnd_r = load("groundedness_scores.csv")
rr_r  = load("retrieval_relevance_scores.csv")
gq_r  = load("generation_quality_scores.csv")
rk_r  = load("recall_at_k_results.csv")

# ── Load WITHOUT reranking ─────────────────────────────────────────────────────
c_nr   = load("correctness_no_rerank.csv")
rel_nr = load("relevance_no_rerank.csv")
gnd_nr = load("groundedness_no_rerank.csv")
rr_nr  = load("retrieval_relevance_no_rerank.csv")
gq_nr  = load("generation_quality_no_rerank.csv")
rk_nr  = load("recall_at_k_results.csv")   # same recall file (retrieval only)

# ── Helper ─────────────────────────────────────────────────────────────────────
def fmt(val):
    return f"{val:.2f}%" if isinstance(val, float) else "N/A"

def get(df, col, val=1):
    if df is None or col not in df.columns:
        return None
    return pct(df, col, val)

# ── Compute ────────────────────────────────────────────────────────────────────
rows = [
    ("Correctness",          get(c_r,   "correctness"),       get(c_nr,   "correctness")),
    ("Relevance",            get(rel_r, "relevance"),         get(rel_nr, "relevance")),
    ("Groundedness",         get(gnd_r, "groundedness"),      get(gnd_nr, "groundedness")),
    ("Retrieval Relevance",  get(rr_r,  "retrieval_relevance"), get(rr_nr, "retrieval_relevance")),
    ("Specificity",          get(gq_r,  "specificity"),       get(gq_nr, "specificity")),
    ("Conciseness",          get(gq_r,  "conciseness"),       get(gq_nr, "conciseness")),
    ("Completeness",         get(gq_r,  "completeness"),      get(gq_nr, "completeness")),
]

rk_rows = []
if rk_r is not None:
    n = len(rk_r)
    for k in [5, 7, 10, 20]:
        col = f"hit@{k}"
        if col in rk_r.columns:
            rk_rows.append((f"Recall@{k}", rk_r[col].sum()/n*100, None))

# ── Print ──────────────────────────────────────────────────────────────────────
W = 62
print("\n" + "="*W)
print("         RERANKING vs NO RERANKING — COMPARISON")
print("         Dataset: tiet_rag_filtered.csv (367 Q)")
print("="*W)
print(f"{'Metric':<26} {'With Reranking':>16} {'No Reranking':>16}")
print("-"*W)

for name, with_r, without_r in rows:
    print(f"{name:<26} {fmt(with_r):>16} {fmt(without_r):>16}")

if rk_rows:
    print("-"*W)
    print(f"{'Recall@K (retrieval only — same for both)':}")
    for name, val, _ in rk_rows:
        print(f"  {name:<24} {fmt(val):>16}")

print("="*W)

# ── Delta ─────────────────────────────────────────────────────────────────────
print("\nDelta (With - Without Reranking):")
print("-"*W)
for name, with_r, without_r in rows:
    if with_r is not None and without_r is not None:
        delta = with_r - without_r
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:<24} {sign}{delta:.2f}%")
print("="*W + "\n")
