"""
Score the agentic RAG answers using all individual scorer scripts.
Same methodology as run_all.py — scores are directly comparable.

Input  : evaluation/generated_answers_agentic.csv
Outputs: evaluation/*_agentic.csv
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

BASE    = os.path.dirname(__file__)
AGENTIC = os.path.join(BASE, "generated_answers_agentic.csv")

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


section("STEP 1 / 6 — Correctness")
import correctness
correctness.INPUT  = AGENTIC
correctness.OUTPUT = os.path.join(BASE, "correctness_agentic.csv")
correctness.main()

section("STEP 2 / 6 — Relevance")
import relevance
relevance.INPUT  = AGENTIC
relevance.OUTPUT = os.path.join(BASE, "relevance_agentic.csv")
relevance.main()

section("STEP 3 / 6 — Groundedness")
import groundedness
groundedness.INPUT  = AGENTIC
groundedness.OUTPUT = os.path.join(BASE, "groundedness_agentic.csv")
groundedness.main()

section("STEP 4 / 6 — Retrieval Relevance")
import retrieval_relevance
retrieval_relevance.INPUT  = AGENTIC
retrieval_relevance.OUTPUT = os.path.join(BASE, "retrieval_relevance_agentic.csv")
retrieval_relevance.main()

section("STEP 5 / 6 — Generation Quality")
import generation_quality
generation_quality.INPUT  = AGENTIC
generation_quality.OUTPUT = os.path.join(BASE, "generation_quality_agentic.csv")
generation_quality.main()

section("STEP 6 / 6 — Recall @ K")
import recall_at_k
recall_at_k.OUTPUT = os.path.join(BASE, "recall_at_k_agentic.csv")
recall_at_k.main()

# ── Summary ────────────────────────────────────────────────────────────────────
section("FINAL SUMMARY — Agentic RAG")

def pct(path, col):
    try:
        df = pd.read_csv(path)
        return df[col].mean() * 100
    except:
        return None

corr  = pct(os.path.join(BASE,"correctness_agentic.csv"),         "correctness")
rel   = pct(os.path.join(BASE,"relevance_agentic.csv"),            "relevance")
gnd   = pct(os.path.join(BASE,"groundedness_agentic.csv"),         "groundedness")
rr    = pct(os.path.join(BASE,"retrieval_relevance_agentic.csv"),  "retrieval_relevance")
gq    = pd.read_csv(os.path.join(BASE,"generation_quality_agentic.csv"))
spec  = gq["specificity"].mean()  * 100
conc  = gq["conciseness"].mean()  * 100
comp  = gq["completeness"].mean() * 100

try:
    rk = pd.read_csv(os.path.join(BASE,"recall_at_k_agentic.csv"))
    n  = len(rk)
    r5  = rk["hit@5"].sum()  / n * 100
    r10 = rk["hit@10"].sum() / n * 100
    r20 = rk["hit@20"].sum() / n * 100
except:
    r5 = r10 = r20 = None

print("\n" + "="*65)
print("         AGENTIC RAG — EVALUATION RESULTS (367 Q)")
print("="*65)
print(f"Correctness          : {corr:.2f}%")
print(f"Relevance            : {rel:.2f}%")
print(f"Groundedness         : {gnd:.2f}%")
print(f"Retrieval Relevance  : {rr:.2f}%")
print(f"Specificity          : {spec:.2f}%")
print(f"Conciseness          : {conc:.2f}%")
print(f"Completeness         : {comp:.2f}%")
if r5:
    print(f"Recall@5             : {r5:.2f}%")
    print(f"Recall@10            : {r10:.2f}%")
    print(f"Recall@20            : {r20:.2f}%")
print("="*65)

# ── Per hop-type breakdown ─────────────────────────────────────────────────────
try:
    ag = pd.read_csv(AGENTIC)
    co = pd.read_csv(os.path.join(BASE,"correctness_agentic.csv"))
    ag["correctness"] = co["correctness"]

    print("\nCorrectness by hop type:")
    for ht, grp in ag.groupby("hop_type"):
        acc = grp["correctness"].mean() * 100
        print(f"  {ht:<20}: {acc:.1f}%  (n={len(grp)})")

    print("\nCorrectness by agent classification:")
    for qt, grp in ag.groupby("query_type"):
        acc = grp["correctness"].mean() * 100
        print(f"  {qt:<10}: {acc:.1f}%  (n={len(grp)})")
except Exception as e:
    print(f"(Could not compute breakdown: {e})")

print("="*65)
