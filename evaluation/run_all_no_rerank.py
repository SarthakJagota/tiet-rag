"""
Runs all scorers on generated_answers_no_rerank.csv.
Saves results with _no_rerank suffix so rerank results are not overwritten.

Usage:
    python evaluation/run_all_no_rerank.py
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE = os.path.dirname(__file__)

NR = os.path.join(BASE, "generated_answers_no_rerank.csv")

# ── Patch each module's INPUT/OUTPUT before calling main() ─────────────────────

import correctness
correctness.INPUT  = NR
correctness.OUTPUT = os.path.join(BASE, "correctness_no_rerank.csv")

import relevance
relevance.INPUT  = NR
relevance.OUTPUT = os.path.join(BASE, "relevance_no_rerank.csv")

import groundedness
groundedness.INPUT  = NR
groundedness.OUTPUT = os.path.join(BASE, "groundedness_no_rerank.csv")

import retrieval_relevance
retrieval_relevance.INPUT  = NR
retrieval_relevance.OUTPUT = os.path.join(BASE, "retrieval_relevance_no_rerank.csv")

import generation_quality
generation_quality.INPUT  = NR
generation_quality.OUTPUT = os.path.join(BASE, "generation_quality_no_rerank.csv")

# ── Run ────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  SCORING WITHOUT RERANKING")
print("="*60)

print("\n[1/5] Correctness ...")
correctness.main()

print("\n[2/5] Relevance ...")
relevance.main()

print("\n[3/5] Groundedness ...")
groundedness.main()

print("\n[4/5] Retrieval Relevance ...")
retrieval_relevance.main()

print("\n[5/5] Generation Quality ...")
generation_quality.main()

print("\nDone. Now run:  python evaluation/compare.py")
