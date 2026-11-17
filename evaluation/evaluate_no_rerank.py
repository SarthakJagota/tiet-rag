"""
Generates RAG answers WITHOUT cross-encoder reranking (plain similarity
search, top_k=8). Same Groq model and prompt as the reranked pipeline, so
the comparison isolates the effect of reranking.

Reads  : tiet_rag_filtered.csv (367 Q)
Saves  : evaluation/generated_answers_no_rerank.csv
"""
import os, sys
sys.path.append(os.path.dirname(__file__))

import evaluate

if __name__ == "__main__":
    evaluate.main(rerank=False)
