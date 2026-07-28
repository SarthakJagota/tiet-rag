"""
Information Retrieval Metric — RECALL @ K
Computes R@5, R@7, R@10, R@20.

For each question, retrieves top-20 chunks from the vector store.
A question is a hit at rank k if any of the top-k chunks comes from
any of the ground-truth source PDFs (supports pipe-separated multi-source).

Reads : tiet_rag_filtered.csv (367 Q)
Saves : evaluation/recall_at_k_results.csv
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.embeddings import get_embedding_model
from src.vectordb import load_vectordb

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tiet_rag_filtered.csv")
OUTPUT   = os.path.join(os.path.dirname(__file__), "recall_at_k_results.csv")

K_VALUES = [5, 7, 10, 20]
MAX_K    = max(K_VALUES)


def load_combined():
    df = pd.read_csv(CSV_PATH)
    items = []
    for _, row in df.iterrows():
        # sources may be pipe-separated: "A.pdf | B.pdf"
        raw = str(row.get("sources", ""))
        srcs = [s.strip().lower() for s in raw.split("|") if s.strip()]
        items.append({"question": str(row["question"]), "sources": srcs, "dataset": "filtered"})
    return items


def main():
    data = load_combined()
    print(f"Loaded {len(data)} questions (tiet_rag_filtered.csv).")

    embedding_model = get_embedding_model()
    vectordb = load_vectordb(embedding_model)
    print("Vector store loaded.\n")

    results = []
    hits = {k: 0 for k in K_VALUES}

    for i, item in enumerate(data):
        question   = item["question"]
        gt_sources = item["sources"]  # list of lowercase PDF names

        docs = vectordb.similarity_search(question, k=MAX_K)
        retrieved_sources = [d.metadata.get("source", "").lower() for d in docs]

        row_result = {"question": question, "ground_truth_sources": "|".join(gt_sources), "dataset": item["dataset"]}
        for k in K_VALUES:
            top_k = retrieved_sources[:k]
            # Hit if ANY ground-truth source appears in top-k
            hit = any(
                any(gs in rs or rs in gs for rs in top_k)
                for gs in gt_sources
            )
            row_result[f"hit@{k}"] = int(hit)
            if hit:
                hits[k] += 1

        results.append(row_result)
        print(f"[{i+1}/{len(data)}] " +
              " ".join(f"R@{k}={'Y' if row_result[f'hit@{k}'] else 'N'}" for k in K_VALUES))

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT, index=False)

    total = len(data)
    print(f"\n{'='*55}")
    print("RECALL @ K  (Information Retrieval)")
    print(f"{'='*55}")
    for k in K_VALUES:
        print(f"R@{k:2d}  :  {hits[k]:3d}/{total}  =  {hits[k]/total*100:.2f}%")
    print(f"{'='*55}")
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()

