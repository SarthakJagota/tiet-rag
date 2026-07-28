"""
Generate RAG answers using the AgenticRAGPipeline.
Supports resume — skips questions already answered.

Dataset : tiet_rag_filtered.csv (367 Q)
Saves   : evaluation/generated_answers_agentic.csv

Extra columns vs vanilla evaluate.py:
  query_type   — "simple" or "complex" (as classified by the agent)
  hop_type     — from the dataset (single_hop / 2_hop / 3_hop)
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.agent import AgenticRAGPipeline

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tiet_rag_filtered.csv")
OUTPUT   = os.path.join(os.path.dirname(__file__), "generated_answers_agentic.csv")


def load_dataset():
    df = pd.read_csv(CSV_PATH)
    items = []
    for _, row in df.iterrows():
        items.append({
            "question":            str(row["question"]),
            "ground_truth":        str(row["answer"]),
            "ground_truth_source": str(row.get("sources", "")),
            "hop_type":            str(row.get("hop_type", "")),
            "dataset":             "filtered",
        })
    return items


def main():
    data = load_dataset()
    print(f"Dataset: {len(data)} questions (tiet_rag_filtered.csv)\n")

    # Resume support
    valid_questions = set(item["question"].strip() for item in data)
    already_done    = set()
    existing_rows   = []
    if os.path.exists(OUTPUT):
        existing_df = pd.read_csv(OUTPUT)
        existing_df["question"] = existing_df["question"].astype(str).str.strip()
        existing_df = existing_df[existing_df["question"].isin(valid_questions)].drop_duplicates(subset="question", keep="last")
        existing_rows = existing_df.to_dict("records")
        already_done  = set(existing_df["question"].tolist())
        print(f"Resuming: {len(already_done)} done, {len(data)-len(already_done)} remaining.\n")

    agent = AgenticRAGPipeline(verbose=True)
    print("Agentic RAG pipeline ready.\n")

    rows = list(existing_rows)
    for i, item in enumerate(data):
        question = item["question"]

        if question.strip() in already_done:
            print(f"[{i+1}/{len(data)}] skip")
            continue

        try:
            answer, docs = agent.answer(question)
            context      = "\n\n".join(d.page_content for d in docs)
            ret_srcs     = "|".join(d.metadata.get("source", "") for d in docs)
            query_type   = agent.classify(question)   # already cached in LLM but cheap to redo
        except Exception as e:
            print(f"[{i+1}/{len(data)}] ERROR: {e}")
            answer, context, ret_srcs, query_type = f"ERROR: {e}", "", "", "error"

        rows.append({
            "question":            question,
            "generated_answer":    answer,
            "ground_truth":        item["ground_truth"],
            "context":             context,
            "retrieved_sources":   ret_srcs,
            "ground_truth_source": item["ground_truth_source"],
            "hop_type":            item["hop_type"],
            "query_type":          query_type,
            "dataset":             item["dataset"],
        })

        pd.DataFrame(rows).to_csv(OUTPUT, index=False)
        print(f"[{i+1}/{len(data)}] done  (type={query_type}  sources={len(docs)})\n")

    print(f"\nSaved {len(rows)} rows -> {OUTPUT}")
    print("Next: python evaluation/run_all_agentic.py")


if __name__ == "__main__":
    main()
