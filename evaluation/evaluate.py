"""
Step 1 — Generate RAG answers for all questions.
Dataset: tiet_rag_filtered.csv (367 Q)

Supports resume: if generated_answers.csv already exists, skips
questions that were already answered and appends the rest.

Saves: evaluation/generated_answers.csv
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.rag import RAGPipeline

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tiet_rag_filtered.csv")
OUTPUT   = os.path.join(os.path.dirname(__file__), "generated_answers.csv")


def load_combined():
    df = pd.read_csv(CSV_PATH)
    items = []
    for _, row in df.iterrows():
        items.append({
            "question":            str(row["question"]),
            "ground_truth":        str(row["answer"]),
            "ground_truth_source": str(row.get("sources", "")),
            "dataset":             "filtered",
        })
    return items


def main():
    data = load_combined()
    print(f"Dataset: {len(data)} questions (tiet_rag_filtered.csv)")

    # Resume: load already-answered questions that belong to current dataset
    valid_questions = set(item["question"].strip() for item in data)
    already_done = set()
    existing_rows = []
    if os.path.exists(OUTPUT):
        existing_df = pd.read_csv(OUTPUT)
        existing_df["question"] = existing_df["question"].astype(str).str.strip()
        # Only keep rows that are actually in the current dataset
        existing_df = existing_df[existing_df["question"].isin(valid_questions)].drop_duplicates(subset="question", keep="last")
        existing_rows = existing_df.to_dict("records")
        already_done = set(existing_df["question"].tolist())
        print(f"Resuming: {len(already_done)} already done, {len(data) - len(already_done)} remaining.\n")
    else:
        print("Starting fresh.\n")

    rag = RAGPipeline()
    print("RAG pipeline ready.\n")

    rows = list(existing_rows)
    for i, item in enumerate(data):
        question = item["question"]

        if question.strip() in already_done:
            print(f"[{i+1}/{len(data)}] skip (already done)")
            continue

        try:
            answer, docs = rag.answer(question)
            context  = "\n\n".join(d.page_content for d in docs)
            ret_srcs = "|".join(d.metadata.get("source", "") for d in docs)
        except Exception as e:
            print(f"[{i+1}/{len(data)}] ERROR: {e}")
            answer, context, ret_srcs = f"ERROR: {e}", "", ""

        rows.append({
            "question":            question,
            "generated_answer":    answer,
            "ground_truth":        item["ground_truth"],
            "context":             context,
            "retrieved_sources":   ret_srcs,
            "ground_truth_source": item["ground_truth_source"],
            "dataset":             item["dataset"],
        })
        # Save after every question so progress is never lost
        pd.DataFrame(rows).to_csv(OUTPUT, index=False)
        print(f"[{i+1}/{len(data)}] done")

    print(f"\nSaved {len(rows)} rows -> {OUTPUT}")
    print("Next step: python evaluation/run_all.py")


if __name__ == "__main__":
    main()
