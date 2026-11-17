"""
Step 1 — Generate RAG answers for all questions.
Dataset: tiet_rag_filtered.csv (367 Q)

Uses the same Groq model as the app (src/rag.py GROQ_MODEL).
    python evaluation/evaluate.py              -> with cross-encoder reranking
    python evaluation/evaluate_no_rerank.py    -> plain similarity search, same LLM + prompt

Supports resume: rows already answered by the CURRENT model are kept and
skipped; rows from an older model are regenerated. If Groq's rate/daily limit
stops the run, just run the same command again later.

Saves: evaluation/generated_answers.csv (or generated_answers_no_rerank.csv)
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from langchain_groq import ChatGroq
from src.rag import RAGPipeline, GROQ_MODEL

BASE     = os.path.dirname(__file__)
CSV_PATH = os.path.join(os.path.dirname(BASE), "tiet_rag_filtered.csv")
OUTPUT   = os.path.join(BASE, "generated_answers.csv")
OUTPUT_NO_RERANK = os.path.join(BASE, "generated_answers_no_rerank.csv")


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


def main(rerank=True):
    output = OUTPUT if rerank else OUTPUT_NO_RERANK
    data = load_combined()
    print(f"Dataset: {len(data)} questions | model={GROQ_MODEL} | reranking={'on' if rerank else 'off'}")

    # Resume: keep rows already answered by the current model
    valid_questions = set(item["question"].strip() for item in data)
    existing = {}
    if os.path.exists(output):
        df = pd.read_csv(output)
        if "model" in df.columns:
            df["question"] = df["question"].astype(str).str.strip()
            df = df[df["question"].isin(valid_questions) & (df["model"] == GROQ_MODEL)]
            df = df[~df["generated_answer"].astype(str).str.startswith("ERROR")]
            existing = {r["question"]: r for r in df.to_dict("records")}
    print(f"Already done with this model: {len(existing)}, remaining: {len(data) - len(existing)}\n")

    rag = RAGPipeline()
    rag.llm = ChatGroq(model=GROQ_MODEL, temperature=0, max_retries=8)  # wait out per-minute limits
    print("RAG pipeline ready.\n")

    rows = []
    for i, item in enumerate(data):
        question = item["question"]
        if question.strip() in existing:
            rows.append(existing[question.strip()])
            continue

        try:
            if rerank:
                answer, docs = rag.answer(question)
            else:
                docs = rag.vectordb.similarity_search(question, k=rag.top_k)
                context = "\n\n".join(
                    f"[{d.metadata.get('source', 'unknown')} p.{d.metadata.get('page', '?')}] {d.page_content}"
                    for d in docs
                )
                answer = rag.llm.invoke(rag.prompt.format(context=context, question=question)).content
        except Exception as e:
            pd.DataFrame(rows).to_csv(output, index=False)
            print(f"\n[{i+1}/{len(data)}] STOPPED: {str(e)[:300]}")
            print("Progress saved. Run the same command again later to continue.")
            sys.exit(1)

        rows.append({
            "question":            question,
            "generated_answer":    answer,
            "ground_truth":        item["ground_truth"],
            "context":             "\n\n".join(d.page_content for d in docs),
            "retrieved_sources":   "|".join(d.metadata.get("source", "") for d in docs),
            "ground_truth_source": item["ground_truth_source"],
            "dataset":             item["dataset"],
            "model":               GROQ_MODEL,
        })
        pd.DataFrame(rows).to_csv(output, index=False)  # save every question
        print(f"[{i+1}/{len(data)}] done")

    print(f"\nSaved {len(rows)} rows -> {output}")


if __name__ == "__main__":
    main()
