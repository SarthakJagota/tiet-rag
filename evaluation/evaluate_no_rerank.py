"""
Generates RAG answers WITHOUT cross-encoder reranking.
Uses direct similarity search top_k=8 only.

Reads  : tiet_rag_filtered.csv (367 Q)
Saves  : evaluation/generated_answers_no_rerank.csv
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.embeddings import get_embedding_model
from src.vectordb import load_vectordb

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tiet_rag_filtered.csv")
OUTPUT   = os.path.join(os.path.dirname(__file__), "generated_answers_no_rerank.csv")

TOP_K = 8  # matches top_k in src/rag.py

PROMPT_TEMPLATE = """
You are an assistant for Thapar Institute of Engineering & Technology (TIET).
Answer the question using ONLY the information provided in the context below.

Rules:
- Give a complete, accurate answer that includes all relevant details from the context.
- If the answer involves multiple points, steps, or items, list them clearly.
- Copy names, numbers, dates, fees, and policy terms exactly as written in the context.
- Do not add information that is not in the context.
- Do not say "According to the context" or mention page/source numbers.
- If the answer is not found in the context, respond with exactly: I don't know.

Context:
{context}

Question:
{question}

Answer:
"""


def load_combined():
    df = pd.read_csv(CSV_PATH)
    items = []
    for _, row in df.iterrows():
        items.append({
            "question":            str(row["question"]),
            "ground_truth":        str(row["answer"]),
            "ground_truth_source": str(row.get("sources", "")),
        })
    return items


def main():
    data = load_combined()
    print(f"Dataset: {len(data)} questions | NO reranking | top_k={TOP_K}")

    embedding_model = get_embedding_model()
    vectordb        = load_vectordb(embedding_model)
    llm             = ChatOllama(model="llama3.2", temperature=0)
    prompt_tpl      = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rows = []
    for i, item in enumerate(data):
        question = item["question"]
        try:
            docs    = vectordb.similarity_search(question, k=TOP_K)
            context = "\n\n".join(
                f"[{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}] {d.page_content}"
                for d in docs
            )
            answer  = llm.invoke(prompt_tpl.format(context=context, question=question)).content
            ctx_raw = "\n\n".join(d.page_content for d in docs)
        except Exception as e:
            answer, context, ctx_raw, docs = f"ERROR: {e}", "", "", []

        rows.append({
            "question":            question,
            "generated_answer":    answer,
            "ground_truth":        item["ground_truth"],
            "context":             ctx_raw,
            "retrieved_sources":   "|".join(d.metadata.get("source","") for d in docs),
            "ground_truth_source": item["ground_truth_source"],
            "dataset":             item.get("dataset", "combined"),
        })
        print(f"[{i+1}/{len(data)}] done")

    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(rows)} rows → {OUTPUT}")


if __name__ == "__main__":
    main()
