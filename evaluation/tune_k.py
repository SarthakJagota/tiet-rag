"""
K-value tuning — reranking vs no-reranking comparison across K values.

For each top_k (4, 6, 8, 10, 12):
  - WITH reranking    : retrieve retrieve_k candidates -> cross-encoder -> top_k
  - WITHOUT reranking : direct similarity_search(k=top_k)

Metrics (same scorer scripts as run_all.py — directly comparable to baseline):
  Correctness, Relevance, Groundedness, Retrieval Relevance,
  Specificity, Conciseness, Completeness  (all binary 0/1)
  Recall@5, Recall@7, Recall@10, Recall@20  (deterministic)

Usage:
    python evaluation/tune_k.py
"""
import os, sys, json, random, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import pandas as pd
try:
    from langchain_groq import ChatGroq
    from src.rag import GROQ_MODEL
    _default_generator = ChatGroq(model=GROQ_MODEL, temperature=0, max_retries=5)
except Exception:
    try:
        from langchain_ollama import ChatOllama
        _default_generator = ChatOllama(model="llama3.2", temperature=0, timeout=120)
    except Exception:
        _default_generator = None

from src.embeddings import get_embedding_model
from src.vectordb import load_vectordb
from src.reranker import get_reranker, rerank

CSV_PATH = os.path.join(os.path.dirname(EVAL_DIR), "tiet_rag_filtered.csv")

CONFIGS = [
    {"top_k":  4, "retrieve_k": 15},
    {"top_k":  6, "retrieve_k": 20},
    {"top_k":  8, "retrieve_k": 25},   # current default
    {"top_k": 10, "retrieve_k": 30},
    {"top_k": 12, "retrieve_k": 35},
]

SAMPLE_N = 30

GENERATE_PROMPT = """
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


def load_dataset():
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


def generate_answers(data, vectordb, reranker, llm, top_k, retrieve_k, use_rerank, out_path):
    """
    Generate answers and save CSV (same format as generated_answers.csv).
    use_rerank=True  -> similarity_search(retrieve_k) -> cross-encoder -> top_k
    use_rerank=False -> similarity_search(top_k) directly
    Returns recall_hits dict.
    """
    rows = []
    recall_hits = {5: 0, 7: 0, 10: 0, 20: 0}
    mode = "rerank" if use_rerank else "no-rerank"

    for i, item in enumerate(data):
        q      = item["question"]
        # sources may be pipe-separated: "A.pdf | B.pdf"
        gt_srcs = [s.strip().lower() for s in item["ground_truth_source"].split("|") if s.strip()]

        try:
            if use_rerank:
                candidates  = vectordb.similarity_search(q, k=retrieve_k)
                docs        = rerank(reranker, q, candidates, top_k)
            else:
                docs        = vectordb.similarity_search(q, k=top_k)

            context_raw = "\n\n".join(d.page_content for d in docs)
            context_fmt = "\n\n".join(
                f"[{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}] {d.page_content}"
                for d in docs
            )
            answer   = llm.invoke(GENERATE_PROMPT.format(context=context_fmt, question=q)).content.strip()
            ret_srcs = [d.metadata.get("source", "").lower() for d in docs]
        except Exception as e:
            answer, context_raw, ret_srcs, docs = f"ERROR: {e}", "", [], []

        for k in recall_hits:
            top = ret_srcs[:k]
            if any(any(gs in rs or rs in gs for rs in top) for gs in gt_srcs):
                recall_hits[k] += 1

        rows.append({
            "question":            q,
            "generated_answer":    answer,
            "ground_truth":        item["ground_truth"],
            "context":             context_raw,
            "retrieved_sources":   "|".join(d.metadata.get("source", "") for d in docs),
            "ground_truth_source": item["ground_truth_source"],
            "dataset":             item["dataset"],
        })
        print(f"  [{mode}] [{i+1}/{len(data)}] generated")

    print()
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return recall_hits


def run_scorers(answers_csv, tmp_prefix):
    """Run all 5 individual scorer scripts on answers_csv. Returns dict of metric -> score%."""
    import correctness, relevance, groundedness, retrieval_relevance, generation_quality
    import judge
    _llm = judge.llm
    for mod in [correctness, relevance, groundedness, retrieval_relevance, generation_quality]:
        mod.llm = _llm

    results = {}

    def patch_and_run(mod, suffix, col_or_cols):
        tmp = tmp_prefix + suffix
        mod.INPUT  = answers_csv
        mod.OUTPUT = tmp
        mod.main()
        df = pd.read_csv(tmp)
        if isinstance(col_or_cols, list):
            return {c: df[c].mean() * 100 for c in col_or_cols}
        return {col_or_cols: df[col_or_cols].mean() * 100}

    print("    [2/6] correctness ...")
    results.update(patch_and_run(correctness,        "_c.csv",  "correctness"))

    print("    [3/6] relevance ...")
    results.update(patch_and_run(relevance,          "_r.csv",  "relevance"))

    print("    [4/6] groundedness ...")
    results.update(patch_and_run(groundedness,       "_g.csv",  "groundedness"))

    print("    [5/6] retrieval_relevance ...")
    results.update(patch_and_run(retrieval_relevance,"_rr.csv", "retrieval_relevance"))

    print("    [6/6] generation_quality ...")
    results.update(patch_and_run(generation_quality, "_gq.csv", ["specificity","conciseness","completeness"]))

    return results


def cleanup(prefix):
    for f in glob.glob(prefix + "*.csv"):
        try:
            os.remove(f)
        except Exception:
            pass


METRICS = [
    ("correctness_%",   "correct"),
    ("relevance_%",     "relevnc"),
    ("groundedness_%",  "groundd"),
    ("retrieval_rel_%", "ret_rel"),
    ("specificity_%",   "specif"),
    ("conciseness_%",   "concise"),
    ("completeness_%",  "complet"),
    ("R@5_%",           "R@5"),
    ("R@7_%",           "R@7"),
    ("R@10_%",          "R@10"),
    ("R@20_%",          "R@20"),
]
MKEYS  = [m[0] for m in METRICS]
MLABEL = [m[1] for m in METRICS]


def build_result(scores, recall_hits, n, top_k, retrieve_k, mode):
    return {
        "top_k":         top_k,
        "retrieve_k":    retrieve_k,
        "mode":          mode,
        "correctness_%":   scores.get("correctness",         0) ,
        "relevance_%":     scores.get("relevance",           0) ,
        "groundedness_%":  scores.get("groundedness",        0) ,
        "retrieval_rel_%": scores.get("retrieval_relevance", 0) ,
        "specificity_%":   scores.get("specificity",         0) ,
        "conciseness_%":   scores.get("conciseness",         0) ,
        "completeness_%":  scores.get("completeness",        0) ,
        "R@5_%":  recall_hits[5]  / n * 100,
        "R@7_%":  recall_hits[7]  / n * 100,
        "R@10_%": recall_hits[10] / n * 100,
        "R@20_%": recall_hits[20] / n * 100,
    }


def print_tables(rerank_rows, norerank_rows):
    W = 130

    def table(title, rows, star=True):
        best = {m: max(r[m] for r in rows) for m in MKEYS} if star else {}
        print("\n" + "=" * W)
        print(f"    {title}")
        print("=" * W)
        header = f"{'top_k':>6}  {'rk':>4}  " + "  ".join(f"{lb:>9}" for lb in MLABEL)
        print(header)
        print("-" * W)
        for r in rows:
            tag = " <- current" if r["top_k"] == 8 else ""
            row = f"{r['top_k']:>6}  {r['retrieve_k']:>4}  "
            row += "  ".join(
                f"{r[m]:>8.1f}%{'*' if star and r[m] == best[m] else ' '}"
                for m in MKEYS
            )
            print(row + tag)
        print("=" * W)
        if star:
            print("* = best value for that metric")

    def delta_table(r_rows, nr_rows):
        print("\n" + "=" * W)
        print("    DELTA  (reranking MINUS no-reranking)  — positive = reranking wins")
        print("=" * W)
        header = f"{'top_k':>6}  {'rk':>4}  " + "  ".join(f"{lb:>9}" for lb in MLABEL)
        print(header)
        print("-" * W)
        for r, nr in zip(r_rows, nr_rows):
            tag = " <- current" if r["top_k"] == 8 else ""
            row = f"{r['top_k']:>6}  {r['retrieve_k']:>4}  "
            for m in MKEYS:
                d = r[m] - nr[m]
                marker = "+" if d > 0 else (" " if d == 0 else "-")
                row += f"  {marker}{abs(d):>6.1f}%  "
            print(row + tag)
        print("=" * W)

    table("WITH RERANKING (cross-encoder)", rerank_rows)
    table("WITHOUT RERANKING (direct similarity search)", norerank_rows)
    delta_table(rerank_rows, norerank_rows)


def main():
    print("Loading dataset ...")
    all_data = load_dataset()
    print(f"  Total: {len(all_data)} questions (tiet_rag_filtered.csv)")

    data   = all_data if SAMPLE_N is None else random.sample(all_data, SAMPLE_N)
    n      = len(data)
    print(f"  Using: {n} questions\n")

    print("Loading models (embedding, reranker, LLM) ...")
    embedding_model = get_embedding_model()
    vectordb        = load_vectordb(embedding_model)
    reranker_model  = get_reranker()
    llm             = _default_generator
    print("  Ready.\n")

    rerank_rows   = []
    norerank_rows = []

    for cfg in CONFIGS:
        top_k, retrieve_k = cfg["top_k"], cfg["retrieve_k"]
        tag = " <- current" if top_k == 8 else ""
        print(f"\n{'='*60}")
        print(f"  Config: top_k={top_k}, retrieve_k={retrieve_k}{tag}")
        print(f"{'='*60}")

        for use_rerank in [True, False]:
            mode     = "rerank" if use_rerank else "norerank"
            ans_csv  = os.path.join(EVAL_DIR, f"_tune_k{top_k}_{mode}_answers.csv")
            tmp_pre  = os.path.join(EVAL_DIR, f"_tune_k{top_k}_{mode}")

            print(f"\n  -- {'WITH' if use_rerank else 'WITHOUT'} reranking --")
            print(f"  [1/6] generating answers ...")
            recall = generate_answers(
                data, vectordb, reranker_model, llm,
                top_k, retrieve_k, use_rerank, ans_csv
            )

            scores = run_scorers(ans_csv, tmp_pre)
            result = build_result(scores, recall, n, top_k, retrieve_k, mode)

            if use_rerank:
                rerank_rows.append(result)
            else:
                norerank_rows.append(result)

            print(f"  correct={result['correctness_%']:.1f}%  "
                  f"relevnc={result['relevance_%']:.1f}%  "
                  f"groundd={result['groundedness_%']:.1f}%  "
                  f"R@10={result['R@10_%']:.1f}%")

            cleanup(tmp_pre)
            try:
                os.remove(ans_csv)
            except Exception:
                pass

    print_tables(rerank_rows, norerank_rows)

    out = os.path.join(EVAL_DIR, "tune_k_summary.csv")
    all_rows = []
    for r, nr in zip(rerank_rows, norerank_rows):
        all_rows.append(r)
        all_rows.append(nr)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nSummary saved -> {out}")


if __name__ == "__main__":
    main()
