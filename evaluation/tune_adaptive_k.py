"""
Adaptive-K tuning — compares dynamic K selection strategies vs fixed K=8 baseline.

Strategies tested:
  FIXED K=8      : retrieve 25 candidates -> cross-encoder -> top 8  (baseline)
  THRESHOLD -5   : keep chunks with cross-encoder score >= -5
  THRESHOLD  0   : keep chunks with cross-encoder score >=  0
  THRESHOLD  2   : keep chunks with cross-encoder score >=  2
  THRESHOLD  5   : keep chunks with cross-encoder score >=  5
  KNEE           : cut at steepest score drop between consecutive chunks

All strategies use retrieve_k=50 candidate pool, min_k=3, max_k=15.
Same individual scorer scripts as run_all.py — directly comparable to baseline.

Usage:
    python evaluation/tune_adaptive_k.py
"""
import os, sys, json, random, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import pandas as pd
from langchain_ollama import ChatOllama
from src.embeddings import get_embedding_model
from src.vectordb import load_vectordb
from src.reranker import get_reranker, rerank, rerank_threshold, rerank_knee

CSV_PATH = os.path.join(os.path.dirname(EVAL_DIR), "tiet_rag_filtered.csv")

# Configs: (label, mode, ce_threshold)
# mode="fixed"     -> fixed top_k=8, retrieve_k=25  (baseline)
# mode="threshold" -> adaptive threshold
# mode="knee"      -> adaptive knee detection
CONFIGS = [
    {"label": "fixed_k8",   "mode": "fixed",     "ce_threshold": None, "top_k": 8,  "retrieve_k": 25},
    {"label": "thresh_-5",  "mode": "threshold",  "ce_threshold": -5.0, "retrieve_k": 50},
    {"label": "thresh_0",   "mode": "threshold",  "ce_threshold":  0.0, "retrieve_k": 50},
    {"label": "thresh_2",   "mode": "threshold",  "ce_threshold":  2.0, "retrieve_k": 50},
    {"label": "thresh_5",   "mode": "threshold",  "ce_threshold":  5.0, "retrieve_k": 50},
    {"label": "knee",       "mode": "knee",       "ce_threshold": None, "retrieve_k": 50},
]

MIN_K    = 3
MAX_K    = 15
SAMPLE_N = 30
SEED     = 42

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


def retrieve_docs(vectordb, reranker, query, cfg):
    mode = cfg["mode"]
    if mode == "fixed":
        candidates = vectordb.similarity_search(query, k=cfg["retrieve_k"])
        return rerank(reranker, query, candidates, cfg["top_k"])
    elif mode == "threshold":
        candidates = vectordb.similarity_search(query, k=cfg["retrieve_k"])
        return rerank_threshold(reranker, query, candidates,
                                ce_threshold=cfg["ce_threshold"],
                                min_k=MIN_K, max_k=MAX_K)
    else:  # knee
        candidates = vectordb.similarity_search(query, k=cfg["retrieve_k"])
        return rerank_knee(reranker, query, candidates, min_k=MIN_K, max_k=MAX_K)


def generate_answers(data, vectordb, reranker, llm, cfg, out_path):
    rows = []
    recall_hits = {5: 0, 7: 0, 10: 0, 20: 0}
    k_counts    = []

    for i, item in enumerate(data):
        q      = item["question"]
        gt_srcs = [s.strip().lower() for s in item["ground_truth_source"].split("|") if s.strip()]

        try:
            docs        = retrieve_docs(vectordb, reranker, q, cfg)
            k_counts.append(len(docs))
            context_raw = "\n\n".join(d.page_content for d in docs)
            context_fmt = "\n\n".join(
                f"[{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}] {d.page_content}"
                for d in docs
            )
            answer   = llm.invoke(GENERATE_PROMPT.format(context=context_fmt, question=q)).content.strip()
            ret_srcs = [d.metadata.get("source", "").lower() for d in docs]
        except Exception as e:
            answer, context_raw, ret_srcs, docs = f"ERROR: {e}", "", [], []
            k_counts.append(0)

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
            "chunks_used":         len(docs),
        })
        print(f"  [{i+1}/{len(data)}] k_used={len(docs)}")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    avg_k = sum(k_counts) / len(k_counts) if k_counts else 0
    return recall_hits, avg_k


def run_scorers(answers_csv, tmp_prefix):
    import correctness, relevance, groundedness, retrieval_relevance, generation_quality
    _llm = ChatOllama(model="llama3.2", temperature=0, timeout=120)
    for mod in [correctness, relevance, groundedness, retrieval_relevance, generation_quality]:
        mod.llm = _llm

    results = {}

    def patch_run(mod, suffix, cols):
        tmp = tmp_prefix + suffix
        mod.INPUT  = answers_csv
        mod.OUTPUT = tmp
        mod.main()
        df = pd.read_csv(tmp)
        return {c: df[c].mean() * 100 for c in (cols if isinstance(cols, list) else [cols])}

    print("    [2/6] correctness ...")
    results.update(patch_run(correctness,         "_c.csv",  "correctness"))
    print("    [3/6] relevance ...")
    results.update(patch_run(relevance,           "_r.csv",  "relevance"))
    print("    [4/6] groundedness ...")
    results.update(patch_run(groundedness,        "_g.csv",  "groundedness"))
    print("    [5/6] retrieval_relevance ...")
    results.update(patch_run(retrieval_relevance, "_rr.csv", "retrieval_relevance"))
    print("    [6/6] generation_quality ...")
    results.update(patch_run(generation_quality,  "_gq.csv", ["specificity","conciseness","completeness"]))

    return results


def cleanup(prefix):
    for f in glob.glob(prefix + "*.csv"):
        try:
            os.remove(f)
        except Exception:
            pass


MKEYS  = ["correctness","relevance","groundedness","retrieval_relevance",
           "specificity","conciseness","completeness","R@5","R@7","R@10","R@20"]
MLABEL = ["correct","relevnc","groundd","ret_rel","specif","concise","complet","R@5","R@7","R@10","R@20"]


def print_table(rows):
    W = 135
    best = {m: max(r[m] for r in rows) for m in MKEYS}

    print("\n" + "="*W)
    print("    ADAPTIVE-K TUNING  (30 questions | all metrics | same scorers as run_all.py)")
    print("="*W)
    header = f"{'strategy':<14}  {'avg_k':>5}  " + "  ".join(f"{lb:>9}" for lb in MLABEL)
    print(header)
    print("-"*W)

    for r in rows:
        tag = " <- baseline" if r["label"] == "fixed_k8" else ""
        row = f"{r['label']:<14}  {r['avg_k']:>4.1f}   "
        row += "  ".join(
            f"{r[m]:>8.1f}%{'*' if r[m] == best[m] else ' '}"
            for m in MKEYS
        )
        print(row + tag)

    print("="*W)
    print("* = best value for that metric\n")


def main():
    print("Loading dataset ...")
    all_data = load_dataset()
    print(f"  Total: {len(all_data)} questions")

    random.seed(SEED)
    data = random.sample(all_data, SAMPLE_N)
    print(f"  Sampled: {SAMPLE_N} questions (seed={SEED})\n")

    print("Loading models ...")
    embedding_model = get_embedding_model()
    vectordb        = load_vectordb(embedding_model)
    reranker_model  = get_reranker()
    llm             = ChatOllama(model="llama3.2", temperature=0, timeout=120)
    print("  Ready.\n")

    summary = []

    for cfg in CONFIGS:
        label = cfg["label"]
        print(f"\n{'='*55}")
        print(f"  Strategy: {label}")
        if cfg["mode"] == "threshold":
            print(f"  mode=threshold  ce_threshold={cfg['ce_threshold']}  min_k={MIN_K}  max_k={MAX_K}")
        elif cfg["mode"] == "knee":
            print(f"  mode=knee  min_k={MIN_K}  max_k={MAX_K}")
        else:
            print(f"  mode=fixed  top_k={cfg['top_k']}  retrieve_k={cfg['retrieve_k']}")
        print(f"{'='*55}")

        ans_csv = os.path.join(EVAL_DIR, f"_adaptive_{label}_answers.csv")
        tmp_pre = os.path.join(EVAL_DIR, f"_adaptive_{label}")

        print(f"  [1/6] generating answers ...")
        recall, avg_k = generate_answers(data, vectordb, reranker_model, llm, cfg, ans_csv)

        scores = run_scorers(ans_csv, tmp_pre)
        n = len(data)

        result = {
            "label":   label,
            "avg_k":   avg_k,
            "correctness":         scores.get("correctness",         0),
            "relevance":           scores.get("relevance",           0),
            "groundedness":        scores.get("groundedness",        0),
            "retrieval_relevance": scores.get("retrieval_relevance", 0),
            "specificity":         scores.get("specificity",         0),
            "conciseness":         scores.get("conciseness",         0),
            "completeness":        scores.get("completeness",        0),
            "R@5":  recall[5]  / n * 100,
            "R@7":  recall[7]  / n * 100,
            "R@10": recall[10] / n * 100,
            "R@20": recall[20] / n * 100,
        }
        summary.append(result)

        print(f"\n  {label} (avg_k={avg_k:.1f}): "
              f"correct={result['correctness']:.1f}%  "
              f"groundd={result['groundedness']:.1f}%  "
              f"R@10={result['R@10']:.1f}%")

        cleanup(tmp_pre)
        try:
            os.remove(ans_csv)
        except Exception:
            pass

    print_table(summary)

    out = os.path.join(EVAL_DIR, "tune_adaptive_k_summary.csv")
    pd.DataFrame(summary).to_csv(out, index=False)
    print(f"Summary saved -> {out}")


if __name__ == "__main__":
    main()
