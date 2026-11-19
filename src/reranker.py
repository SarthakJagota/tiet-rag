import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker():
    try:
        return CrossEncoder(MODEL_NAME, local_files_only=True)
    except Exception:
        return CrossEncoder(MODEL_NAME)


def rerank(reranker, query, docs, top_k):
    """Fixed-K reranking (original behaviour)."""
    pairs  = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def rerank_with_scores(reranker, query, docs):
    """Return all (doc, cross-encoder-score) pairs sorted best-first."""
    pairs  = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)


def rerank_threshold(reranker, query, docs, ce_threshold=0.0, min_k=3, max_k=15):
    """
    Adaptive K — keep every doc whose cross-encoder score >= ce_threshold.
    Bounds: always return at least min_k, at most max_k docs.
    """
    ranked   = rerank_with_scores(reranker, query, docs)
    filtered = [(d, s) for d, s in ranked if s >= ce_threshold]

    if len(filtered) < min_k:
        filtered = ranked[:min_k]
    if len(filtered) > max_k:
        filtered = filtered[:max_k]

    return [d for d, _ in filtered]


def rerank_knee(reranker, query, docs, min_k=3, max_k=15):
    """
    Adaptive K — cut at the steepest score drop between consecutive docs.
    Only considers positions between min_k and max_k.
    Falls back to min_k if no clear knee is found.
    """
    ranked = rerank_with_scores(reranker, query, docs)
    scores = [s for _, s in ranked]
    upper  = min(max_k, len(scores))

    if len(scores) <= min_k:
        return [d for d, _ in ranked]

    # drops[i] = score[i] - score[i+1],  index in 0-based terms
    # We search positions min_k-1 .. upper-2 (i.e. cut after min_k .. upper-1 docs)
    drops  = [scores[i] - scores[i + 1] for i in range(min_k - 1, upper - 1)]
    if drops:
        cut = min_k + drops.index(max(drops))   # number of docs to keep
    else:
        cut = min_k

    return [d for d, _ in ranked[:cut]]
