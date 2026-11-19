# TIET Policy Assistant — RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system built over **110 TIET (Thapar Institute of Engineering & Technology) policy PDFs**, featuring cross-encoder reranking, adaptive K selection, and a comprehensive 7-metric evaluation framework.

**[▶ Watch the Demo Video](https://drive.google.com/file/d/1lcgUY-FYY1jipOcPvbUJcWk7kDqm1nDZ/view?usp=sharing)**

---

## Features

- **Vanilla RAG** — bi-encoder retrieval → cross-encoder reranking → LLM generation
- **Adaptive K** — dynamic chunk selection using score thresholds or knee detection
- **Conversational Memory** — context-aware query reformulation for natural multi-turn conversations
- **Streaming Responses** — real-time token streaming in Streamlit UI
- **7-metric evaluation** over 367 ground-truth questions (LLM-as-judge)
- **Streamlit chatbot** with source attribution, session stats, and metrics dashboard

---

## Architecture

```
User Query (+ chat history)
    │
    ▼
Query Rewriting (follow-ups → standalone query)
    │
    ▼
Embed Query
    │
    ▼
ChromaDB Similarity Search (retrieve_k=25)
    │
    ▼
Cross-Encoder Reranking (top_k=8, or adaptive K)
    │
    ▼
LLM Generation (Groq gpt-oss-120b, streamed)
    │
    ▼
Answer + Sources
```
---

## Evaluation Results

Evaluated on **367 questions** across 110 TIET policy PDFs using LLM-as-judge (binary 0/1 scoring, `llama3.2` judge unless noted).

### Generation Quality

| Metric               | With Reranking | No Reranking |
|----------------------|----------------|--------------|
| Correctness          | 55.86%         | 57.23%       |
| Relevance            | 73.02%         | 51.20%       |
| Groundedness*        | 58.23%         | —            |
| Retrieval Relevance  | 63.76%         | 66.87%       |
| Specificity          | 70.84%         | 58.43%       |
| Conciseness          | 77.11%         | 68.67%       |
| Completeness         | 71.39%         | 55.42%       |

\* Groundedness was re-checked on 79 questions with `openai/gpt-oss-120b` as judge, using the full retrieved context. The No Reranking column was evaluated on an earlier 166-question version of the dataset.

### Retrieval Quality (Recall@K)

| K  | Recall |
|----|--------|
| 5  | 85.01% |
| 7  | 88.83% |
| 10 | 90.74% |
| 20 | 92.92% |

**Key finding:** Cross-encoder reranking raises relevance from 51.20% to 73.02% and completeness from 55.42% to 71.39%, and improves specificity and conciseness, at a small cost in correctness (−1.4 points).

---

## Tech Stack

| Component        | Technology                                  |
|------------------|---------------------------------------------|
| LLM              | Groq `openai/gpt-oss-120b`                  |
| Embeddings       | `all-MiniLM-L6-v2` (sentence-transformers)  |
| Reranker         | `cross-encoder/ms-marco-MiniLM-L-6-v2`     |
| Vector Store     | ChromaDB (persistent, local)                |
| Framework        | LangChain                                   |
| UI               | Streamlit                                   |

---

## Project Structure

```
├── src/
│   ├── rag.py          # RAGPipeline + AdaptiveRAGPipeline
│   ├── reranker.py     # Cross-encoder reranking + adaptive K functions
│   ├── loader.py       # PDF loader (handles subdirs, deduplication)
│   ├── chunker.py      # Text chunking (size=1000, overlap=150)
│   ├── embeddings.py   # Sentence-transformer embedding model
│   ├── vectordb.py     # ChromaDB load/save
│   └── build_db.py     # One-time index builder
├── evaluation/
│   ├── evaluate.py               # Generate answers (vanilla RAG, 367 Q)
│   ├── evaluate_no_rerank.py     # Generate answers (no reranking)
│   ├── run_all.py                # Score vanilla answers (all 7 metrics)
│   ├── run_all_no_rerank.py      # Score no-rerank answers
│   ├── judge.py                  # Shared Groq LLM judge (cached, rate-limit aware)
│   ├── compare.py                # With vs without reranking, side by side
│   ├── compare_judges.py         # Agreement check between two judge models
│   ├── recall_at_k.py            # Recall@K evaluation
│   ├── tune_k.py                 # K-value tuning (top_k × retrieve_k grid)
│   └── tune_adaptive_k.py        # Adaptive K strategy comparison
├── app.py                        # Streamlit chatbot
├── chroma_db/                    # Pre-built vector store (110 PDFs)
├── tiet_rag_filtered.csv         # 367-question evaluation dataset
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/SarthakJagota/tiet-rag.git
cd tiet-rag

# 1. Create and activate virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies (or requirements-freeze.txt for exact pinned versions)
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Run the chatbot:
```bash
streamlit run app.py
```

The vector store (`chroma_db/`) is pre-built and included — no need to re-index. The source PDFs are not included in the repository; to rebuild the index, place them in `Data/` and run `python src/build_db.py`.

---

## Reproducing Evaluation

The scripts use a Groq model as judge (`JUDGE_MODEL`, default `openai/gpt-oss-20b`). Verdicts are cached in `evaluation/judge_cache/`, so a run stopped by rate limits resumes where it left off.

```bash
# 1. Generate answers
python evaluation/evaluate.py

# 2. Score with all 7 metrics
python evaluation/run_all.py

# 3. Recall@K
python evaluation/recall_at_k.py

# 4. K-value tuning
python evaluation/tune_k.py
```

---

## Dataset

- **367 questions** with ground-truth answers and source PDFs
- **110 policy PDFs** across 10 categories (admissions, curriculum, hostel, fees, research, etc.)
- Multi-hop questions: 1-hop, 2-hop, 3-hop
- Pipe-separated multi-source annotations for Recall@K evaluation
