# TIET Policy Assistant — RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system built over **110 TIET (Thapar Institute of Engineering & Technology) policy PDFs**, featuring cross-encoder reranking, adaptive K selection, agentic query decomposition, and a comprehensive 7-metric evaluation framework.

**[Live Demo](https://sarthakjagota-tiet-rag.streamlit.app)** · **[GitHub](https://github.com/SarthakJagota/tiet-rag)**

---

## Features

- **Vanilla RAG** — bi-encoder retrieval → cross-encoder reranking → LLM generation
- **Adaptive K** — dynamic chunk selection using score thresholds or knee detection
- **Agentic RAG** — query classification → multi-hop decomposition → synthesis
- **7-metric evaluation** over 367 ground-truth questions (LLM-as-judge)
- **Streamlit chatbot** with source attribution, session stats, and metrics dashboard

---

## Architecture

```
User Query
    │
    ▼
Query Classifier ──── simple ────► Embed Query
    │                                   │
  complex                          ChromaDB Similarity Search (retrieve_k=25)
    │                                   │
    ▼                              Cross-Encoder Reranking (top_k=8)
Decompose into                          │
2 Sub-questions                    LLM Generation (Groq llama-3.3-70b)
    │                                   │
    ▼                                   ▼
Retrieve + Answer               Answer + Sources
each sub-question
    │
    ▼
Synthesize final answer
```

---

## Evaluation Results

Evaluated on **367 questions** across 110 TIET policy PDFs using LLM-as-judge (binary 0/1 scoring).

### Generation Quality

| Metric               | Vanilla RAG | No Reranking | Agentic RAG |
|----------------------|-------------|--------------|-------------|
| Correctness          | 55.86%      | 57.23%       | 43.87%      |
| Relevance            | 73.02%      | 51.20%       | 59.13%      |
| Groundedness         | 13.62%      | 27.11%       | 5.45%       |
| Retrieval Relevance  | 63.76%      | 66.87%       | 49.59%      |
| Specificity          | 70.84%      | 58.43%       | 50.14%      |
| Conciseness          | 77.11%      | 68.67%       | 64.03%      |
| Completeness         | 71.39%      | 55.42%       | 55.59%      |

### Retrieval Quality (Recall@K)

| K  | Vanilla RAG | Agentic RAG |
|----|-------------|-------------|
| 5  | 83.33%      | 85.01%      |
| 10 | 87.50%      | 90.74%      |
| 20 | 90.83%      | 92.92%      |

**Key finding:** Agentic RAG consistently improves Recall@K (+3.24% at K=10) through multi-hop retrieval, but degrades generation quality — synthesis with a 7B parameter model loses precision. Vanilla RAG with reranking wins on most generation metrics.

---

## Tech Stack

| Component        | Technology                                  |
|------------------|---------------------------------------------|
| LLM              | Groq `llama-3.3-70b-versatile`              |
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
│   ├── agent.py        # AgenticRAGPipeline (classify → decompose → synthesize)
│   ├── reranker.py     # Cross-encoder reranking + adaptive K functions
│   ├── loader.py       # PDF loader (handles subdirs, deduplication)
│   ├── chunker.py      # Text chunking (size=1000, overlap=150)
│   ├── embeddings.py   # Sentence-transformer embedding model
│   ├── vectordb.py     # ChromaDB load/save
│   └── build_db.py     # One-time index builder
├── evaluation/
│   ├── evaluate.py               # Generate answers (vanilla RAG, 367 Q)
│   ├── evaluate_no_rerank.py     # Generate answers (no reranking)
│   ├── evaluate_agentic.py       # Generate answers (agentic RAG)
│   ├── run_all.py                # Score vanilla answers (all 7 metrics)
│   ├── run_all_agentic.py        # Score agentic answers
│   ├── recall_at_k.py            # Recall@K evaluation
│   ├── tune_k.py                 # K-value tuning (top_k × retrieve_k grid)
│   ├── tune_adaptive_k.py        # Adaptive K strategy comparison
│   └── compare_agentic.py        # Vanilla vs agentic side-by-side
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

The vector store (`chroma_db/`) is pre-built and included — no need to re-index.

---

## Reproducing Evaluation

```bash
# 1. Generate answers
python evaluation/evaluate.py

# 2. Score with all 7 metrics
python evaluation/run_all.py

# 3. Recall@K
python evaluation/recall_at_k.py

# 4. K-value tuning
python evaluation/tune_k.py

# 5. Agentic RAG comparison
python evaluation/evaluate_agentic.py
python evaluation/run_all_agentic.py
python evaluation/compare_agentic.py
```

---

## Dataset

- **367 questions** with ground-truth answers and source PDFs
- **110 policy PDFs** across 10 categories (admissions, curriculum, hostel, fees, research, etc.)
- Multi-hop questions: 1-hop, 2-hop, 3-hop
- Pipe-separated multi-source annotations for Recall@K evaluation
