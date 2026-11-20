"""
TIET RAG Chatbot — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import sys, os, time
from dotenv import load_dotenv

load_dotenv()
# On Streamlit Cloud, secrets override .env
if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.rag import RAGPipeline, AdaptiveRAGPipeline, GROQ_MODEL

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TIET Policy Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0F172A; }

/* Header */
.main-header {
    background: linear-gradient(135deg, #1E3A8A 0%, #1E293B 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #2563EB33;
}
.main-header h1 { color: #F1F5F9; margin: 0; font-size: 1.8rem; }
.main-header p  { color: #94A3B8; margin: 0.3rem 0 0 0; font-size: 0.9rem; }

/* Suggestion buttons */
.suggestion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin: 1rem 0;
}
.suggestion-btn {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    color: #CBD5E1;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
    text-align: left;
    width: 100%;
}
.suggestion-btn:hover {
    background: #2563EB22;
    border-color: #2563EB;
    color: #F1F5F9;
}

/* Source cards */
.source-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-left: 3px solid #2563EB;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    color: #94A3B8;
    font-family: monospace;
}

/* Stats row */
.stats-row {
    display: flex;
    gap: 1rem;
    margin: 0.5rem 0;
}
.stat-chip {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    color: #64748B;
}

/* Mode badge */
.badge-simple  { background:#15803D22; color:#4ADE80; border:1px solid #15803D; border-radius:20px; padding:0.2rem 0.8rem; font-size:0.75rem; }
.badge-complex { background:#1D4ED822; color:#60A5FA; border:1px solid #1D4ED8; border-radius:20px; padding:0.2rem 0.8rem; font-size:0.75rem; }

/* Chat input styling */
.stChatInput > div { background: #1E293B !important; border-color: #334155 !important; border-radius: 12px !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0F172A; border-right: 1px solid #1E293B; }

/* Metrics cards */
.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-value { font-size: 1.8rem; font-weight: bold; color: #60A5FA; }
.metric-label { font-size: 0.8rem; color: #64748B; margin-top: 0.2rem; }

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load pipelines ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading RAG pipeline...")
def load_pipeline(strategy="Vanilla (Fixed Top-8)", model=GROQ_MODEL):
    if strategy == "Adaptive (Knee Detection)":
        return AdaptiveRAGPipeline(mode="knee", min_k=3, max_k=15, retrieve_k=50)
    elif strategy == "Adaptive (Score Threshold)":
        return AdaptiveRAGPipeline(mode="threshold", ce_threshold=0.0, min_k=3, max_k=15, retrieve_k=50)
    else:
        return RAGPipeline(top_k=8, retrieve_k=25)

# ── Suggested questions ────────────────────────────────────────────────────────
SUGGESTIONS = [
    "What is the tuition fee for MCA program?",
    "What are the hostel rules for room swapping?",
    "What is the discipline policy at TIET?",
    "How to apply for MTech admission?",
    "What scholarships are available at TIET?",
    "What are the library timings and facilities?",
]

# ── Session state ──────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "total_queries" not in st.session_state: st.session_state.total_queries = 0
if "total_time"    not in st.session_state: st.session_state.total_time    = 0.0
if "pending_query" not in st.session_state: st.session_state.pending_query = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 TIET Assistant")
    st.caption("AI-powered policy & information chatbot")
    st.divider()

    st.markdown("### ⚙️ Settings")
    rag_strategy = st.selectbox(
        "🎯 Retrieval Strategy",
        options=[
            "Vanilla (Fixed Top-8)",
            "Adaptive (Knee Detection)",
            "Adaptive (Score Threshold)",
        ],
        index=0,
        help="Vanilla uses fixed top-8 cross-encoder reranking. Adaptive modes dynamically choose chunk count.",
    )
    enable_memory = st.toggle("🧠 Conversational memory", value=True, help="Reformulates follow-up queries using prior chat context.")
    show_sources  = st.toggle("📄 Show sources",          value=True)
    show_chunks   = st.toggle("🔍 Show retrieved chunks", value=False)

    st.divider()
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        avg = (st.session_state.total_time / st.session_state.total_queries
               if st.session_state.total_queries > 0 else 0)
        st.metric("Avg Time", f"{avg:.1f}s")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.total_queries = 0
        st.session_state.total_time    = 0.0
        st.rerun()

    st.divider()
    st.markdown("### 🛠️ Tech Stack")
    st.caption("🔹 LangChain + ChromaDB")
    st.caption("🔹 Groq gpt-oss-120b")
    st.caption("🔹 Cross-encoder reranking")
    st.caption("🔹 110 TIET policy PDFs")
    st.caption("🔹 367-question eval dataset")

    st.divider()
    st.markdown("### 📈 Model Performance")
    st.caption("Correctness : 55.86%")
    st.caption("Relevance   : 73.02%")
    st.caption("Recall@10   : 90.74%")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_chat, tab_metrics, tab_about = st.tabs(["💬 Chat", "📊 Metrics", "ℹ️ About"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 TIET Policy Assistant</h1>
        <p>Ask anything about TIET admissions, fees, programs, hostel, regulations and more.</p>
    </div>
    """, unsafe_allow_html=True)

    # Suggested questions (only when chat is empty)
    if not st.session_state.messages:
        st.markdown("#### 💡 Try asking:")
        cols = st.columns(2)
        for idx, suggestion in enumerate(SUGGESTIONS):
            with cols[idx % 2]:
                if st.button(suggestion, key=f"sug_{idx}", use_container_width=True):
                    st.session_state.pending_query = suggestion
                    st.rerun()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎓"):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                # Stats row
                stats_html = f"""
                <div class="stats-row">
                    <span class="stat-chip">⏱️ {msg.get('time', 0):.1f}s</span>
                    <span class="stat-chip">📚 {msg.get('num_sources', 0)} sources</span>
                    <span class="stat-chip">📝 {len(msg['content'].split())} words</span>
                </div>"""
                st.markdown(stats_html, unsafe_allow_html=True)

                # Sources
                if show_sources and msg.get("sources"):
                    with st.expander(f"📄 Sources ({len(msg['sources'])})", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(f'<div class="source-card">📋 {src}</div>', unsafe_allow_html=True)

                # Chunks
                if show_chunks and msg.get("chunks"):
                    with st.expander(f"🔍 Retrieved chunks ({len(msg['chunks'])})", expanded=False):
                        for i, chunk in enumerate(msg["chunks"]):
                            st.markdown(f"**Chunk {i+1}** — `{chunk['source']}` p.{chunk['page']}")
                            st.text_area("", chunk["text"][:600], height=100, key=f"chunk_{id(msg)}_{i}", disabled=True)

    # Process pending query (from suggestion buttons)
    query_to_run = None
    if st.session_state.pending_query:
        query_to_run = st.session_state.pending_query
        st.session_state.pending_query = None

    # Chat input
    if user_input := st.chat_input("Ask about TIET policies, admissions, fees, programs..."):
        query_to_run = user_input

    # Generate answer
    if query_to_run:
        st.session_state.messages.append({"role": "user", "content": query_to_run})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(query_to_run)

        with st.chat_message("assistant", avatar="🎓"):
            t0 = time.time()
            search_query_used = query_to_run
            try:
                pipeline = load_pipeline(strategy=rag_strategy)
                history = st.session_state.messages[:-1] if enable_memory else None
                with st.spinner("🔍 Searching TIET documents..."):
                    stream, docs, search_query_used = pipeline.stream_answer(query_to_run, chat_history=history)

                if enable_memory and search_query_used != query_to_run:
                    st.caption(f"🔍 *Standalone query:* `{search_query_used}`")

                answer = st.write_stream(stream)
                elapsed = time.time() - t0
                sources = sorted(set(d.metadata.get("source", "?") for d in docs))
                chunks = [{"source": d.metadata.get("source", "?"),
                           "page":   d.metadata.get("page", "?"),
                           "text":   d.page_content} for d in docs]

            except Exception as e:
                err = str(e)
                if "429" in err or "rate limit" in err.lower():
                    msg = "⏳ The demo has used up its free daily LLM quota. Please try again in a little while."
                else:
                    msg = "❌ Something went wrong while generating the answer. Please try again."
                print(f"Answer generation failed: {err}")  # full error goes to the server log
                elapsed, answer, sources, chunks = time.time()-t0, msg, [], []
                st.markdown(answer)

            stats_html = f"""
            <div class="stats-row">
                <span class="stat-chip">⏱️ {elapsed:.1f}s</span>
                <span class="stat-chip">📚 {len(sources)} sources</span>
                <span class="stat-chip">📝 {len(answer.split())} words</span>
                <span class="stat-chip">🎯 {rag_strategy.split()[0]}</span>
            </div>"""
            st.markdown(stats_html, unsafe_allow_html=True)

            if show_sources and sources:
                with st.expander(f"📄 Sources ({len(sources)})", expanded=True):
                    for src in sources:
                        st.markdown(f'<div class="source-card">📋 {src}</div>', unsafe_allow_html=True)

            if show_chunks and chunks:
                with st.expander(f"🔍 Retrieved chunks ({len(chunks)})", expanded=False):
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Chunk {i+1}** — `{chunk['source']}` p.{chunk['page']}")
                        st.text_area("", chunk["text"][:600], height=100, key=f"new_chunk_{i}", disabled=True)

        st.session_state.messages.append({
            "role":        "assistant",
            "content":     answer,
            "sources":     sources,
            "chunks":      chunks,
            "time":        elapsed,
            "num_sources": len(sources),
        })
        st.session_state.total_queries += 1
        st.session_state.total_time    += elapsed
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — METRICS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("## 📊 Evaluation Results")
    st.caption("Evaluated on 367 TIET policy questions using LLM-as-judge (llama3.2; groundedness re-checked with gpt-oss-120b)")

    import pandas as pd, os
    BASE = os.path.join(os.path.dirname(__file__), "evaluation")

    def load_pct(fname, col):
        try:
            return pd.read_csv(os.path.join(BASE, fname))[col].mean() * 100
        except:
            return None

    # Metric cards row
    metrics = [
        ("Correctness",         load_pct("correctness_scores.csv",        "correctness"),        "✅"),
        ("Relevance",           load_pct("relevance_scores.csv",           "relevance"),          "🎯"),
        ("Groundedness*",       load_pct("groundedness_recheck.csv",       "groundedness"),       "⚓"),
        ("Retrieval Relevance", load_pct("retrieval_relevance_scores.csv", "retrieval_relevance"),"🔍"),
    ]

    cols = st.columns(4)
    for col, (label, val, icon) in zip(cols, metrics):
        with col:
            if val is not None:
                color = "#4ADE80" if val >= 60 else ("#FACC15" if val >= 40 else "#F87171")
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.5rem">{icon}</div>
                    <div class="metric-value" style="color:{color}">{val:.1f}%</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.5rem">{icon}</div>
                    <div class="metric-value" style="color:#475569">N/A</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

    st.caption("*Groundedness: 79-question re-check judged by gpt-oss-120b with the full retrieved context.")
    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Generation Quality")
        try:
            gq = pd.read_csv(os.path.join(BASE, "generation_quality_scores.csv"))
            gq_data = {
                "Specificity":  gq["specificity"].mean()  * 100,
                "Conciseness":  gq["conciseness"].mean()  * 100,
                "Completeness": gq["completeness"].mean() * 100,
            }
            for k, v in gq_data.items():
                st.markdown(f"**{k}**")
                st.progress(int(v), text=f"{v:.1f}%")
        except:
            st.info("Run evaluation/run_all.py first")

    with col_right:
        st.markdown("### Recall @ K")
        try:
            rk = pd.read_csv(os.path.join(BASE, "recall_at_k_results.csv"))
            n  = len(rk)
            for k in [5, 7, 10, 20]:
                col_name = f"hit@{k}"
                if col_name in rk.columns:
                    v = rk[col_name].sum() / n * 100
                    st.markdown(f"**R@{k}**")
                    st.progress(int(v), text=f"{v:.1f}%")
        except:
            st.info("Run evaluation/recall_at_k.py first")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("## ℹ️ About This Project")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
### 🎯 Project Goal
Build a production-ready RAG system for TIET (Thapar Institute of Engineering & Technology)
that answers questions about university policies, admissions, fees, and regulations.

### 🏗️ Architecture
```
User Query
    ↓
Embed query
    ↓
ChromaDB similarity search
    ↓
Cross-encoder reranking
    ↓
LLM Generation
(Groq gpt-oss-120b)
    ↓
Answer + Sources
```
        """)

    with col2:
        st.markdown("""
### 🛠️ Tech Stack
| Component | Technology |
|---|---|
| LLM | Groq gpt-oss-120b |
| Eval judge | llama3.2 (groundedness: gpt-oss-120b) |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Framework | LangChain |
| UI | Streamlit |

### 📁 Dataset
- **110 PDFs** across 10 categories
- **367 questions** with ground truth answers
- Multi-hop questions (1-hop, 2-hop, 3-hop)
- Categories: Admissions, Curriculum, Hostel, Faculty, Research...

### 📊 Key Findings
- Retrieval finds a correct source in the top 10 for **90.7%** of questions
- Reranked pipeline: **73.0% relevance**, **71.4% completeness**
- Adaptive K selection → better retrieval for complex queries
- Best config: top_k=8, retrieve_k=25 with cross-encoder reranking
        """)
