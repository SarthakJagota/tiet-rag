import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from embeddings import get_embedding_model
from vectordb import load_vectordb
from reranker import get_reranker, rerank, rerank_threshold, rerank_knee

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

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

CONDENSE_PROMPT_TEMPLATE = """Given the conversation history and a follow-up question, rewrite the follow-up question into an independent, standalone search query that includes all necessary context (such as program names, departments, policy topics).
Do NOT answer the question. Return ONLY the standalone query. If the question is already standalone, return it as is.

Conversation History:
{chat_history}

Follow-up Question: {question}
Standalone Query:"""


class BaseRAGPipeline:
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vectordb = load_vectordb(self.embedding_model)
        self.reranker = get_reranker()
        self.llm = ChatGroq(model=GROQ_MODEL, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.condense_prompt = ChatPromptTemplate.from_template(CONDENSE_PROMPT_TEMPLATE)

    def condense_query(self, query, chat_history=None):
        if not chat_history:
            return query
        history_lines = []
        for msg in chat_history[-6:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            if content:
                history_lines.append(f"{role}: {content[:250]}")
        if not history_lines:
            return query
        try:
            formatted = self.condense_prompt.format(
                chat_history="\n".join(history_lines),
                question=query,
            )
            condensed = self.llm.invoke(formatted).content.strip()
            return condensed if condensed else query
        except Exception:
            return query

    def format_context(self, docs):
        return "\n\n".join(
            f"[{d.metadata.get('source', 'unknown')} p.{d.metadata.get('page', '?')}] {d.page_content}"
            for d in docs
        )

    def answer(self, query, chat_history=None):
        search_query = self.condense_query(query, chat_history)
        docs = self.retrieve(search_query)
        context = self.format_context(docs)
        message = self.prompt.format(context=context, question=query)
        response = self.llm.invoke(message)
        return response.content, docs

    def stream_answer(self, query, chat_history=None):
        search_query = self.condense_query(query, chat_history)
        docs = self.retrieve(search_query)
        context = self.format_context(docs)
        message = self.prompt.format(context=context, question=query)
        return self.llm.stream(message), docs, search_query


class RAGPipeline(BaseRAGPipeline):
    def __init__(self, top_k=8, retrieve_k=25):
        super().__init__()
        self.top_k = top_k
        self.retrieve_k = retrieve_k

    def retrieve(self, query):
        candidates = self.vectordb.similarity_search(query, k=self.retrieve_k)
        return rerank(self.reranker, query, candidates, self.top_k)


class AdaptiveRAGPipeline(BaseRAGPipeline):
    """
    RAG pipeline with dynamic K selection.

    mode="threshold" : keeps every chunk whose cross-encoder score >= ce_threshold
    mode="knee"      : cuts at the steepest score drop between consecutive chunks

    Both modes respect min_k / max_k bounds.
    retrieve_k is the candidate pool size fetched from ChromaDB before reranking.
    """

    def __init__(self, mode="threshold", ce_threshold=0.0,
                 min_k=3, max_k=15, retrieve_k=50):
        super().__init__()
        self.mode         = mode
        self.ce_threshold = ce_threshold
        self.min_k        = min_k
        self.max_k        = max_k
        self.retrieve_k   = retrieve_k

    def retrieve(self, query):
        candidates = self.vectordb.similarity_search(query, k=self.retrieve_k)
        if self.mode == "threshold":
            return rerank_threshold(
                self.reranker, query, candidates,
                ce_threshold=self.ce_threshold,
                min_k=self.min_k, max_k=self.max_k,
            )
        else:  # knee
            return rerank_knee(
                self.reranker, query, candidates,
                min_k=self.min_k, max_k=self.max_k,
            )


_shared_pipeline = None


def query_rag(question):
    global _shared_pipeline
    if _shared_pipeline is None:
        _shared_pipeline = RAGPipeline()
    answer, docs = _shared_pipeline.answer(question)
    return {"answer": answer, "contexts": [doc.page_content for doc in docs]}


def main():
    pipeline = RAGPipeline()
    print("RAG ready. Type your question (or 'exit' to quit).")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        answer, docs = pipeline.answer(query)
        print(f"\nAnswer:\n{answer}")
        print("\nSources:")
        for d in docs:
            print(f" - {d.metadata.get('source')} (page {d.metadata.get('page')})")


if __name__ == "__main__":
    main()
