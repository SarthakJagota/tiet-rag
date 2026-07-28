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

GROQ_MODEL = "llama-3.3-70b-versatile"

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


class RAGPipeline:
    def __init__(self, top_k=8, retrieve_k=25):
        self.top_k = top_k
        self.retrieve_k = retrieve_k
        self.embedding_model = get_embedding_model()
        self.vectordb = load_vectordb(self.embedding_model)
        self.reranker = get_reranker()
        self.llm = ChatGroq(model=GROQ_MODEL, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def retrieve(self, query):
        candidates = self.vectordb.similarity_search(query, k=self.retrieve_k)
        return rerank(self.reranker, query, candidates, self.top_k)

    def answer(self, query):
        docs = self.retrieve(query)
        context = "\n\n".join(
            f"[{d.metadata.get('source', 'unknown')} p.{d.metadata.get('page', '?')}] {d.page_content}"
            for d in docs
        )
        message = self.prompt.format(context=context, question=query)
        response = self.llm.invoke(message)
        return response.content, docs


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

class AdaptiveRAGPipeline:
    """
    RAG pipeline with dynamic K selection.

    mode="threshold" : keeps every chunk whose cross-encoder score >= ce_threshold
    mode="knee"      : cuts at the steepest score drop between consecutive chunks

    Both modes respect min_k / max_k bounds.
    retrieve_k is the candidate pool size fetched from ChromaDB before reranking.
    """

    def __init__(self, mode="threshold", ce_threshold=0.0,
                 min_k=3, max_k=15, retrieve_k=50):
        self.mode         = mode
        self.ce_threshold = ce_threshold
        self.min_k        = min_k
        self.max_k        = max_k
        self.retrieve_k   = retrieve_k

        self.embedding_model = get_embedding_model()
        self.vectordb        = load_vectordb(self.embedding_model)
        self.reranker        = get_reranker()
        self.llm             = ChatGroq(model=GROQ_MODEL, temperature=0)
        self.prompt          = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

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

    def answer(self, query):
        docs = self.retrieve(query)
        context = "\n\n".join(
            f"[{d.metadata.get('source', 'unknown')} p.{d.metadata.get('page', '?')}] {d.page_content}"
            for d in docs
        )
        message  = self.prompt.format(context=context, question=query)
        response = self.llm.invoke(message)
        return response.content, docs


def query_rag(question):
    pipeline = RAGPipeline()
    answer, docs = pipeline.answer(question)
    return {"answer": answer, "contexts": [doc.page_content for doc in docs]}


if __name__ == "__main__":
    main()
