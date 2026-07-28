"""
Agentic RAG Pipeline for TIET policy documents.

Flow:
  1. Classify query  -> simple or complex
  2. If simple  -> delegate to RAGPipeline (single retrieval)
  3. If complex -> decompose into sub-questions
                   retrieve + answer each sub-question independently
                   synthesize sub-answers into final answer

Same interface as RAGPipeline: .answer(query) -> (answer, docs)
"""

import os
import sys
import re

sys.path.append(os.path.dirname(__file__))

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from embeddings import get_embedding_model
from vectordb import load_vectordb
from reranker import get_reranker, rerank

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_SUB_QUESTIONS = 2

# ── Prompts ────────────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """You are analyzing a question about TIET (Thapar Institute of Engineering & Technology).

Question: {question}

A question is COMPLEX only if it EXPLICITLY asks to:
- Compare two different programs, policies, or documents (uses words like "compare", "differ", "vs", "both")
- Combine facts from two clearly separate topics (e.g., hostel rules AND exam rules)
- Answer requires looking up multiple unrelated documents

A question is SIMPLE if:
- It asks about one specific topic, fee, rule, or program
- It can be answered from one section of one document
- It asks "what is", "how many", "when", "who", "list the" about one thing

When in doubt, return simple.

Return ONLY one word — simple or complex:"""


DECOMPOSE_PROMPT = """The following question requires information from multiple documents.
Break it into exactly 2 focused sub-questions (no more).
Each sub-question must be short, specific, and answerable from a single document.

Original Question: {question}

Return ONLY a numbered list of 2 sub-questions. No explanations, no intro text.
1.
2."""


SUB_ANSWER_PROMPT = """You are an assistant for TIET (Thapar Institute of Engineering & Technology).
Answer using ONLY the information in the context below.

Rules:
- Give a complete, accurate answer with all relevant details.
- Copy names, numbers, dates, and policy terms exactly as written.
- Do not add information not in the context.
- If the answer is not found, respond with exactly: I don't know.

Context:
{context}

Question:
{question}

Answer:"""


SYNTHESIZE_PROMPT = """You are an assistant for TIET (Thapar Institute of Engineering & Technology).
Write a single combined answer to the original question using ONLY the sub-answers provided.

Original Question: {question}

Sub-answers:
{sub_qa}

Rules:
- Use only the information in the sub-answers above. Do not add anything else.
- Keep names, numbers, and dates exactly as they appear.
- If a sub-answer says "I don't know", skip it.

Answer:"""


# ── Pipeline ───────────────────────────────────────────────────────────────────

class AgenticRAGPipeline:
    """
    Agentic RAG pipeline with query classification and decomposition.

    Parameters
    ----------
    top_k : int
        Number of chunks to keep after reranking per retrieval call.
    retrieve_k : int
        Candidate pool fetched from ChromaDB before reranking.
    max_sub_questions : int
        Maximum sub-questions to decompose a complex query into.
    verbose : bool
        Print intermediate steps (classification, sub-questions, sub-answers).
    """

    def __init__(self, top_k=8, retrieve_k=25,
                 max_sub_questions=MAX_SUB_QUESTIONS, verbose=True):
        self.top_k             = top_k
        self.retrieve_k        = retrieve_k
        self.max_sub_questions = max_sub_questions
        self.verbose           = verbose

        self.embedding_model = get_embedding_model()
        self.vectordb        = load_vectordb(self.embedding_model)
        self.reranker        = get_reranker()
        self.llm             = ChatGroq(model=GROQ_MODEL, temperature=0)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _llm(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content.strip()

    def _retrieve(self, query: str):
        candidates = self.vectordb.similarity_search(query, k=self.retrieve_k)
        return rerank(self.reranker, query, candidates, self.top_k)

    def _generate(self, question: str, docs) -> str:
        context = "\n\n".join(
            f"[{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}] {d.page_content}"
            for d in docs
        )
        return self._llm(SUB_ANSWER_PROMPT.format(context=context, question=question))

    # ── Step 1: Classify ───────────────────────────────────────────────────────

    def classify(self, query: str) -> str:
        """Returns 'simple' or 'complex'."""
        response = self._llm(CLASSIFY_PROMPT.format(question=query)).lower()
        if "complex" in response:
            return "complex"
        return "simple"

    # ── Step 2: Decompose ──────────────────────────────────────────────────────

    def decompose(self, query: str) -> list:
        """Break a complex query into sub-questions."""
        prompt   = DECOMPOSE_PROMPT.format(question=query, n=self.max_sub_questions)
        response = self._llm(prompt)

        sub_questions = []
        for line in response.splitlines():
            line = line.strip()
            # Match lines like "1. question text"
            m = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if m:
                sq = m.group(1).strip()
                if sq:
                    sub_questions.append(sq)

        # Fallback: if parsing fails, use the original query
        if not sub_questions:
            sub_questions = [query]

        return sub_questions[:self.max_sub_questions]

    # ── Step 3: Answer each sub-question ──────────────────────────────────────

    def answer_sub(self, sub_question: str):
        """Retrieve + generate answer for a single sub-question."""
        docs   = self._retrieve(sub_question)
        answer = self._generate(sub_question, docs)
        return answer, docs

    # ── Step 4: Synthesize ─────────────────────────────────────────────────────

    def synthesize(self, original_query: str, sub_qa: list) -> str:
        """Combine sub-question answers into one final answer."""
        sub_qa_text = "\n\n".join(
            f"Q{i+1}: {sq}\nA{i+1}: {ans}"
            for i, (sq, ans) in enumerate(sub_qa)
        )
        return self._llm(SYNTHESIZE_PROMPT.format(
            question=original_query,
            sub_qa=sub_qa_text,
        ))

    # ── Main entry point ───────────────────────────────────────────────────────

    def answer(self, query: str):
        """
        Answer a query using the agentic pipeline.
        Returns (answer, docs) — same interface as RAGPipeline.
        """
        # Step 1: Classify
        query_type = self.classify(query)
        if self.verbose:
            print(f"  [agent] type={query_type}")

        # Simple query — single retrieval, same as vanilla RAG
        if query_type == "simple":
            docs   = self._retrieve(query)
            answer = self._generate(query, docs)
            return answer, docs

        # Complex query — decompose, retrieve per sub-question, synthesize
        sub_questions = self.decompose(query)
        if self.verbose:
            print(f"  [agent] sub-questions ({len(sub_questions)}):")
            for i, sq in enumerate(sub_questions):
                print(f"    {i+1}. {sq}")

        sub_qa   = []
        all_docs = []
        seen_ids = set()

        for sq in sub_questions:
            ans, docs = self.answer_sub(sq)
            sub_qa.append((sq, ans))
            if self.verbose:
                print(f"  [agent] sub-answer: {ans[:80]}...")

            # Deduplicate docs by content
            for d in docs:
                doc_id = (d.metadata.get("source"), d.metadata.get("page"))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(d)

        # Step 4: Synthesize
        final_answer = self.synthesize(query, sub_qa)
        if self.verbose:
            print(f"  [agent] synthesized answer ready.")

        return final_answer, all_docs


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    pipeline = AgenticRAGPipeline(verbose=True)
    print("Agentic RAG ready. Type your question (or 'exit' to quit).")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        answer, docs = pipeline.answer(query)
        print(f"\nAnswer:\n{answer}")
        print(f"\nSources ({len(docs)} chunks):")
        seen = set()
        for d in docs:
            src = d.metadata.get("source", "?")
            if src not in seen:
                seen.add(src)
                print(f"  - {src}")


if __name__ == "__main__":
    main()
