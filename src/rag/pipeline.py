from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag.embedder import Embedder
from rag.store import VectorStore


class RAG:
    """Retrieval-Augmented Generation: fetch relevant chunks then answer with an LLM."""

    SYSTEM_TEMPLATE = """\
You are a helpful fragrance price assistant. Use the context below to answer questions.

Rules:
1. For prices listed in the context, answer directly.
2. If a quantity is NOT listed (e.g. 10g, 50g, 500g) but 100g or 1kg price IS available, \
calculate proportionally. Example: 10g = 100g price ÷ 10. Always label it "(estimated)".
3. Only say "I don't have enough information" if the product itself is not in the context at all.

At the end of your answer add a "Sources:" line with the file name and page number.

Context:
{context}"""

    def __init__(self, vector_store: VectorStore, embedder: Embedder,
                 model: str = "gpt-5.4-nano", top_k: int = 10):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.llm = ChatOpenAI(model=model, temperature=0)

    def answer(self, question: str, history: list[dict] | None = None) -> str:
        """Retrieve context for question, then generate and return an answer.

        history: Gradio chat history — list of {"role": "user"|"assistant", "content": str}
        """
        hits = self.vector_store.query(question, self.embedder, top_k=self.top_k)

        context = "\n\n---\n\n".join(
            f"[Source: {h['metadata'].get('file_name', 'unknown')}  "
            f"page {h['metadata'].get('page', '?')}]\n{h['text']}"
            for h in hits
        )

        messages = [SystemMessage(content=self.SYSTEM_TEMPLATE.format(context=context))]

        for turn in (history or []):
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=question))

        return self.llm.invoke(messages).content
