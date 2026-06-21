from langchain_openai import OpenAIEmbeddings


class Embedder:
    """Wraps OpenAI text-embedding-3-small for RAG indexing and retrieval."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._embeddings = OpenAIEmbeddings(model=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (used when building the vector store)."""
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (used at retrieval time)."""
        return self._embeddings.embed_query(text)

    def as_langchain(self) -> OpenAIEmbeddings:
        """Return the raw LangChain embeddings object (for vector store constructors)."""
        return self._embeddings
