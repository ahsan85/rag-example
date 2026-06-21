import uuid
import numpy as np
import chromadb
from pathlib import Path


class VectorStore:
    """Manages document embeddings in a persistent ChromaDB collection."""

    def __init__(self, persist_dir: str = "data/chroma", collection_name: str = "rag_docs"):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Collection '{collection_name}' ready — {self.collection.count()} doc(s) on disk")

    def reset(self) -> None:
        """Drop and recreate the collection (clears all stored documents)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Collection '{self.collection_name}' reset — 0 doc(s)")

    def add_documents(self, documents: list, embeddings: list | np.ndarray) -> None:
        """Insert documents + their embeddings into ChromaDB."""
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} document(s) but {len(embeddings)} embedding(s)."
            )

        ids, vecs, metas, texts = [], [], [], []

        for idx, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(str(uuid.uuid4()))
            vecs.append(emb.tolist() if isinstance(emb, np.ndarray) else list(emb))
            meta = {**doc.metadata, "chunk_index": idx, "content_length": len(doc.page_content)}
            meta = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
            metas.append(meta)
            texts.append(doc.page_content)

        self.collection.add(ids=ids, embeddings=vecs, metadatas=metas, documents=texts)
        print(f"Added {len(ids)} doc(s) — collection total: {self.collection.count()}")

    def query(self, query_text: str, embedder, top_k: int = 5) -> list[dict]:
        """Convert query_text to an embedding, run cosine search, return top_k results."""
        query_vector = embedder.embed_query(query_text)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for rank, (text, meta, distance) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
            start=1,
        ):
            hits.append({
                "rank": rank,
                "score": round(1 - distance, 4),
                "text": text,
                "metadata": meta,
            })

        return hits
