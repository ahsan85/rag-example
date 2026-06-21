"""Re-build the ChromaDB index from scratch using the updated PDF loader."""
import os
from dotenv import load_dotenv

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found"

from rag.loader import load_pdfs, split_documents
from rag.embedder import Embedder
from rag.store import VectorStore

docs = load_pdfs("data/pdf")
chunks = split_documents(docs)

embedder = Embedder()
vs = VectorStore(persist_dir="data/chroma")

print("Resetting collection...")
vs.reset()

print("Embedding and indexing chunks...")
embeddings = embedder.embed_documents([c.page_content for c in chunks])
vs.add_documents(chunks, embeddings)

print("\nDone! Run `uv run python main.py` to start the UI.")
