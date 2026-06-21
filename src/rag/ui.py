import os
import gradio as gr
from dotenv import load_dotenv

from rag.embedder import Embedder
from rag.store import VectorStore
from rag.pipeline import RAG

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), (
    "OPENAI_API_KEY not found — copy .env.example → .env and fill in your key"
)

embedder = Embedder()
vs = VectorStore(persist_dir="data/chroma")
rag = RAG(vs, embedder)


def chat(message: str, history: list) -> str:
    return rag.answer(message, history=history)


def launch():
    gr.ChatInterface(
        fn=chat,
        title="Fragrance RAG",
        description="Ask questions about fragrance prices and invoices.",
        examples=[
            "What is the price of 1 Million perfume?",
            "Show me all Dior fragrances",
            "What is the cheapest perfume per kg?",
        ],
    ).launch()
