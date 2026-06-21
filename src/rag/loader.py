import pdfplumber
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdfs(pdf_dir: str = "data/pdf") -> list:
    """Load all PDF files using pdfplumber for table-aware extraction."""
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDF files found in '{pdf_dir}'")
        return []

    documents = []
    for path in pdf_paths:
        docs = _load_single_pdf(path)
        documents.extend(docs)
        print(f"Loaded '{path.name}' — {len(docs)} page(s)")

    print(f"\nTotal pages: {len(documents)}")
    return documents


def _load_single_pdf(path: Path) -> list:
    """Extract each page as a Document, preserving table column structure."""
    documents = []

    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages):
            parts = []

            for table in page.extract_tables():
                formatted = _format_table(table)
                if formatted:
                    parts.append(formatted)

            plain = page.extract_text(x_tolerance=3, y_tolerance=3)
            if plain and plain.strip():
                parts.append(plain.strip())

            content = "\n\n".join(parts)
            if not content.strip():
                continue

            documents.append(Document(
                page_content=content,
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "file_type": path.suffix.lower(),
                    "file_size_kb": round(path.stat().st_size / 1024, 2),
                    "page": page_num,
                    "total_pages": len(pdf.pages),
                },
            ))

    return documents


def _format_table(table: list[list]) -> str:
    """Format a pdfplumber table as clearly labelled rows.

    Detects the header row and prefixes each data row with column names so
    the LLM can unambiguously match numbers to their column (e.g. 100g vs 1kg).
    """
    if not table:
        return ""

    rows = [[str(c).strip() if c else "" for c in row] for row in table]
    header = rows[0]
    lines = []

    for row in rows[1:]:
        if not any(row):
            continue
        pairs = [
            f"{h}: {v}" if h else v
            for h, v in zip(header, row)
            if v
        ]
        if pairs:
            lines.append(" | ".join(pairs))

    return "\n".join(lines)


def split_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} page(s) → {len(chunks)} chunk(s)  "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks
