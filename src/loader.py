import os
import pdfplumber
from langchain_core.documents import Document


def load_pdf(pdf_path):

    documents = []
    source = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:

        for page_num, page in enumerate(pdf.pages):

            text = page.extract_text()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": source, "page": page_num + 1}
                    )
                )

    return documents


def load_pdfs_from_dir(data_dir):
    """Recursively load every PDF under data_dir into a single list of Documents.
    Skips .ipynb_checkpoints folders and skips duplicate filenames."""

    documents = []
    seen = set()  # track basenames already loaded

    for root, dirs, files in os.walk(data_dir):
        # Skip Jupyter checkpoint directories in-place so os.walk won't descend
        dirs[:] = [d for d in dirs if d != ".ipynb_checkpoints"]

        for filename in files:
            if not filename.lower().endswith(".pdf"):
                continue
            if filename in seen:
                print(f"  Skipping duplicate: {filename}")
                continue
            seen.add(filename)
            pdf_path = os.path.join(root, filename)
            try:
                docs = load_pdf(pdf_path)
                documents.extend(docs)
                if docs:
                    print(f"  Loaded {filename} ({len(docs)} pages)")
                else:
                    print(f"  Warning: {filename} has 0 extractable text pages (scanned/empty)")
            except Exception as e:
                print(f"  Failed to load {pdf_path}: {e}")

    return documents