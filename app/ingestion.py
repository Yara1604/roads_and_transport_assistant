import glob
import json
import re
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.types import DoclingDocument
from langchain_core.documents import Document
import pandas as pd


def load_documents(source_folder: str, processed_folder: str | None = None) -> list[dict]:

    pdf_paths = glob.glob(str(Path(source_folder) / "*.pdf"))

    documents = []

    processed_dir = Path(processed_folder) if processed_folder else Path(source_folder).parent.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    converter: DocumentConverter | None = None

    for document_id, pdf in enumerate(pdf_paths):
        pdf_path = Path(pdf)
        title = pdf_path.stem

        processed_path = processed_dir / f"{title}.json"

        try:
            if processed_path.exists():
                with processed_path.open("r", encoding="utf-8") as f:
                    doc_dict = json.load(f)
                doc = DoclingDocument.model_validate(doc_dict)
            else:
                if converter is None:
                    converter = DocumentConverter()

                doc = converter.convert(pdf_path).document
                with processed_path.open("w", encoding="utf-8") as f:
                    json.dump(doc.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
                print(f"Converted and saved: {processed_path.resolve()}")
        except Exception as e:
            print(f"Could not process {title}: {e}")
            continue

        documents.append(
            {
                "doc": doc,
                "document_id": document_id,
                "title": title,
                "source_file": str(pdf_path),
            }
        )

    return documents


def table_to_md(table, doc: DoclingDocument) -> str:
    table_df: pd.DataFrame = table.export_to_dataframe(
        doc=doc
    )  # table_df is the variable name, :pd is the type hint
    return table_df.to_markdown(index=False)


def chunk_documents(documents: list[dict], max_tokens: int = 512) -> list[Document]:

    chunker = HybridChunker(max_tokens=max_tokens, repeat_table_header=True)

    langchain_docs = []

    for document in documents:
        # Document level info
        doc = document["doc"]
        document_id = document["document_id"]
        title = document["title"]

        # Extract type
        if "No." in title:
            legislation_type = title.rsplit("No.", 1)[0].strip()
        else:
            legislation_type = None

        # Extract year

        year_match = re.search(r"\b(19|20)\d{2}\b", title)

        if year_match:
            year = int(year_match.group())
        else:
            year = None

        # Chunking
        chunk_iter = chunker.chunk(dl_doc=doc)

        for j, chunk in enumerate(chunk_iter):
            text = chunker.contextualize(chunk)

            chunk_id = f"{title}_{j}"

            # page number
            page = None
            if chunk.meta.doc_items:
                prov = chunk.meta.doc_items[0].prov
                if prov:
                    page = prov[0].page_no

            # Chroma metadata values must be scalar, so retain these fields as text.
            headings = " | ".join(chunk.meta.headings or [])
            captions = " | ".join(chunk.meta.captions or [])

            chunk_metadata = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "filename": title,
                "year": year,
                "type": legislation_type,
                "page": page,
                "headings": headings,
                "captions": captions,
                "source_file": document["source_file"],
                "primary_category": "Roads and Transport",
            }

            langchain_docs.append(Document(page_content=text, metadata=chunk_metadata))

    return langchain_docs
