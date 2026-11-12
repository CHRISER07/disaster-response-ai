import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_and_chunk_knowledge_base(
    directory: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[Document]:
    """
    Recursively finds all PDF manuals in the KB directory, loads them,
    and applies RecursiveCharacterTextSplitter to create semantically
    coherent chunks for embedding into the vector store.
    """
    all_chunks = []

    if not os.path.exists(directory):
        print(f"  WARNING: KB directory not found: {directory}")
        return all_chunks

    # Using paragraph  sentence  word  char hierarchy instead.
    # Using paragraph  sentence  word  char hierarchy instead.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for file in sorted(os.listdir(directory)):
        if not file.endswith(".pdf"):
            continue

        file_path = os.path.join(directory, file)
        print(f"  Processing: {file}")

        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Attach uniform metadata before splitting
            for doc in docs:
                doc.metadata["modality"]      = "knowledge_base"
                doc.metadata["file_name"]     = file
                doc.metadata["source_type"]   = "official_manual"
                doc.metadata["source"]        = file.replace(".pdf", "")
                # Explicit temporal tag so kb_search_tool era-detection works correctly.
                # Without this, KB chunks have no timestamp  mislabeled [LIVE/RECENT].
                doc.metadata["temporal_tag"]  = "[KB \u2014 FEMA/CDC Protocol]"
                doc.metadata["timestamp"]     = ""   # intentionally empty - not a time-series doc

            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"    -> {len(chunks)} chunks from {file}")

        except Exception as e:
            print(f"  ERROR loading {file}: {e}")

    return all_chunks


if __name__ == "__main__":
    docs = load_and_chunk_knowledge_base(
        os.path.join(os.path.dirname(__file__), "..", "data", "kb")
    )
    print(f"\nTotal KB chunks ready for embedding: {len(docs)}")
