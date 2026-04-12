import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def get_embeddings():
    """Returns the production BGE embedding model (runs locally, no cost)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_vector_store() -> Chroma:
    """Returns the persistent ChromaDB vector store."""
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
        collection_name="disaster_response"
    )

def add_documents_to_store(documents: list, batch_size: int = 500):
    """Ingests a list of LangChain Documents into ChromaDB in batches."""
    store = get_vector_store()
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        store.add_documents(batch)
        print(f"  Added batch {i // batch_size + 1} ({min(i + batch_size, total)}/{total})")

def clear_collection():
    """Wipes the entire collection for a clean repopulation."""
    store = get_vector_store()
    store.delete_collection()
    print("Collection cleared.")
