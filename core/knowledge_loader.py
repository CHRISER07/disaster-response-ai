import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_and_chunk_knowledge_base(directory: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Recursively finds all PDF survival manuals in the KB directory, loads them, 
    and applies research-grade text splitting (RecursiveCharacterTextSplitter) 
    to create semantically coherent chunks for the Vector DB.
    """
    all_chunks = []
    if not os.path.exists(directory):
        print(f"Warning: KB directory {directory} not found.")
        return all_chunks
        
    # Text splitter optimized for maintaining paragraph context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", r"(?<=\. )", " ", ""]
    )
        
    for file in os.listdir(directory):
        if file.endswith('.pdf'):
            file_path = os.path.join(directory, file)
            print(f"Processing Knowledge Base PDF: {file}")
            
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Add uniform metadata before splitting
                for doc in docs:
                    doc.metadata["modality"] = "knowledge_base"
                    doc.metadata["file_name"] = file
                    doc.metadata["source_type"] = "official_manual"
                    
                # Split pages into semantic chunks
                chunks = text_splitter.split_documents(docs)
                all_chunks.extend(chunks)
                print(f" -> Yielded {len(chunks)} semantic chunks from {file}.")
                
            except Exception as e:
                print(f"Failed to load {file}: {e}")
                
    return all_chunks

if __name__ == "__main__":
    docs = load_and_chunk_knowledge_base(os.path.join(os.path.dirname(__file__), "..", "data", "kb"))
    print(f"Total KB chunks ready for embedding: {len(docs)}")
