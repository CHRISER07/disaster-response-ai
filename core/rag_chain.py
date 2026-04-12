import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from core.vector_store import get_vector_store
from dotenv import load_dotenv

load_dotenv()

def get_retriever(allowed_modalities=None):
    """
    Returns the vector store retriever configured for research-grade accuracy.
    Uses similarity matching and fetches top context chunks.
    If allowed_modalities is provided, it filters the retrieval for ablation studies.
    """
    store = get_vector_store()
    
    search_kwargs = {"k": 7, "lambda_mult": 0.7}
    
    if allowed_modalities:
        search_kwargs["filter"] = {"modality": {"$in": allowed_modalities}}
        
    return store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

class SimpleRAGChain:
    def __init__(self, llm, prompt, retriever):
        self.llm = llm
        self.prompt = prompt
        self.retriever = retriever
        
    def invoke(self, inputs: dict) -> dict:
        query = inputs["input"]
        docs = self.retriever.invoke(query)
        context_str = "\n\n".join(doc.page_content for doc in docs)
        
        prompt_val = self.prompt.invoke({"context": context_str, "input": query})
        answer = self.llm.invoke(prompt_val).content
        
        return {"answer": answer, "context": docs}

def get_rag_chain(allowed_modalities=None):
    """
    Constructs the overall LangChain RAG pipeline.
    Synthesizes answers from multi-modal sources (Text/Tweets, IoT, Drone, KB PDFs).
    Uses the FREE Groq API with Llama-3.3-70B for synthesis.
    """
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)
    
    # Academic/Research grade prompt encouraging source citations and temporal awareness
    system_prompt = (
        "You are an advanced Disaster Response AI Assistant.\n"
        "You have access to a real-time Retrieval-Augmented Generation (RAG) system containing:\n"
        "1. Static official knowledge bases (protocols, FEMA guides).\n"
        "2. Dynamic streams from social media text reports (CrisisLex).\n"
        "3. IoT Telemetry data (Water levels).\n"
        "4. Drone imagery descriptions.\n\n"
        "Use the provided context to answer the user's inquiry.\n"
        "CRITICAL RULES:\n"
        "- Base your answer ONLY on the provided context.\n"
        "- If the context contains conflicting reports, mention the conflict and note the timestamp of the latest information.\n"
        "- Explicitly cite the source metadata of the context you used (e.g., 'According to the KB manual...', 'Based on a CrisisLex tweet at time...', 'Drone observation reports...').\n"
        "- If you do not know the answer based on the context, say so.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    return SimpleRAGChain(llm, prompt, get_retriever(allowed_modalities))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query the Disaster Response RAG.")
    parser.add_argument("query", type=str, help="The query string.")
    args = parser.parse_args()
    
    try:
        chain = get_rag_chain()
        print(f"\nQuerying: '{args.query}'...\n")
        response = chain.invoke({"input": args.query})
        print("=== ANSWER ===")
        print(response["answer"])
        print("\n=== SOURCES CITED ===")
        for i, doc in enumerate(response["context"]):
            print(f"{i+1}. [{doc.metadata.get('modality', 'unknown')}] {doc.page_content[:150]}...")
    except Exception as e:
        print(f"\nError: {e}\nDid you forget to add GROQ_API_KEY to your .env file?")
