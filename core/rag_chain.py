"""
core/rag_chain.py

Legacy SimpleRAGChain — kept for the IEEE evaluation harness (run_evals.py).
NOT used by the live dashboard (which uses the LangGraph agent).

Fix log:
  - FIXED: MMR search_kwargs — added fetch_k=20 (MMR ignores k= at retriever level)
  - FIXED: allowed_modalities filter normalized to lowercase to match stored metadata
  - FIXED: CRLF → LF line endings
"""
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from core.vector_store import get_vector_store
from dotenv import load_dotenv

load_dotenv()


def get_retriever(allowed_modalities: list | None = None):
    """
    Returns the vector store MMR retriever configured for research-grade accuracy.
    Uses Maximal Marginal Relevance (MMR) to reduce redundancy in retrieved chunks.

    Args:
        allowed_modalities: Optional list of modality strings to filter by.
            Valid values: "text", "knowledge_base", "telemetry", "imagery_description"
            (must be lowercase — stored metadata uses lowercase)
    """
    store = get_vector_store()

    # FIX: MMR uses fetch_k (candidates to fetch) then selects k from them.
    # lambda_mult=0.7 means 70% relevance, 30% diversity.
    search_kwargs = {"k": 7, "fetch_k": 20, "lambda_mult": 0.7}

    if allowed_modalities:
        # FIX: Normalize to lowercase — all stored metadata values are lowercase
        normalized = [m.lower() for m in allowed_modalities]
        search_kwargs["filter"] = {"modality": {"$in": normalized}}

    return store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)


class SimpleRAGChain:
    """
    Minimal RAG chain for the evaluation harness.
    Retrieves top-K documents via MMR then synthesizes with Llama-3.3-70B.
    """
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


def get_rag_chain(allowed_modalities: list | None = None) -> SimpleRAGChain:
    """
    Constructs the legacy RAG pipeline used by the evaluation harness.
    Synthesizes answers from multi-modal sources (Text/Tweets, IoT, Drone, KB PDFs).
    Uses Groq Llama-3.3-70B for synthesis (free API key required).

    Args:
        allowed_modalities: Optional modality filter for ablation studies.
            Pass ["text", "knowledge_base"] for Text RAG ablation.
            Pass None for full Multimodal RAG (no filter).
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

    system_prompt = (
        "You are an advanced Disaster Response AI Assistant.\n"
        "You have access to a multimodal Retrieval-Augmented Generation (RAG) system containing:\n"
        "1. Official knowledge bases (FEMA protocols, CDC emergency guides).\n"
        "2. Social media text reports from CrisisLex (historical, 2013 Colorado Floods).\n"
        "3. IoT telemetry data (USGS water level readings).\n"
        "4. Drone/satellite imagery descriptions (xView2 disaster imagery).\n\n"
        "CRITICAL RULES:\n"
        "- Base your answer ONLY on the provided context.\n"
        "- If context contains conflicting reports, note the conflict and cite the most recent timestamp.\n"
        "- Explicitly cite source metadata (e.g., 'According to FEMA Field Operations Guide...', "
        "'Based on CrisisLex tweet at [timestamp]...', 'Drone observation reports...').\n"
        "- If context is insufficient to answer, say so explicitly — do not speculate.\n"
        "- NEVER give advice that could endanger lives (e.g., 'safe to cross floodwater').\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    return SimpleRAGChain(llm, prompt, get_retriever(allowed_modalities))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query the Disaster Response RAG (legacy mode).")
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
