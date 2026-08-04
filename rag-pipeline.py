"""RAG pipeline for querying Olympic PDF data with ChromaDB and Claude.

This script reads country-specific Olympic PDFs, stores their text chunks in a
shared vector database, routes questions to the relevant country metadata,
retrieves the most relevant context, and generates grounded answers using an
LLM. A lightweight semantic cache is also included to reuse answers for similar
queries.
"""

import os
import time
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

# Local paths for the persisted vector database and the PDF source folder.
CHROMA_PATH = "./chroma_db"
DOCUMENTS_DIR = "./Documents/"

# Maps each supported country to the PDF file that contains its data.
COUNTRY_PDF_MAPPING = {
    "India": "Olympics_india.pdf",
    "UK": "Olympics_uk.pdf",
    "USA": "Olympics_usa.pdf",
    "China": "Olympics_china.pdf",
    "Germany": "Olympics_germany.pdf",
    "Japan": "Olympics_japan.pdf",
}
# ------------------------------------------------------------------
# SEMANTIC CACHE CLASS (In-Memory Vector Search)
# ------------------------------------------------------------------
class InMemorySemanticCache:
    """Stores query embeddings and answers in RAM for fast reuse of similar questions."""
    
    def __init__(self, embedding_model, similarity_threshold: float = 0.90, max_capacity: int = 200):
        # The embedding model is used to convert text into vectors for similarity matching.
        self.embedding_model = embedding_model
        # Higher threshold means the cache only returns a hit when the new query is very close to an old one.
        self.similarity_threshold = similarity_threshold
        # Keep the cache bounded so memory usage stays predictable.
        self.max_capacity = max_capacity
        self.cache: List[Dict] = []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Return a normalized similarity score between two vectors.

        A score close to 1.0 means the vectors point in nearly the same direction,
        which indicates the two queries are semantically similar.
        """
        v1, v2 = np.array(vec1, dtype=np.float32), np.array(vec2, dtype=np.float32)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        similarity = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return float(np.clip(similarity, -1.0, 1.0))  # Prevent precision overflow above 1.0.

    def lookup(self, query: str) -> Optional[str]:
        """Return a cached answer if a similar query was asked before."""
        if not self.cache:
            return None

        # Convert the incoming question into an embedding vector.
        query_vector = self.embedding_model.embed_query(query)
        
        best_score = -1.0
        best_response = None
        best_query = None

        # Compare the new query against every cached query vector.
        for entry in self.cache:
            score = self._cosine_similarity(query_vector, entry["vector"])
            if score > best_score:
                best_score = score
                best_response = entry["response"]
                best_query = entry["query"]

        if best_score >= self.similarity_threshold:
            print(f"\n⚡ [CACHE HIT]: Similarity Score: {best_score:.4f} >= Threshold: {self.similarity_threshold}")
            print(f"   Matched Original Query: '{best_query}'")
            return best_response

        print(f"\n💨 [CACHE MISS]: Best Similarity Score was {best_score:.4f} (Required: {self.similarity_threshold})")
        return None

    def add(self, query: str, response: str):
        """Save a new query-response pair into the in-memory cache."""
        if len(self.cache) >= self.max_capacity:
            # Evict the oldest entry to keep the cache bounded in size.
            self.cache.pop(0)

        query_vector = self.embedding_model.embed_query(query)
        self.cache.append({
            "query": query,
            "vector": query_vector,
            "response": response,
            "timestamp": time.time()
        })
        print(f"💾 [Cache Saved]: Successfully saved query to RAM cache. Total entries: {len(self.cache)}")


# ------------------------------------------------------------------
# STEP 1: Load or Initialize Vector DB
# ------------------------------------------------------------------
def initialize_vector_store():
    """Load an existing ChromaDB index or build one from the PDF documents."""
    # Use a lightweight sentence-transformer embedding model for text-to-vector conversion.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(CHROMA_PATH):
        # Reuse the already-created index to avoid reprocessing the PDFs every run.
        print("⚡ Loading pre-indexed Chroma Vector DB...")
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    else:
        # First run: read each PDF, split it into smaller chunks, and store them with metadata.
        print("📦 First-time setup: Ingesting PDFs into unified Chroma DB...")
        all_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for country, pdf in COUNTRY_PDF_MAPPING.items():
            pdf_path = os.path.join(DOCUMENTS_DIR, pdf)
            if not os.path.exists(pdf_path):
                print(f"📄 File not found: {pdf_path}")
                continue

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            docs_chunk = text_splitter.split_documents(documents)

            # Tag each chunk with the country so retrieval can be filtered precisely later.
            for doc in docs_chunk:
                doc.metadata["country"] = country
            all_chunks.extend(docs_chunk)

        vector_store = Chroma.from_documents(
            all_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
        )
        print("✅ Ingestion complete!")
    return vector_store

# ------------------------------------------------------------------
# STEP 2: Router Schema & Setup
# ------------------------------------------------------------------
class RouterInput(BaseModel):
    """Schema used by the router to extract a country from a user query."""
    target_country:Optional[str] = Field(
        description="Extracted country name from query. Must be one of: 'India', 'UK', 'USA', 'Germany', 'Japan', 'China'. If no country is referenced, return None."
    )

def create_router_chain(llm):
    """Create a lightweight chain that identifies whether a question targets a country."""
    # Ask the LLM to return a structured object instead of free-form text.
    structured_llm = llm.with_structured_output(RouterInput)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Identify if the prompt is about a specific country: 'India', 'UK', 'USA', 'Germany', 'Japan', or 'China'. Map variants like 'United Kingdom', 'Great Britain', or 'Britain' to 'UK'."),
        ("human", "{question}")
    ])

    # The router uses the same large model instance for simple intent extraction.
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    router_chain = (
        prompt_template
        | structured_llm
    )
    return router_chain

# ------------------------------------------------------------------
# STEP 3: RAG Processing Function
# ------------------------------------------------------------------
def process_query(query: str, vector_store, router_chain, llm_generator, semantic_cache: InMemorySemanticCache):
    """Run the full RAG flow for a single user question."""
    start_time = time.time()

    # Step 1: Check whether a similar question has already been answered.
    cached_response = semantic_cache.lookup(query)
    if cached_response:
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"⏱️ [RAM CACHE] Served response in {elapsed_ms:.2f} ms")
        return cached_response

    # Step 2: If there is no useful cache hit, run the full retrieval-augmented pipeline.
    print("🚀 Running full RAG pipeline...")

    # Ask the router to identify if the question targets a specific country.
    router_output = router_chain.invoke({"question": query})
    target_country = router_output.target_country

    # Build the retrieval settings. If a country is detected, filter the search to that country.
    search_kwargs = {"k": 5}
    if not target_country:
        print("[System Router]: No specific country detected. Searching globally...")
    else:
        print(f"[System Router]: Detected Country -> '{target_country}'. Applying ChromaDB Filter: {{'country': '{target_country}'}}")
        search_kwargs["filter"] = {"country": target_country}

    # Step 3: Retrieve the most relevant document chunks from ChromaDB.
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    retrieved_docs = retriever.invoke(query)

    sources = set(d.metadata.get('source', 'Unknown') for d in retrieved_docs)
    print(f"[Vector Search]: Filtered lookup returned {len(retrieved_docs)} chunks from {list(sources)}")

    # Step 4: Build a prompt that includes only the retrieved context.
    rag_prompt_template = ChatPromptTemplate.from_template(
        """
        You are an expert Olympic Assistant. Answer the user's question clearly and concisely using ONLY the provided context.
        If the information is not explicitly present in the context, say "I don't have records for that request in the database."

        Context:
        {context}

        Question: {question}

        Answer:
        """
    )

    formatted_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    chain = rag_prompt_template | llm_generator | StrOutputParser()

    response = chain.invoke({"context": formatted_context, "question": query})

    # Step 5: Cache the final answer so similar follow-up questions can be answered faster.
    semantic_cache.add(query, response)

    elapsed_s = time.time() - start_time
    print(f"⏱️ [Full Pipeline] Executed in {elapsed_s:.2f} seconds")

    return response

# ------------------------------------------------------------------
# STEP 4: Interactive Terminal Loop (Interactive CLI)
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize the embedding model once and reuse it for both caching and vector store operations.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the semantic cache once so it persists across the interactive loop.
    semantic_cache = InMemorySemanticCache(embedding_model=embeddings, similarity_threshold=0.90)

    # Initialize the LLM and the vector database for retrieval.
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    vector_store = initialize_vector_store()
    router_chain = create_router_chain(llm)

    print("\n" + "="*50)
    print("🏆 Olympic RAG Assistant Ready!")
    print("Type your questions below (e.g., '2012 gold medals for India').")
    print("Type 'exit' or 'quit' to close the program.")
    print("="*50 + "\n")

    # Run an interactive loop so the user can ask multiple questions in one session.
    while True:
        query = input("User Prompt: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        response = process_query(query, vector_store, router_chain, llm, semantic_cache=semantic_cache)
        print("\n[Response]:", response)