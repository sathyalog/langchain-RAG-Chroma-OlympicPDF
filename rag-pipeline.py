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

CHROMA_PATH = "./chroma_db"
DOCUMENTS_DIR = "./Documents/"

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
    """Stores query vectors and responses in RAM to serve rephrased queries in < 5ms."""
    
    def __init__(self, embedding_model, similarity_threshold: float = 0.90, max_capacity: int = 200):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_capacity = max_capacity
        self.cache: List[Dict] = []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculates normalized cosine similarity between two 1D vectors."""
        v1, v2 = np.array(vec1, dtype=np.float32), np.array(vec2, dtype=np.float32)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        similarity = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return float(np.clip(similarity, -1.0, 1.0))  # Prevent precision overflow > 1.0

    def lookup(self, query: str) -> Optional[str]:
        """Checks if a semantically similar query exists in RAM."""
        if not self.cache:
            return None

        # Generate embedding vector for incoming query
        query_vector = self.embedding_model.embed_query(query)
        
        best_score = -1.0
        best_response = None
        best_query = None

        # Iterate through all entries saved in RAM
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
        """Saves query vector and LLM response to RAM."""
        if len(self.cache) >= self.max_capacity:
            self.cache.pop(0)  # Evict oldest entry (LRU)

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
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(CHROMA_PATH):
        print("⚡ Loading pre-indexed Chroma Vector DB...")
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    else:
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
    """Schema for extracting target country from user queries."""
    target_country:Optional[str] = Field(
        description="Extracted country name from query. Must be one of: 'India', 'UK', 'USA', 'Germany', 'Japan', 'China'. If no country is referenced, return None."
    )

def create_router_chain(llm):
    structured_llm = llm.with_structured_output(RouterInput)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Identify if the prompt is about a specific country: 'India', 'UK', 'USA', 'Germany', 'Japan', or 'China'. Map variants like 'United Kingdom', 'Great Britain', or 'Britain' to 'UK'."),
        ("human", "{question}")
    ])
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
    start_time = time.time()

    # 1. Lookup in Cache First
    cached_response = semantic_cache.lookup(query)
    if cached_response:
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"⏱️ [RAM CACHE] Served response in {elapsed_ms:.2f} ms")
        return cached_response

    # 2. Cache Miss -> Proceed with Normal Pipeline
    print("🚀 Running full RAG pipeline...")
    router_output = router_chain.invoke({"question": query})
    target_country = router_output.target_country

    search_kwargs = {"k": 5}
    if not target_country:
        print("[System Router]: No specific country detected. Searching globally...")
    else:
        print(f"[System Router]: Detected Country -> '{target_country}'. Applying ChromaDB Filter: {{'country': '{target_country}'}}")
        search_kwargs["filter"] = {"country": target_country}

    # 3. Retrieve Documents
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    retrieved_docs = retriever.invoke(query)

    sources = set(d.metadata.get('source', 'Unknown') for d in retrieved_docs)
    print(f"[Vector Search]: Filtered lookup returned {len(retrieved_docs)} chunks from {list(sources)}")

    # 4. Generate Response
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

    # 5. Store in RAM Cache AFTER response generation
    semantic_cache.add(query, response)

    elapsed_s = time.time() - start_time
    print(f"⏱️ [Full Pipeline] Executed in {elapsed_s:.2f} seconds")

    return response

# ------------------------------------------------------------------
# STEP 4: Interactive Terminal Loop (Interactive CLI)
# ------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Initialize Embeddings ONCE
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Instantiate Semantic Cache ONCE (Persists in RAM across loop iterations)
    semantic_cache = InMemorySemanticCache(embedding_model=embeddings, similarity_threshold=0.90)
    # 3. Initialize LLM & Vector Store
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    vector_store = initialize_vector_store()
    router_chain = create_router_chain(llm)
    print("\n" + "="*50)
    print("🏆 Olympic RAG Assistant Ready!")
    print("Type your questions below (e.g., '2012 gold medals for India').")
    print("Type 'exit' or 'quit' to close the program.")
    print("="*50 + "\n")
    while True:
        query = input("User Prompt: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        response = process_query(query, vector_store, router_chain, llm, semantic_cache=semantic_cache)
        print("\n[Response]:", response)