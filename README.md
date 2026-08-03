 📚 Olympic PDF RAG Pipeline with LangChain, Chroma & Claude

 How did I generated PDF?
 I used Gemini to generate a PDF file with Olympic data from 1900 to till date for 4 countries i.e., India, UK, USA & China.

 ![output](<Screenshot 2026-08-03 at 9.35.43 PM.png>)

 Now I want to convert this application to real-time enterprise systems handle multi-document/multi-entity RAG. We can achieve this by following  Single-Index Metadata-Filtered RAG pattern .

 [ User Prompt ] 
       │
       ▼
[ Step 2: Metadata Router ] ── (Extracts: {"country": "India"})
       │
       ▼
[ Step 3: Filtered Vector Search ] ── (Queries ChromaDB with filter={"country": "India"})
       │
       ▼
[ Step 4: RAG Generation Chain ] ── (Passes filtered context to Claude)

📋 Step-by-Step Technical Implementation
Step 1: Ingestion Pipeline with Enriched Metadata
Instead of saving vectors into separate folders, process all PDFs through an ingestion pipeline that attaches a country tag to every single chunk before storing them in one shared Chroma database.
	1.	Load each PDF file.
	2.	Split the PDF into standard chunks (RecursiveCharacterTextSplitter).
	3.	Attach metadata: Inject {"country": ""} into each chunk's metadata dictionary.
	4.	Add all chunks from all PDFs into a single Chroma collection.
Key Technical Concept:
When LangChain stores documents in Chroma DB, each document payload looks like this under the hood:
Document(
    page_content="Neeraj Chopra won Silver in Javelin...",
    metadata={"country": "India", "source": "olympics_india.pdf", "page": 1}
)

Step 2: Build a Fast Metadata Router
Create a lightweight extraction step that inspects the incoming user query and identifies which country the user is asking about.
	1.	Option A (Deterministic / NER): Check if any known country keyword exists inside the string query.
	2.	Option B (Structured LLM Output): Use LangChain's .with_structured_output() (or a fast lightweight model like claude-3-haiku-20240307) with a simple schema (e.g., Pydantic model containing country_name: str).
Example Logic Output:
⚬	User Query: "List 2012 UK silver medals"
⚬	Router Output: "UK"
Step 3: Execute Filtered Retrieval
Instead of calling a different database for every country, pass the extracted country name into Chroma DB as a Search Filter.
	1.	Define a retriever from your single Chroma DB store.
	2.	Pass search arguments dynamically containing the metadata constraint:
# Chroma DB metadata filter format
search_kwargs = {
    "k": 5,
    "filter": {"country": target_country}
}

	3.	Chroma DB performs an index lookup only across vectors matching {"country": target_country}.
Step 4: Connect the Pipeline with LCEL
Chain the router, retriever, prompt, and LLM together using LangChain Expression Language (LCEL).
	1.	Input: User asks a question ("How many gold medals did India win in 2020?").
	2.	Router Execution: Extracts country = "India".
	3.	Retriever Execution: Queries Chroma DB with filter={"country": "India"} and returns only India-related chunks.
	4.	Prompt Construction: Combines the filtered chunks + original question into the standard prompt template.
	5.	LLM Generation: Claude generates the final grounded response.

### key technical highlights:
⚬	Zero Memory Overhead: One shared database connection pool rather than dynamic connection switching across multiple databases.
⚬	Deterministic Isolation: Guarantees zero "bleed-through" (e.g., query about India will physically never read a UK chunk).
⚬	Instant Extensibility: To add a 7th or 100th PDF (e.g., olympics_france.pdf), you just run the ingestion pipeline. No code updates, re-indexing, or server restarts required.

Why Chroma DB?
Chroma DB uses HNSW (Hierarchical Navigable Small World) as its default underlying vector indexing engine.
When Chroma stores your HuggingFace embeddings locally, it automatically builds an HNSW graph index. When you execute a search with a metadata filter, Chroma performs a filtered HNSW graph traversal, searching only the nodes tagged with {"country": "India"}

🚀 Summary of Your Local Tech Stack
Component	Production System Equivalent	Your Local Setup (100% Free)
Vector DB	Enterprise Pinecone / Qdrant Cluster	Local Chroma DB
Indexing Engine	HNSW Graph Search	Built-in HNSW (Chroma)
Embeddings	OpenAI text-embedding-3	Local HuggingFace (all-MiniLM-L6-v2)
Metadata Filter	Cloud Payloads	Native Chroma SQL/Metadata Filters
LLM Orchestrator	Claude / GPT-4	Anthropic Claude API via LangChain LCEL

