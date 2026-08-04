## 📚 Multi-Country Olympic PDF RAG Pipeline (Production-Grade Architecture)
A lightweight, enterprise-ready Retrieval-Augmented Generation (RAG) system built with Python, LangChain, Chroma DB, HuggingFace, and Anthropic's Claude 3.

#### 💡 Origin & Project Inspiration
"With the Olympics currently happening, I wanted to build a domain-specific search system to query historical medal records across countries. Instead of just building a simple single-document RAG, I challenged myself to design a multi-document system architecture capable of scaling to all 200+ competing nations without needing any future logic rewrites or new database clusters."

#### 📄 Dataset Creation & Genesis
Because real-world web PDFs are often noisy, poorly formatted, or buried across hundreds of sites, I conceptualized and built my own structured dataset from scratch:
⚬	**Generation Method:** Initially, Used Google Gemini to generate clean, text-searchable historical data spanning 1900 to present for 4 initial countries: India 🇮🇳, Great Britain 🇬🇧, USA 🇺🇸, and China 🇨🇳.
⚬	**Future-Proofing:**  Formatted the dataset explicitly to test clean text-splitting and metadata isolation before scaling to more nations.
How to run single PDF rag?
`uv run main.py`

This is an output of a single PDF document reading(Simple system): 
 ![output](<Screenshot 2026-08-03 at 9.35.43 PM.png>)

 Now I want to convert this application to real-time enterprise systems handle multi-document/multi-entity RAG. We can achieve this by following  **Single-Index Metadata-Filtered RAG** pattern .

 I added 2 more countries and total 6 countries with separate PDF's like Olympic_india.pdf, Olympic_uk.pdf, Olympic_usa.pdf, Olympic_germany.pdf, Olympic_japan.pdf, Olympics_china.pdf and not single PDF(Olympics.pdf).

#### How to run Single-Index Metadata-Filtered RAG?
`uv run rag-pipeline.py`
 **Single-Index Metadata-Filtered RAG Output**
 ![final-output](<Screenshot 2026-08-04 at 2.59.33 PM.png>)

 #### 🏗️ From Idea to Production Architecture
The Naive Approach vs. Production Standard
⚬	❌ The Naive Prototype: Create separate vector databases for every country file, or spin up complex LLM tool-calling agents to choose files. (This inflates hosting costs, introduces huge latency, and breaks when scaling to 100+ countries).
⚬	✅ The Production Standard (Implemented Here): Use a Single-Index Metadata-Filtered RAG Pattern. All country PDFs live in one unified vector database, with chunks tagged using metadata like {"country": "India"}. Chroma DB filters vectors at the hardware level during graph traversal.

#### 🛠️ System Architecture Flow
                                [ User Prompt ]
                                       │
                                       ▼
                         [ Step 1: Metadata Router ] 
                    (Extracts: {"country": "India"} via Pydantic)
                                       │
                                       ▼
                   [ Step 2: Filtered Vector Retrieval ] 
               (Queries ChromaDB with filter={"country": "India"})
                                       │
                                       ▼
                      [ Step 3: RAG Generation Chain ] 
                   (Passes grounded context to Claude 3)

####  🚀 Technical Implementation Steps
**Step 1:** Unified Ingestion with Enriched Metadata
Instead of creating isolated databases, all PDF documents (olympics_india.pdf, olympics_uk.pdf, etc.) are processed through a single ingestion pipeline. Every chunk receives strict metadata prior to vector storage:
```
Document(
    page_content="Neeraj Chopra won Silver in Javelin...",
    metadata={"country": "India", "source": "olympics_india.pdf"}
)
```

**Step 2:** Lightweight Intent Routing
When a prompt comes in, a fast, lightweight router inspects the user string and extracts the targeted entity:
⚬	User Prompt: "List 2012 UK silver medals"
⚬	Router Output: "UK"
Production Note: Router logic can utilize fast string/regex matching for common queries, with a fast LLM (claude-3-haiku) as fallback for ambiguous queries.
**Step 3:** Filtered Vector Retrieval (HNSW Engine)
The extracted entity ("UK") is injected directly into Chroma DB's lookup query as a search filter:

####  Chroma DB Metadata Search Filter
```
search_kwargs = {
    "k": 5,
    "filter": {"country": target_country}
}
```

**Under the Hood:** Chroma DB uses **HNSW (Hierarchical Navigable Small World)** graph indexing. When a metadata filter is supplied, Chroma performs graph traversal only on vector nodes matching the constraint, achieving total data isolation.
**Step 4:** LCEL Pipeline Generation
Using **LangChain Expression Language (LCEL)**, the output of the metadata filter feeds into the final prompt context, returning strictly relevant data to Anthropic's Claude model.

#### 🛠️ Tech Stack: Local vs. Cloud Production Equivalent
This pipeline mirrors enterprise-grade architecture using 100% free local tools:
![table](<Screenshot 2026-08-03 at 11.31.44 PM.png>)

####  ⭐ Key Architecture Highlights
⚬	🔒 Deterministic Isolation: Eliminates data bleed-through. A search about India physically cannot search or return chunks tagged for the UK.
⚬	⚡ Zero-Overhead Scaling: Adding a 5th or 200th country PDF requires zero server restarts or structural code updates. Just ingest the file with its respective metadata tag.
⚬	💰 Cost & Latency Optimized: Operates out of a single local database connection pool, drastically reducing memory usage and eliminating multi-database network overhead.
