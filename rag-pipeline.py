import os
import sys
from typing import Optional
from pydantic import BaseModel, Field
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
def process_query(query:str, vector_store, router_chain, llm_generator):
    #1 : Route intent
    router_output = router_chain.invoke({"question": query})
    target_country = router_output.target_country

    # 2. Configure Dynamic Filter
    search_kwargs = {"k": 5}
    if not target_country:
        return "I'm sorry, I don't have any data on that. Please try rephrasing your question."
    else:
        print(f"\n[System Router]: Detected Country -> '{target_country}'. Applying ChromaDB Filter: {{'country': '{target_country}'}}")
        search_kwargs["filter"] = {"country": target_country}

    # 3. Retrieve Documents
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    retrieved_docs = retriever.invoke(query)

    # Optional debug logging for source verification
    sources = set(d.metadata.get('source', 'Unknown') for d in retrieved_docs)
    print(f"[Vector Search]: Filtered lookup returned {len(retrieved_docs)} chunks from {list(sources)}")

    #4. Generate response
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
    return response

# ------------------------------------------------------------------
# STEP 4: Interactive Terminal Loop (Interactive CLI)
# ------------------------------------------------------------------

if __name__ == "__main__":
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
        response = process_query(query, vector_store, router_chain, llm)
        print("\n[Response]:", response)