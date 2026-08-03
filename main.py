import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)

def main():
    # step1: load the PDF document
    loader = PyPDFLoader("./Documents/Olympics.pdf")
    documents = loader.load()

    # step2: split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_chunk = text_splitter.split_documents(documents)

    # step3: embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # step4: store into vector database
    vector_store = Chroma.from_documents(docs_chunk, embedding=embeddings, persist_directory="./chroma_db")

    #step5: create a retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}) 

    # step 6: Prompt template
    prompt_template = PromptTemplate(
        template="""
        You are an AI assistant.
        Answer the question using only the context below.

        Context: {context}
        Question: {question}
        """,
        input_variables=["question", "context"],
    )

    # step7: Initialize the LLM
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
    )

    # step 8: Create pipeline
    rag_chain = (
        RunnableParallel(context=retriever, question=RunnablePassthrough())
        | prompt_template
        | llm
        | StrOutputParser()
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the program.")
            break
        answer = rag_chain.invoke(user_input)
        print(f"AI: {answer}\n")

if __name__ == "__main__":
    main()