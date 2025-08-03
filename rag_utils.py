import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load .env variables (make sure GOOGLE_API_KEY is in your .env)
load_dotenv()

# Global RAG chain variable
rag_chain = None


def split_pdf_to_chunks(pdf_path):
    """Load PDF and split it into text chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=10
    )
    chunks = splitter.split_documents(pages)
    return chunks


def process_pdf(pdf_path):
    """Process the uploaded PDF and build vector store + RAG chain."""
    global rag_chain

    # Step 1: Split into chunks
    documents = split_pdf_to_chunks(pdf_path)
    texts = [doc.page_content for doc in documents]

    # Step 2: Embedding
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)
    vectorstore.save_local("vector_db")

    # Step 3: Set up Gemini LLM and RAG chain
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )


def get_rag_chain():
    """Expose the global RAG chain."""
    return rag_chain
