import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def embed_and_store(text, store_path="faiss_store"):
    """Create or load FAISS vector store for given text."""
    from langchain_community.embeddings import HuggingFaceEmbeddings  # Move import here

    # Initialize embeddings model inside the function
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Check if already exists
    if os.path.exists(store_path):
        print("✅ Loading existing FAISS store...")
        return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

    print("⚡ Creating new embeddings...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = splitter.split_text(text)
    
    vectordb = FAISS.from_texts(texts, embedding=embeddings)
    vectordb.save_local(store_path)
    print("💾 FAISS store saved!")
    return vectordb
