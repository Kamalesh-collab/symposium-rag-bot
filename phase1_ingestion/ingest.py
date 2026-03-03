import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==========================
# 1. LOAD PDF
# ==========================

print("Loading PDF...")

pdf_path = "data/symposium.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Loaded {len(documents)} pages")


# ==========================
# 2. CHUNK TEXT
# ==========================

print("Chunking text...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# ==========================
# 3. CREATE EMBEDDINGS
# ==========================

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings ready")


# ==========================
# 4. STORE IN FAISS
# ==========================

print("Creating vector store...")

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("Saving vector store...")

vector_store.save_local("vector_store")

print("Done. Vector store created successfully!")