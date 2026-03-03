import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ==========================
# Load environment variables
# ==========================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# ==========================
# Initialize FastAPI
# ==========================

app = FastAPI()

# ==========================
# Load Embedding Model
# (Must match ingestion model)
# ==========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================
# Load Vector Store
# ==========================

vector_store = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

# ==========================
# Request Model
# ==========================

class ChatRequest(BaseModel):
    question: str


# ==========================
# Chat Endpoint
# ==========================

@app.post("/chat")
def chat(request: ChatRequest):

    # 1️⃣ Semantic Search
    docs = vector_store.similarity_search(request.question, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    # 2️⃣ Build Prompt
    prompt = f"""
You are a helpful assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{request.question}
"""

    # 3️⃣ Call Groq LLM (UPDATED MODEL)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # ✅ Updated working model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    # 4️⃣ Return JSON Response
    return {
        "answer": answer,
        "sources": [doc.metadata.get("page", "Unknown") for doc in docs]
    }


# ==========================
# Root Endpoint
# ==========================

@app.get("/")
def root():
    return {"status": "Backend running successfully 🚀"}