import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# Load environment variables
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="Symposium RAG Bot")
st.title("🤖 Symposium RAG Bot")

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector store
vector_store = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Chat input
user_question = st.chat_input("Ask something about the symposium...")

if user_question:
    with st.spinner("Thinking..."):

        # Step 1: Search similar chunks
        docs = vector_store.similarity_search(user_question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Step 2: Build prompt
        prompt = f"""
        Answer the question using the context below.
        If answer not found, say you don't know.

        Context:
        {context}

        Question:
        {user_question}
        """

        # Step 3: Call Groq LLM
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        answer = response.choices[0].message.content

        st.write(answer)