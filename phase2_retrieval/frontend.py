import streamlit as st
import requests

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="RAG Symposium Bot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Symposium RAG Chatbot")
st.write("Ask questions about the Symposium PDF.")

# ==========================
# Backend URL
# ==========================

BACKEND_URL = "http://127.0.0.1:8000/chat"

# ==========================
# Chat Input
# ==========================

question = st.text_input("Enter your question:")

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):

            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"question": question}
                )

                result = response.json()

                st.subheader("Answer:")
                st.write(result["answer"])

                st.subheader("Sources:")
                for src in result["sources"]:
                    st.write(f"Page {src + 1}")

            except Exception as e:
                st.error(f"Error connecting to backend: {e}")