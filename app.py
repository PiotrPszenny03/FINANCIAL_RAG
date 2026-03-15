

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Ensure we have our modules
try:
    from ingest import ingest_pdf
    from rag_chain import get_rag_chain
except ImportError as e:
    st.error(f"Error loading modules: {e}")
    st.stop()

load_dotenv()

st.set_page_config(page_title="Financial RAG", page_icon="📈", layout="wide")

st.title("📈 Financial RAG Assistant")
st.markdown("This system can analyze uploaded PDF files (e.g., prospectuses, annual reports) and answer questions by extracting specific information from the text.")

# Sidebar for configuration and file upload
with st.sidebar:
    st.header("Configuration")
    api_key_env = os.getenv("GOOGLE_API_KEY", "")
    if api_key_env == "Twój_klucz_API_Gemini_tutaj" or api_key_env == "Your_Gemini_API_Key_Here":
        api_key_env = ""
        
    api_key = st.text_input("Gemini API Key", value=api_key_env, type="password")
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.header("1. Upload a PDF file")
    uploaded_file = st.file_uploader("Select a financial report (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Analyze document"):
            with st.spinner("Processing and vectorizing the document (FAISS)..."):
                # Save uploaded file to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Ingest
                success = ingest_pdf(tmp_path)
                os.remove(tmp_path)
                
                if success:
                    st.success("Document processed successfully!")
                    st.session_state["document_loaded"] = True
                else:
                    st.error("Error during processing. Please check your API key.")

# Main area for chat
st.header("2. Ask a question about the report")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.get("document_loaded", False) or os.path.exists("faiss_index"):
    # React to user input
    if prompt := st.chat_input("E.g. What are the main currency risks of the company?"):
        
        if not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") in ["Twój_klucz_API_Gemini_tutaj", "Your_Gemini_API_Key_Here"]:
            st.error("Please provide a valid Gemini API Key in the sidebar first.")
            st.stop()

        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Searching for an answer in the document..."):
                try:
                    chain = get_rag_chain()
                    response = chain.invoke(prompt)
                    answer = response["answer"]
                    sources = response["context"]
                    
                    st.markdown(answer)
                    
                    with st.expander("📄 Show sources from PDF (Context)"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Fragment {i+1}:**")
                            st.info(doc.page_content)
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.info("👈 First, upload and analyze a PDF file in the sidebar, or ensure the 'faiss_index' database exists in the folder.")
