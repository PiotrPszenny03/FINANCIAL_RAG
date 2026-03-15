import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def ingest_pdf(pdf_path: str, vector_db_path: str = "faiss_index"):
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks.")
    
    print("Initializing Google Generative AI Embeddings...")
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "Twój_klucz_API_Gemini_tutaj":
        print("WARNING: GOOGLE_API_KEY is not set or is using the default placeholder in .env")
        print("Please update the .env file with your actual Gemini API key.")
        return False
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    print("Creating FAISS vector database (with rate limiting for free tier)...")
    import time
    
    # Process chunks in small batches to avoid 429 Rate Limit
    # Google Free Tier limits Requests Per Minute (RPM) explicitly
    batch_size = 5
    db = None
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
        
        # If it's the first batch, initialize the db
        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)
            
        # Severe restrictions for free-tier Gemini API (10 or 15 RPM for some endpoints)
        if i + batch_size < len(chunks):
            print("Pausing for 15 seconds to avoid Google Free Tier API limits...")
            time.sleep(15)
    
    print(f"Saving database to {vector_db_path}...")
    db.save_local(vector_db_path)
    
    print("Ingestion complete!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF file into FAISS vector DB")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    args = parser.parse_args()
    
    if os.path.exists(args.pdf_path):
        ingest_pdf(args.pdf_path)
    else:
        print(f"Error: File '{args.pdf_path}' not found.")
