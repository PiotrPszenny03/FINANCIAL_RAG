import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_rag_chain(vector_db_path: str = "faiss_index"):
    if not os.path.exists(vector_db_path):
        raise FileNotFoundError(f"Vector DB not found at {vector_db_path}. Please run ingest.py first.")
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.0
    )
    
    system_prompt = (
        "You are an AI assistant specialized in analyzing financial reports. "
        "Use the following retrieved context fragments to answer the question. "
        "If you don't know the answer, say that you don't know. Do not invent information about the company's finances outside the provided context; rely exclusively on it.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    )
    
    rag_chain = setup_and_retrieval.assign(
        answer=(
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )
    )
    
    return rag_chain

def ask_question(question: str):
    try:
        chain = get_rag_chain()
        print(f"\nQuestion: {question}")
        print("Analyzing context and generating answer...\n")
        response = chain.invoke(question) # in LCEL RunnablePassthrough() expects string input
        print("Answer:")
        print(response["answer"])
        print("\nContext used to answer:")
        for i, doc in enumerate(response["context"]):
            print(f"--- Fragment {i+1} ---")
            print(doc.page_content[:200] + "...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ask_question(question)
    else:
        print("Provide a question as a script argument.")
