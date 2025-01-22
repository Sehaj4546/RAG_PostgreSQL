## RAG model : LLAMA3.2 LLM model used to chat with pdf and user history stored in POSTGRES SQL DATABASE.

import os
import torch
import psycopg2
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# PostgreSQL Database Connection
def connect_db():
    return psycopg2.connect(
        dbname="rag_system",
        user="postgres",   # Change this
        password="Sehaj@4546",  # Change this
        host="localhost",
        port="5432"
    )

# Function to Save User Query & Response in PostgreSQL
def save_user_history(user_id, query, response):
    """Stores user query and response in PostgreSQL"""
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_history (user_id, query, response) VALUES (%s, %s, %s)",
            (user_id, query, response)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error saving history: {e}")

# Function to Retrieve Last 5 User Queries from PostgreSQL
def get_user_history(user_id):
    """Retrieves last 5 user interactions from PostgreSQL"""
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT query, response FROM user_history WHERE user_id = %s ORDER BY timestamp DESC LIMIT 5",
            (user_id,)
        )
        history = cursor.fetchall()
        cursor.close()
        conn.close()
        return history
    except Exception as e:
        print(f"Error retrieving history: {e}")
        return []

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize LLM Model
try:
    llm = OllamaLLM(
        model="llama3.2",  
        temperature=0.5,  
        device=device  # Use GPU for LLM if available
    )
except Exception as e:
    print(f"Error initializing Ollama: {e}")
    exit()

# Load PDFs from Folder
def load_pdfs_from_folder(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            try:
                loader = PDFPlumberLoader(pdf_path)
                documents.extend(loader.load())
                print(f"Loaded PDF: {file_name}")
            except Exception as e:
                print(f"Error loading PDF {file_name}: {e}")
    return documents

# Set PDF Folder Path
pdf_folder_path = r"C:\Users\LEGION\Desktop\OLLAMA\data\combine pdfs"

# Load Documents
documents = load_pdfs_from_folder(pdf_folder_path)

# Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

print(f"Total number of chunks created: {len(texts)}\n")

# Initialize Embeddings Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs={"device": device}  # Use GPU if available
)

# Create FAISS Vector Store
db = FAISS.from_documents(texts, embeddings)

# Create Retriever
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# Define RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  
    retriever=retriever
)

# Function to Handle User Queries
def get_response(user_id, query):
    try:
        # Fetch user history from PostgreSQL
        user_history = get_user_history(user_id)

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(query)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]

        if not retrieved_texts:
            return "I can only answer questions based on the uploaded HR documents. No relevant information was found for your query."

        # Filter retrieved texts based on query words
        filtered_texts = [text for text in retrieved_texts if any(word in text.lower() for word in query.lower().split())]

        if not filtered_texts:
            return "I can only answer questions based on the uploaded HR documents. No relevant information was found for your query."

        limited_texts = filtered_texts[:4]

        document_context = "\n\n".join(limited_texts)

        # Format conversation history from database
        history_context = "\n".join([f"User: {q}\nAssistant: {r}" for q, r in user_history[-5:]])

        if history_context:
            prompt = f"You are an assistant for answering the questions asked by user, from the provided documents. STRICTLY Use the given documents to answer the question. Use at most five sentences and keep the answer concise. If asked a question outside the stored data or out of context say you don't know. Some questions may be asked related to the context of previous questions, so in such cases utilize the user history to get context.\n\nConversation history:\n{history_context}\n\nDocuments:\n{document_context}\n\nCurrent query:\nUser: {query}\nAssistant:"
        else:
            prompt = f"You are an assistant for question-answering tasks. STRICTLY Use the given documents to answer the question. Use at most five sentences and keep the answer concise. If asked a question outside the stored data or out of context say you don't know.\n\nDocuments:\n{document_context}\n\nUser: {query}\nAssistant:"

        # Generate response
        result = qa.invoke({"query": prompt})
        response = result["result"]

        # Save user interaction in database
        save_user_history(user_id, query, response)

        return response
    except Exception as e:
        return f"Error during QA: {e}"

# Main Loop for User Queries
if __name__ == "__main__":
    while True:
        user_id = input("Enter your user ID (or 'exit' to terminate): ")
        if user_id.lower() == 'exit':
            print("Program terminated. Goodbye!")
            break
        
        while True:
            query = input(f"Enter your question, {user_id} (or 'quit' to switch users): ")
            if query.lower() == 'quit':  
                print(f"Session ended for user {user_id}.\n")
                break
            response = get_response(user_id, query)
            print("\nResponse:", response, "\n")
