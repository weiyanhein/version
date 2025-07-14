import json
import os
import shutil # For removing directory
from pymongo import MongoClient

# Updated imports for LangChain components
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma # Using langchain_chroma for Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import configuration from your 'source' directory
from config import MONGO_URI, DB_NAME, COLLECTION_NAME, CHROMA_PERSIST_DIR, OLLAMA_BASE_URL, EMBEDDING_MODEL

def ingest_data():
    """
    Loads sample data into MongoDB, processes it for RAG,
    and stores embeddings in ChromaDB.
    """
    print("--- Starting Data Ingestion ---")

    # 1. Connect to MongoDB and load data
    try:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Clear existing data in MongoDB for a fresh ingest
        collection.delete_many({})
        print(f"Cleared existing documents from MongoDB '{DB_NAME}.{COLLECTION_NAME}'.")

        # Load data from JSON file
        json_file_path = 'C:\\Users\\WaiYanHein\\Desktop\\version\\data\\cosmetic_products.json'
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Sample data file not found: {json_file_path}")

        with open(json_file_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)

        # Insert data into MongoDB
        print(f"Inserting {len(sample_data)} documents into MongoDB '{DB_NAME}.{COLLECTION_NAME}'...")
        collection.insert_many(sample_data)
        print("MongoDB ingestion complete.")

    except Exception as e:
        print(f"Error during MongoDB operations: {e}")
        print("Please ensure MongoDB is running and accessible at the specified URI.")
        return # Exit if MongoDB connection fails

    # 2. Prepare data for embedding (create text chunks)
    documents_for_chroma = []
    for doc in sample_data:
        # Construct a descriptive text from the document fields for embedding
        content = (
            f"Product ID: {doc.get('id', 'N/A')}\n"
            f"Name: {doc.get('name', 'N/A')}\n"
            f"Type: {doc.get('type', 'N/A')}\n"
            f"Brand: {doc.get('brand', 'N/A')}\n"
            f"Ingredients: {', '.join(doc.get('ingredients', []))}\n"
            f"Benefits: {', '.join(doc.get('benefits', []))}\n"
            f"Usage: {doc.get('usage', 'N/A')}"
        )
        
        # Prepare metadata for ChromaDB - ONLY include simple types (str, int, float, bool, None)
        metadata = {
            "id": doc.get('id', 'N/A'),
            "name": doc.get('name', 'N/A'),
            "type": doc.get('type', 'N/A'),
            "brand": doc.get('brand', 'N/A'),
        }
        
        documents_for_chroma.append({
            "page_content": content,
            "metadata": metadata
        })
    mongo_client.close() # Close MongoDB connection after data retrieval

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    
    # Create LangChain Document objects from prepared data
    chunks = text_splitter.create_documents(
        [d["page_content"] for d in documents_for_chroma],
        metadatas=[d["metadata"] for d in documents_for_chroma]
    )

    print(f"Created {len(chunks)} text chunks for embedding.")

    # 3. Generate Embeddings and Store in ChromaDB
    try:
        print(f"Initializing OllamaEmbeddings with model: {EMBEDDING_MODEL} at {OLLAMA_BASE_URL}...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

        print(f"Storing embeddings in ChromaDB at {CHROMA_PERSIST_DIR}...")

        # Ensure the persist directory exists
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        # Delete existing ChromaDB data to ensure a fresh start
        if os.path.exists(CHROMA_PERSIST_DIR) and os.path.isdir(CHROMA_PERSIST_DIR):
            try:
                shutil.rmtree(CHROMA_PERSIST_DIR)
                print(f"Removed existing ChromaDB data at {CHROMA_PERSIST_DIR}")
            except OSError as e:
                print(f"Error removing ChromaDB directory: {e}. Please delete it manually if persists.")
                return # Exit if directory removal fails

        # Create ChromaDB from documents and persist
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        # Note: vectorstore.persist() is deprecated in Chroma 0.4.x+ as docs are automatically persisted.
        print("Embeddings stored in ChromaDB.")

    except Exception as e:
        print(f"Error during embedding generation or ChromaDB storage: {e}")
        print("Please ensure Ollama is running and the embedding model is pulled.")
        return # Exit if embedding/ChromaDB fails

    print("--- Data Ingestion Complete ---")

if __name__ == "__main__":
    ingest_data()

