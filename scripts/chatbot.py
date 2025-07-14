# Updated imports for LangChain components
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings

from langchain_chroma import Chroma # Using langchain_chroma for Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

# Import configuration from your 'source' directory
from config import CHROMA_PERSIST_DIR, OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODEL
import os

class CosmeticChatbot:
    def __init__(self):
        """
        Initializes the chatbot with Ollama LLM, Embedding Model,
        ChromaDB vector store, and LangChain RAG chain.
        """
        print("Initializing Cosmetic Chatbot...")
        
        try:
            # Initialize Ollama LLM for text generation
            self.llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
            print(f"Ollama LLM '{LLM_MODEL}' initialized.")

            # Initialize Ollama Embeddings for vector creation
            self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
            print(f"Ollama Embeddings '{EMBEDDING_MODEL}' initialized.")

            # Check if ChromaDB directory exists before loading
            if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
                raise FileNotFoundError(
                    f"ChromaDB directory not found or empty at '{CHROMA_PERSIST_DIR}'. "
                    "Please run 'python scripts/ingest_data.py' first to populate the vector store."
                )

            # Load existing ChromaDB vector store
            print(f"Loading vector store from {CHROMA_PERSIST_DIR}...")
            self.vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=self.embeddings)
            self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            print("Vector store loaded and retriever configured.")

            # Define the RAG prompt template for product-related questions
            self.rag_prompt_template = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are a helpful AI assistant specialized in cosmetic products and give  guideline treatment for their problems. "
                 "Your goal is to provide accurate and concise information based *only* on the provided context. "),
                ("human", "Context: {context}\nQuestion: {question}")
            ])

            # Define a general prompt template for non-product related questions
            self.general_prompt_template = PromptTemplate.from_template(
                "You are a friendly and helpful beauty consultant. Respond to the user's query naturally. "
                "If the query is a greeting, respond with a friendly greeting back. "
                "If asked 'who are you', introduce yourself as a cosmetic product AI assistant. "
                "User: {query}"
            )

            # Initialize conversation memory
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="result")
            print("Conversation memory initialized.")

            # The RAG chain will be used conditionally
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.rag_prompt_template} # Use the RAG specific prompt
            )
            print("Chatbot initialized successfully.")

        except Exception as e:
            print(f"Error during chatbot initialization: {e}")
            print("Please ensure Ollama is running, models are pulled, and data is ingested.")
            raise

    def _is_general_query(self, query: str) -> bool:
        """
        Simple check to determine if a query is a general greeting or non-product related.
        Could be expanded with more sophisticated intent classification.
        """
        query_lower = query.lower().strip()
        general_phrases = ["hello", "hi", "hey", "how are you", "who are you", "what can you do", "can you help me"]
        for phrase in general_phrases:
            if phrase in query_lower:
                return True
        return False

    def chat(self, user_query: str):
        """
        Processes a user query, retrieves relevant information (conditionally),
        and generates a response from the AI agent.
        """
        # FIX: Removed redundant "User: {user_query}" print here
        # The `input("You: ")` already handles the user's side of the conversation.

        try:
            if self._is_general_query(user_query):
                # For general queries, use the LLM directly with a general prompt
                print(f"You: {user_query}") # Print user query for clarity in output
                general_response = self.llm.invoke(self.general_prompt_template.format(query=user_query))
                print(f"AI Chatbot: {general_response}")
                # No retrieved sources for general queries
                return general_response
            else:
                # For product-related queries, use the RAG chain
                # print(f"You: {user_query}") # Print user query for clarity in output
                response = self.qa_chain.invoke({"query": user_query})

                ai_response = response["result"]
                source_documents = response.get("source_documents", [])

                print(f"AI Chatbot: {ai_response}")

                if source_documents:
                    print("\n--- Retrieved Sources ---")
                    for i, doc in enumerate(source_documents):
                        prod_id = doc.metadata.get('id', 'N/A')
                        prod_name = doc.metadata.get('name', 'N/A')
                        print(f"Source {i+1}: ID={prod_id}, Name='{prod_name}'")
                        print(f"  Content snippet: {doc.page_content[:200]}...") 
                    print("-------------------------")
                return ai_response

        except Exception as e:
            print(f"AI Chatbot: An error occurred during chat: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."

if __name__ == "__main__":
    try:
        chatbot = CosmeticChatbot()
        print("\nHello! I'm your Cosmetic Product AI Assistant. How can I help you today?")
        print("Type 'exit' to quit the chatbot.")

        while True:
            query = input("You: ") # This is the primary user input prompt
            if query.lower() == 'exit':
                print("AI Chatbot: Goodbye! Have a great day.")
                break
            
            chatbot.chat(query)
    except Exception as e:
        print(f"Failed to start the chatbot: {e}")
        print("Please ensure all prerequisites are met and scripts are run in order.")

