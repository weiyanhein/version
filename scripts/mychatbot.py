import os
from pymongo import MongoClient

# LangChain Core components
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough

# LangChain Ollama integrations
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings

# LangChain ChromaDB integration
from langchain_chroma import Chroma

# LangChain Agent components
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationBufferWindowMemory

# Import configuration from your 'source' directory
from config import CHROMA_PERSIST_DIR, OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODEL, DB_NAME, COLLECTION_NAME, MONGO_URI

class CosmeticChatbot:
    def __init__(self):
        """
        Initializes the chatbot with Ollama LLM, Embedding Model,
        ChromaDB vector store, and LangChain Agent.
        """
        print("Initializing Cosmetic Chatbot...")
        
        try:
            # 1. Initialize Ollama LLM and Embeddings
            # RECOMMENDATION: For better agent reasoning, ensure LLM_MODEL is "llama3" (which defaults to 8B)
            # and not a smaller variant like "llama3.2:3b". Run 'ollama pull llama3' if needed.
            self.llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
            self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
            print(f"Ollama LLM '{LLM_MODEL}' and Embeddings '{EMBEDDING_MODEL}' initialized.")

            # 2. Load ChromaDB Vector Store
            if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
                raise FileNotFoundError(
                    f"ChromaDB directory not found or empty at '{CHROMA_PERSIST_DIR}'. "
                    "Please run 'python scripts/ingest_data.py' first to populate the vector store."
                )
            print(f"Loading vector store from {CHROMA_PERSIST_DIR}...")
            self.vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=self.embeddings)
            self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            print("Vector store loaded and retriever configured.")

            # 3. Define Tools for the Agent
            self.product_search_tool = create_retriever_tool(
                self.retriever,
                "product_information_search",
                "Searches for detailed information about cosmetic products based on name, type, brand, ingredients, or benefits. "
                "Use this tool for any questions about specific products, their properties, or usage.",
            )

            # Store tools as a list of actual tool objects
            # self.general_skincare_advice is a method defined below, correctly bound as a tool
            self.tools = [self.product_search_tool, self.general_skincare_advice] 

            # 4. Define the Agent's Prompt
            # FIX: Ensure agent_scratchpad is the last placeholder.
            # LangChain's create_react_agent expects specific placeholders for its internal logic.
            # The order and presence of these placeholders are important.
            self.agent_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are BeautyBuddy, a highly knowledgeable and friendly AI assistant specializing in cosmetic products and general skincare advice. "
                 "Your primary goal is to assist users with their beauty-related questions by either providing information directly from your knowledge or by using the tools available to you. "
                 "When answering, be concise, helpful, and always refer to the context provided by your tools if you use them. "
                 "If a query is a greeting (e.g., 'hello', 'hi'), respond with a friendly greeting and ask how you can help with cosmetic products. "
                 "If asked 'who are you', introduce yourself as BeautyBuddy, a cosmetic product AI assistant. "
                 "If the user asks for general skincare advice (e.g., 'I have dry skin', 'how to reduce dark spots'), use the 'general_skincare_advice' tool. "
                 "For any questions about specific products (e.g., 'benefits of X', 'ingredients in Y', 'usage of Z'), use the 'product_information_search' tool. "
                 "If you cannot find relevant information using your tools, state that you don't have enough information in your database."
                 "\n\nAvailable tools: {tools}" # Provided by AgentExecutor
                 "\nTool names: {tool_names}"   # Provided by AgentExecutor
                 ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"), # This must be the last message placeholder
            ])

            # 5. Initialize Conversation Memory for the Agent
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history", 
                return_messages=True, 
                output_key="output", # AgentExecutor's output key is usually 'output'
                k=5 # Keep last 5 turns of conversation
            )
            print("Conversation memory initialized.")

            # 6. Create the Agent
            self.agent = create_react_agent(self.llm, self.tools, self.agent_prompt)

            # 7. Create the Agent Executor
            self.agent_executor = AgentExecutor(
                agent=self.agent, 
                tools=self.tools, 
                memory=self.memory, 
                verbose=True, # Set to True to see the agent's thought process
                handle_parsing_errors=True # Helps with debugging agent errors
            )
            print("Chatbot initialized successfully with Agent architecture.")

        except Exception as e:
            print(f"Error during chatbot initialization: {e}")
            print("Please ensure Ollama is running, models are pulled, and data is ingested.")
            raise

    # Tool 2: General Skincare Advice (defined as a method of the class)
    @tool
    def general_skincare_advice(self, query: str) -> str:
        """
        Provides general skincare advice or recommendations for common skin concerns (e.g., dry skin, dark spots).
        Do NOT use this for specific product information; use product_information_search instead.
        """
        # This is a simplified direct LLM call for general advice.
        # In a real system, this might involve another RAG source,
        # a more complex chain, or even a different LLM.
        advice_prompt = PromptTemplate.from_template(
            "You are a helpful skincare advisor. Provide general, non-product-specific advice for the following concern: {query}. "
            "If the query is about specific products, state that this tool is for general advice and suggest using product_information_search."
        )
        return self.llm.invoke(advice_prompt.format(query=query))


    def chat(self, user_query: str):
        """
        Processes a user query using the AI agent.
        """
        print(f"You: {user_query}")

        try:
            # The agent_executor expects 'input' and 'chat_history' (from memory)
            # It will manage the agent_scratchpad internally.
            response = self.agent_executor.invoke({"input": user_query})

            ai_response = response["output"]

            print(f"AI Chatbot: {ai_response}")
            
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
            query = input("You: ")
            if query.lower() == 'exit':
                print("AI Chatbot: Goodbye! Have a great day.")
                break
            
            chatbot.chat(query)
    except Exception as e:
        print(f"Failed to start the chatbot: {e}")
        print("Please ensure all prerequisites are met and scripts are run in order.")

