import os
import glob
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class RAGAgent:
    def __init__(self, 
                 milvus_host: str = "localhost", 
                 milvus_port: str = "19530",
                 collection_name: str = "hr_policy_rag",
                 data_dir: str = "./data"):
        
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.data_dir = data_dir
        
        # Initialize Embeddings
        # Verify OPENAI_API_KEY is set in environment
        if not os.getenv("OPENAI_API_KEY"):
            print("WARNING: OPENAI_API_KEY not found in environment variables.")
        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=1, model_name="gpt-5-mini-2025-08-07")
        
        # Initialize Vector Store (Lazy load or connect)
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self) -> Milvus:
        """
        Connects to Milvus. If collection exists, loads it.
        If not, loads documents, splits them, and creates the collection.
        This provides a basic check. For production, more robust migration logic is needed.
        """
        # Check if we should re-index or just connect
        # For simplicity, we'll assume if we can connect and search, it's good.
        # But here we probably want to ensure data is loaded at least once.
        
        # Note: Langchain's Milvus wrapper handles connection internally upon instantiation
        # based on connection_args.
        
        # Strategy: Try to connect to an existing collection. 
        # If it's empty or doesn't exist, index the data.
        
        # We really need to know if we've indexed.
        # For this agent, we'll try to load and if the collection seems new, we ingest.
        # A simple way with LangChain is just to use `from_documents` if we are initializing,
        # or just the constructor if we are connecting.
        
        # Since checking existence efficiently via Langchain wrapper is tricky without
        # direct pymilvus usage, we will try to connect. 
        
        vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={"host": self.milvus_host, "port": self.milvus_port}
        )
        
        # Check if collection is empty (simplified check)
        # In a real scenario, we might want persistent state or a check file.
        # Here, let's just create a method `ingest_data` that the user calls manually 
        # or we call if we detect 0 docs (if possible). 
        # For now, let's just return the store and rely on explicit ingestion or 
        # check if we can simply "add" if empty.
        
        return vector_store

    def ingest_data(self):
        """
        Loads PDFs from data_dir, splits them, and adds to vector store.
        """
        print(f"Loading data from {self.data_dir}...")
        pdf_files = glob.glob(os.path.join(self.data_dir, "*.pdf"))
        
        if not pdf_files:
            print("No PDF files found.")
            return

        documents = []
        for pdf_file in pdf_files:
            print(f"Loading {pdf_file}...")
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)
            
        print(f"Loaded {len(documents)} document pages.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")
        
        print("Indexing into Milvus with batching to avoid rate limits...")
        batch_size = 10
        import time
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"Indexing batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}...")
            self.vector_store.add_documents(batch)
            time.sleep(2) # Sleep to respect rate limits
            
        print("Ingestion complete.")

    def query(self, user_query: str) -> str:
        """
        Retrieves relevant documents and generates a response.
        """
        print(f"Querying: {user_query}")
        
        # Retrieve
        docs = self.vector_store.similarity_search(user_query, k=3)
        
        if not docs:
            return "I couldn't find any relevant information in the policy documents."
            
        # Context
        context = "\n\n".join([d.page_content for d in docs])
        
        # Generate
        # Simple prompt construction
        prompt = f"""You are an HR Policy Assistant. Use the following context to answer the user's question.
If the answer is not in the context, just say you don't know in a polite manner. Do not disclose your sources

Context:
{context}

Question: {user_query}

Answer:"""

        response = self.llm.invoke(prompt)
        return response.content

# Example usage (if run directly)
if __name__ == "__main__":
    # Ensure env vars are set or load .env
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = RAGAgent(milvus_host="localhost", milvus_port="19530", data_dir="./data")
    
    # Uncomment to ingest data (run once)
    # agent.ingest_data()
    
    # Test query
    # print(agent.query("What is the policy for annual leave?"))
