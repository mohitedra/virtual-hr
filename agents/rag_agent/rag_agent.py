"""
RAG Agent for the Virtual HR system.
Answers policy questions using document retrieval from Milvus and Claude for response generation.
"""
import os
import glob
import sys
from typing import List, Optional

# Add parent path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from anthropic import Anthropic
from config import Config


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
        
        # Initialize Embeddings (still using OpenAI for embeddings)
        if not Config.OPENAI_API_KEY:
            print("WARNING: OPENAI_API_KEY not found in environment variables.")
        
        self.embeddings = OpenAIEmbeddings(
            model=Config.OPENAI_EMBEDDING_MODEL,
            api_key=Config.OPENAI_API_KEY
        )
        
        # Initialize Claude for response generation
        if not Config.ANTHROPIC_API_KEY:
            print("WARNING: ANTHROPIC_API_KEY not found in environment variables.")
        
        self.claude = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.claude_model = Config.CLAUDE_MODEL
        
        # Initialize Vector Store (Lazy load or connect)
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self) -> Milvus:
        """
        Connects to Milvus. If collection exists, loads it.
        If not, loads documents, splits them, and creates the collection.
        """
        vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={"host": self.milvus_host, "port": self.milvus_port}
        )
        
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
            time.sleep(2)  # Sleep to respect rate limits
            
        print("Ingestion complete.")

    def query(self, user_query: str) -> str:
        """
        Retrieves relevant documents and generates a response using Claude.
        """
        print(f"Querying: {user_query}")
        
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(user_query, k=3)
        
        if not docs:
            return "I couldn't find any relevant information in the policy documents."
            
        # Build context from retrieved documents
        context = "\n\n".join([d.page_content for d in docs])
        
        # Generate response using Claude
        prompt = f"""You are an HR Policy Assistant. Use the following context to answer the user's question.
If the answer is not in the context, just say you don't know in a polite manner. Do not disclose your sources.

Context:
{context}

Question: {user_query}"""

        response = self.claude.messages.create(
            model=self.claude_model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text


# Example usage (if run directly)
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = RAGAgent(milvus_host="localhost", milvus_port="19530", data_dir="./data")
    
    # Uncomment to ingest data (run once)
    agent.ingest_data()
    
    # Test query
    # print(agent.query("What is the policy for annual leave?"))
