import os
import sys

# Add the project root to sys path so we can import modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.rag_agent.rag_agent import RAGAgent

def test_agent():
    print("Initializing RAG Agent...")
    # Adjust path if running from root or elsewhere. 
    # Assuming running from root: agents/rag-agent/data
    # Assuming running from agents/rag-agent: ./data
    
    # We'll use absolute path to be safe or relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    agent = RAGAgent(data_dir=data_dir)
    
    # Check if we assume Milvus is running.
    # In a real test we might mock, but this is an integration test.
    
    # Trigger ingestion (idempotency checks aren't robust in the simple class)
    # Uncomment the following lines to ingest data if running for the first time.
    # print("Ingesting data...")
    # try:
    #     agent.ingest_data()
    # except Exception as e:
    #     print(f"Ingestion failed (maybe Milvus isn't up?): {e}")
    #     return

    questions = [
        "What is the leave policy for marriage?",
        "How much travel allowance do I get?",
        "Tell me about the sexual harassment policy."
    ]

    for q in questions:
        print(f"\n--- Question: {q} ---")
        try:
            answer = agent.query(q)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Query failed: {e}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    test_agent()
