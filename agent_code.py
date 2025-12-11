from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from databricks.vector_search.client import VectorSearchClient
from langchain_core.retrievers import BaseRetriever, Document
from langchain_core.documents import Document
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-base-en-v1.5'
)

load_dotenv()
vsc = VectorSearchClient(
    workspace_url=os.getenv("DATABRICKS_HOST"),
    service_principal_client_id=os.getenv("DATABRICKS_CLIENT_ID"),
    service_principal_client_secret=os.getenv("DATABRICKS_CLIENT_SECRET")
)

# langchain retreiver that replaced ragquery in manual implementation
class DatabricksRetriever(BaseRetriever):
    vsc: VectorSearchClient
    index_name: str
    endpoint_name: str
    embeddings: HuggingFaceEmbeddings
    k: int = 5

    # embed query, search vector db, and return top k results
    def _get_relevant_documents(self, query: str) -> list[Document]:
        index = self.vsc.get_index(
            endpoint_name=self.endpoint_name,
            index_name=self.index_name
        )

        query_embedding = self.embeddings.embed_query(query)

        results = index.similarity_search(
            query_vector=query_embedding,
            columns=["text", "source", "chunk_id"],
            num_results=self.k
        )

        # results need to be langchain documents
        docs = []
        for row in results["result"]["data_array"]:
            docs.append(Document(
                page_content=row[0],
                metadata={
                    "source": row[1],
                    "chunk_id": row[2],
                    "score": row[3]
                }
            ))

        return docs
    
    def get_relevant_documents(self, query: str) -> list[Document]:
        return self._get_relevant_documents(query)

    
retriever = DatabricksRetriever(
    vsc=vsc,
    endpoint_name="[endpoint name]",
    index_name="[index name]",
    embeddings=embeddings,
    k = 5
)

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="local",
    temperature=0.7,
    max_tokens=1024,
)

# wraps extisting retreiver as a tool. will return top 5 relevant chunks
@tool
def search_documents(query: str) -> str:
    """Search through project documentation to find relevant information.
    
    Args:
        query: The search query to find relevant documents
        
    Returns:
        Formatted results with content, source, and chunk IDs
    """
    docs = retriever.get_relevant_documents(query)

    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"Result {i}:")
        results.append(f"Content: {doc.page_content}")
        results.append(f"Source: {doc.metadata.get('source', 'Unknown')}")
        results.append(f"Chunk ID: {doc.metadata.get('chunk_id', 'Unknown')}")
        results.append("---")
    
    return "\n".join(results)

tools = [search_documents]
memory = MemorySaver()

system_prompt = """You are a helpful assistant that answers questions about project documentation.
When you need information, use the search_documents tool to find relevant content."""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=memory
)
def chat_loop():
    session_id = "default_session"
    MAX_TURNS = 15
    turn_count = 0
    print("\nRAG Chatbot with Agent and Memory")
    print(f"Note that the conversations are limited to {MAX_TURNS} exchanges")
    print("Type 'quit' to end the conversation")
    print("Type 'new' to start a fresh conversation")
    print("Type 'history' to see conversation history")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break
            
        if user_input.lower() == 'new':
            session_id = f"session_{os.urandom(4).hex()}"
            turn_count = 0
            print("\nStarted new conversation\n")
            continue
        
        if user_input.lower() == 'history':
            config = {"configurable": {"thread_id": session_id}}
            try:
                state = agent.get_state(config)
                messages = state.values.get("messages", [])
                
                if not messages:
                    print("\nNo conversation history yet.\n")
                else:
                    print("\nConversation History")
                    for msg in messages:
                        role = msg.type.upper() if hasattr(msg, 'type') else "UNKNOWN"
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        if len(content) > 200:
                            content = content[:200] + "..."
                        print(f"{role}: {content}")
                    print(f"\n[Turn {turn_count}/{MAX_TURNS}]\n")
            except Exception as e:
                print(f"\nCouldn't retrieve history: {e}\n")
            continue
            
        if not user_input:
            continue
        
        if turn_count >= MAX_TURNS:
            print(f"\nConversation limit reached ({MAX_TURNS} exchanges)")
            print("Starting a new conversation to maintain quality...")
            session_id = f"session_{os.urandom(4).hex()}"
            turn_count = 0
            print()
        
        turn_count += 1
        print(f"[Turn {turn_count}/{MAX_TURNS}]\n")
        
        try:
            config = {"configurable": {"thread_id": session_id}}
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config
            )
            
            answer = response["messages"][-1].content
            print(f"Assistant: {answer}\n")
            
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")
            turn_count -= 1

if __name__ == "__main__":
    chat_loop()