from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from databricks.vector_search.client import VectorSearchClient
from langchain_core.retrievers import BaseRetriever, Document
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import torch
import os

class ChatTemplateHuggingFacePipeline(HuggingFacePipeline):
    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return super()._call(formatted_prompt, stop=stop, run_manager=run_manager, **kwargs)

embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-base-en-v1.5'
)

model_name = r"[model path or hf url]"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="xpu",
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# "pipeline" - tokenization, generations, and decoding in one object
pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    return_full_text=False,
)

# langchain wrapper for pipeline
llm = ChatTemplateHuggingFacePipeline(pipeline=pipe)

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
    
retriever = DatabricksRetriever(
    vsc=vsc,
    endpoint_name="[endpoint name]",
    index_name="[index name]",
    embeddings=embeddings,
    k = 5
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an assisstant that answers questions based on project documentation. 

    Critical Instructions:
    - If the answer is not in the context, reply with "I need more information to answer that question."
    - Answer only the user's question. Do not generate follow-up questions on your own.
    
    Wait for the user to ask the next question."""),
    ("placeholder", "{chat_history}"),
    ("human", """Context:
    {context}

    Question:
    {input}""")
])

# stuff puts all the docs into the context
combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt_template
)

qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain
)

# stores message history
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
    
# wrap the chain with message history
qa_chain_with_history = RunnableWithMessageHistory(
    qa_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

def chat_loop():
    session_id = "default_session"
    MAX_TURNS = 15

    print("\nRAG Chatbot with Memory")
    print(f"Note that the conversations are limited to {MAX_TURNS} exchanges")
    print("Type 'quit' to end the conversation")
    print("Type 'new' to start a fresh conversation")
    print("Type 'history' to see conversation history")

    while True:

        current_history = get_session_history(session_id)
        turn_count = len(current_history.messages) // 2

        if turn_count >= MAX_TURNS:
            print(f"Conversation limit reached ({MAX_TURNS} exchanges)")
            print("Starting a new conversation to maintain quality")
            current_history.clear()
            turn_count = 0

        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break

        if user_input.lower() == 'new':
            get_session_history(session_id).clear()
            print("\nStarted a new conversation")
            continue

        if user_input.lower() =='history':
            history = get_session_history(session_id)
            print("\nConversation History")
            if history.messages:
                for msg in history.messages:
                    role = "USER" if msg.type == "human" else "ASSISTANT"
                    print(f"{role}: {msg.content}")
            else:
                print("No converstation history yet.")
            continue

        if not user_input:
            continue

        turn_count += 1
        print(f"[Turn {turn_count}/{MAX_TURNS}]", end=" ")

        try:
            result = qa_chain_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            answer = result["answer"]

            print(f"\nAssistant: {answer}")

        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")

if __name__ == "__main__":
    chat_loop()