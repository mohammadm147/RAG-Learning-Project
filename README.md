# RAG Learning Project

A Retrieval-Augmented Generation (RAG) chatbot implementation that answers questions based on project documentation using local LLMs and vector search.

## Overview

This project demonstrates a complete RAG pipeline that processes documents, embeds them into a vector database, and uses an intelligent agent to answer questions based on retrieved context. The system runs entirely locally using open-source models and includes conversation memory for contextual multi-turn interactions.

## Features

- **Document Processing**: Automatically chunks and embeds documents for efficient retrieval
- **Vector Search**: Integrates with Databricks Vector Search for semantic similarity matching
- **Local LLM Integration**: Multiple inference options for flexibility
- **Multiple Implementation Approaches**:
  - `pipeline.py`: Traditional RAG chain with direct HuggingFace model loading
  - `agent_code.py`: Agent-based approach using llama.cpp server with OpenAI-compatible API
- **Conversation Memory**: Maintains chat history for contextual responses
- **Flexible Model Support**: Works with both HuggingFace models (pipeline.py) and quantized GGUF models via llama.cpp (agent_code.py)

## Technologies Used

### Core Frameworks
- **LangChain**: Orchestrates the RAG pipeline and manages retrievers, chains, and agents
- **LangGraph**: Enables agent workflows with checkpointing and memory
- **HuggingFace Transformers**: Powers local model inference
- **Sentence Transformers**: Generates document embeddings

### Vector Database
- **Databricks Vector Search**: Cloud-based vector database for similarity search

### Language Models
- **Llama 3.1 8B Instruct**: Primary instruction-following model
- **Hermes-4 14B**: Alternative larger model for improved responses
- **BGE-Base-EN-v1.5**: Embedding model for document and query vectorization

### Additional Tools
- **llama.cpp**: Efficient CPU/GPU inference for quantized GGUF models
- **PyTorch**: Deep learning framework for model execution
- **Intel XPU**: Hardware acceleration support

## Project Structure

```
├── agent_code.py              # Agent-based RAG implementation with tools
├── pipeline.py                # Traditional RAG chain implementation
├── chunk_and_embed.py         # Document preprocessing and embedding
├── start_server.py            # llama.cpp server launcher
├── Documents/                 # Store source documents here for the knowledge base
```

## How It Works

### 1. Document Processing (`chunk_and_embed.py`)
- Reads all text files from the `Documents/` directory
- Splits documents into manageable chunks (750 words each with 150-word overlap)
- Generates embeddings using BGE-Base-EN-v1.5
- Saves embedded chunks to `embedded_documents.pkl`

### 2. Vector Search Setup
- Embedded documents are uploaded to Databricks Vector Search
- Creates a searchable index for semantic similarity queries
- Retrieves top-k most relevant chunks for user queries

### 3. RAG Pipeline

#### Traditional Chain Approach (`pipeline.py`)
1. Loads HuggingFace model directly into memory using PyTorch
2. User submits a question
3. Question is embedded and searched against vector database
4. Top 5 relevant chunks are retrieved
5. LLM generates answer using retrieved context
6. Conversation history is maintained for follow-up questions

#### Agent Approach (`agent_code.py`)
1. Connects to llama.cpp server via OpenAI-compatible API
2. User submits a question
3. Agent decides whether to use the document search tool
4. Retrieves relevant documents if needed
5. Generates response with LangGraph state management
6. Checkpointing allows for more complex conversation flows

### 4. Local LLM Server (`start_server.py`)
- **Only required for `agent_code.py`**
- Launches llama.cpp server in a new PowerShell console
- Exposes OpenAI-compatible API on `localhost:8000`
- Loads quantized GGUF models for efficient inference
- Supports 16K context window with GPU acceleration

## Setup and Installation

### Prerequisites
- Python 3.8+
- Conda (recommended for environment management)
- llama.cpp binaries
- GPU with Intel XPU support (optional but recommended)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RAG-Learning-Project
   ```

2. **Create a virtual environment**
   ```bash
   conda create -n rag-project python=3.10
   conda activate rag-project
   ```

3. **Install dependencies**
   ```bash
   pip install langchain langchain-huggingface langchain-openai
   pip install transformers sentence-transformers
   pip install databricks-vector-search
   pip install python-dotenv torch
   pip install langgraph
   ```

4. **Download models**
   - Download Llama 3.1 8B Instruct from HuggingFace
   - Download Hermes-4 GGUF models from HuggingFace
   - Place models in respective directories

5. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   DATABRICKS_HOST=<your-databricks-workspace-url>
   DATABRICKS_CLIENT_ID=<your-service-principal-client-id>
   DATABRICKS_CLIENT_SECRET=<your-service-principal-secret>
   ```

6. **Prepare documents**
   - Place your text documents in the `Documents/` folder
   - Run the embedding script:
     ```bash
     python chunk_and_embed.py
     ```

7. **Upload embeddings to Databricks**
   - Use the Databricks UI or API to create a vector search index
   - Upload the embedded documents from `embedded_documents.pkl`

## Running the Project

### Option 1: Using the Traditional RAG Chain (Direct Model Loading)

1. **Run the pipeline directly**
   ```bash
   python pipeline.py
   ```
   *Note: This loads the HuggingFace model directly in memory - no separate server needed*

2. **Interact with the chatbot**
   - Type your questions about the documents
   - Use `history` to view conversation history
   - Use `new` to start a fresh conversation
   - Use `quit` to exit

### Option 2: Using the Agent-Based Approach (llama.cpp Server)

1. **Start the LLM server**
   ```bash
   python start_server.py
   ```
   *This launches llama.cpp server with a quantized GGUF model*

2. **Run the agent** (in a separate terminal)
   ```bash
   python agent_code.py
   ```

3. **Interact with the agent**
   - The agent will automatically decide when to search documents
   - Same commands as the pipeline approach

## Configuration

### Adjusting Chunk Size
In `chunk_and_embed.py`:
```python
chunks = chunk(text, chunk_size=3750, chunk_overlap=750)
```

### Changing Model Parameters
In `pipeline.py`:
```python
pipe = pipeline(
    "text-generation",
    max_new_tokens=512,    # Maximum response length
    temperature=0.7,        # Creativity (0.0-1.0)
    do_sample=True,        # Enable sampling
)
```

### Adjusting Retrieval
```python
retriever = DatabricksRetriever(
    k=5  # Number of chunks to retrieve
)
```

## Limitations and Notes

- Conversations are limited to 15 turns to maintain quality and prevent context overflow
- The system is designed for document-based Q&A and may not perform well on general knowledge questions
- Performance depends on the quality and relevance of documents in the knowledge base
- GPU acceleration significantly improves inference speed

## Future Improvements

- [ ] Add support for more document formats (PDF, DOCX, etc.)
- [ ] Implement semantic caching for frequently asked questions
- [ ] Add evaluation metrics for RAG performance
- [ ] Create a web interface for easier interaction
- [ ] Support for multi-modal documents (images, tables)
- [ ] Implement re-ranking for improved retrieval quality

## License

This project is provided as-is for learning and educational purposes.

## Acknowledgments

- LangChain for the RAG framework
- HuggingFace for model hosting and transformers library
- llama.cpp for efficient local inference
- Databricks for vector search capabilities
