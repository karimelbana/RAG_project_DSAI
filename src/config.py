from pathlib import Path

# --- LLM Model Configuration ---
# LLM_BACKEND environment variable, e.g. LLM_BACKEND=groq python main.py
LLM_BACKEND = "ollama" # or "groq"

# --- Local LLM ---

# Local Ollama model. gemma4:e2b fully fits the 8 GB GPU and is far faster than
# gemma4:12b (which spills to CPU and times out on long answers here), at equal
# quality on the eval questions. Swap with `ollama pull <tag>` then change here.
OLLAMA_MODEL: str = "gemma3:4b" # "gemma4:e2b"

# Context window for the local Ollama model. Must comfortably fit the system
# prompt + retrieved chunks + conversation history + the generated answer,
# otherwise Ollama silently truncates the prompt and answer quality drops.
OLLAMA_CONTEXT_WINDOW: int = 8192

# A local 12B model can take ~2 minutes for a long, non-streamed answer, so the
# timeout is generous. The web UI streams, so users see tokens long before this.
LLM_REQUEST_TIMEOUT: float = 300.0

# --- GROQ LLM ---

GROQ_MODEL: str = "llama-3.3-70b-versatile"
LLM_MAX_NEW_TOKENS: int = 768
# Low temperature keeps answers factual and grounded in the sources.
LLM_TEMPERATURE: float = 0.01
LLM_TOP_P: float = 0.95
LLM_REPETITION_PENALTY: float = 1.03
#LLM_QUESTION: str = "What is the capital of Paris?"
LLM_SYSTEM_PROMPT: str = (
    "You are a helpful chatbot. Be friendly and conversational."
)


# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# --- RAG/VectorStore Configuration ---
# The number of most relevant text chunks to retrieve from the vector store
SIMILARITY_TOP_K: int = 3
# The size of each text chunk in tokens
CHUNK_SIZE: int = 512
# The overlap between adjacent text chunks in tokens
CHUNK_OVERLAP: int = 50

# --- Chat Memory Configuration ---
CHAT_MEMORY_TOKEN_LIMIT: int = 3900


# --- Persistent Storage Paths (using pathlib for robust path handling) ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage/embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage/vector_store/"

# --- Reranker Configuration ---
RERANKER_TOP_N = 3  # For example, The number of nodes to return after reranking
# Different from embedding model, reranker uses question and document as input and directly output similarity instead of embedding
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base" # Or another cross-encoder model
