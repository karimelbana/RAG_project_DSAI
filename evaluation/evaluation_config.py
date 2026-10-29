from pathlib import Path

from ragas.metrics.base import Metric
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,

)

# --- LLM Model Configuration ---

# --- LOCAL LLM ---

EVALUATION_LLM_BACKEND = "ollama"

EVALUATION_OLLAMA_MODEL = "llama3.2:3b" #ollama pull llama3.2:3b

# Context window for the local Ollama model. Must comfortably fit the system
# prompt + retrieved chunks + conversation history + the generated answer,
# otherwise Ollama silently truncates the prompt and answer quality drops.
OLLAMA_CONTEXT_WINDOW: int = 8192

# A local 12B model can take ~2 minutes for a long, non-streamed answer, so the
# timeout is generous. The web UI streams, so users see tokens long before this.
LLM_REQUEST_TIMEOUT: float = 300.0

# Low temperature keeps answers factual and grounded in the sources.
LLM_TEMPERATURE: float = 0.01

# --- GROQ LLM ---

# EVALUATION_LLM_MODEL: str = "moonshotai/kimi-k2-instruct-0905"
EVALUATION_GROQ_MODEL = "qwen/qwen3-32b"

# --- Embedding Model Configuration ---
EVALUATION_EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"

# --- Paths for Evaluation ---
EVALUATION_ROOT_PATH: Path = Path(__file__).parent
EVALUATION_RESULTS_PATH: Path = EVALUATION_ROOT_PATH / "evaluation_results/"
EXPERIMENTAL_VECTOR_STORES_PATH: Path = (
    EVALUATION_ROOT_PATH
    / "evaluation_vector_stores/"
)
EVALUATION_EMBEDDING_CACHE_PATH: Path = (
    EVALUATION_ROOT_PATH
    / "evaluation_embedding_models/"
)

# --- Ragas Evaluation Metrics ---
EVALUATION_METRICS: list[Metric] = [
    # Faithfulness(),
    # AnswerCorrectness(),
    # ContextPrecision(),
    ContextRecall(),
]

# --- Sleep Timers for API Limits ---
SLEEP_PER_EVALUATION: int = 6
SLEEP_PER_QUESTION: int = 6

# --- Configuration for Chunking Strategy Evaluation ---
CHUNKING_STRATEGY_CONFIGS: list[dict[str, int]] = [
    {'size': 512, 'overlap': 50},
    {'size': 768, 'overlap': 115},
   #  {'size': 1024, 'overlap': 200},
]

# --- Cross-encoder Model for Reranking ---
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Configuration for Reranking Evaluation ---
RERANKER_CONFIGS: list[dict[str, int]] = [
    {'retriever_k': 10, 'reranker_n': 2},
    {'retriever_k': 10, 'reranker_n': 5},
    {'retriever_k': 20, 'reranker_n': 5},
]

# --- Query Rewrite Evaluation ---
# The 'best' reranker strategy found in the previous evaluation stage.
# IMPORTANT: You must update this with the values you found to be optimal.
BEST_RERANKER_STRATEGY: dict[str, int] = {'retriever_k': 10, 'reranker_n': 2}
