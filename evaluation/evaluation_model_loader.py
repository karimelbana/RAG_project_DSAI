import os
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms.base import LlamaIndexLLMWrapper
from llama_index.llms.ollama import Ollama

from evaluation.evaluation_config import (
    EVALUATION_EMBEDDING_MODEL_NAME,
    EVALUATION_EMBEDDING_CACHE_PATH,
    EVALUATION_GROQ_MODEL,
    EVALUATION_LLM_BACKEND,
    EVALUATION_OLLAMA_MODEL,
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
    OLLAMA_CONTEXT_WINDOW
)


# Load environment variables from the .env file
load_dotenv()


def initialise_evaluation_llm() -> Groq:
    """Initialises the Groq LLM with core parameters from config."""

    if EVALUATION_LLM_BACKEND == "groq":

        api_key: str | None = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Make sure it's set in your .env file."
            )

        return Groq(
            api_key=api_key,
            model=EVALUATION_GROQ_MODEL,
        )

    # Default: local Ollama. Requires the Ollama app running and the model
    # pulled (`ollama pull gemma3:4b`) gemma4:e2b
    print(f'Loading EVALUATION local LLM: {EVALUATION_OLLAMA_MODEL} through Ollama 🏠')
    return Ollama(
        model=EVALUATION_OLLAMA_MODEL,
        request_timeout=LLM_REQUEST_TIMEOUT,
        temperature=LLM_TEMPERATURE,
        context_window=OLLAMA_CONTEXT_WINDOW)


def load_ragas_models(
) -> tuple[LlamaIndexLLMWrapper, HuggingFaceEmbeddings]:
    """
    Loads the LLM and embedding models
    required for Ragas evaluation.
    """
    print("--- 🧠 Loading Ragas LLM and Embeddings ---")

    llm_for_evaluation: Groq = initialise_evaluation_llm()

    # Wrap the LlamaIndex LLM for compatibility with Ragas
    ragas_llm = LlamaIndexLLMWrapper(llm=llm_for_evaluation)

    # Initialise the embedding model Ragas will use for its metrics
    ragas_embeddings = HuggingFaceEmbeddings(
        model=EVALUATION_EMBEDDING_MODEL_NAME,
        cache_folder=EVALUATION_EMBEDDING_CACHE_PATH.as_posix()
    )

    return ragas_llm, ragas_embeddings
