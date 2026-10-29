import os

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama


from src.config import (
    EMBEDDING_CACHE_PATH,
    # LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    # LLM_TOP_P,
    # LLM_REPETITION_PENALTY
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL,
    OLLAMA_CONTEXT_WINDOW,
    OLLAMA_MODEL,
    LLM_BACKEND,
    LLM_REQUEST_TIMEOUT)

# Load environment variables from the .env file
load_dotenv()

def initialise_llm() -> Groq:
    """Initialises the Groq LLM with core parameters from config."""

    api_key: str | None = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Make sure it's set in your .env file."
        )

    # return Groq(
    #     api_key=api_key,
    #     model=LLM_MODEL,
    #     # The following parameters are optional
    #     # and will default to the model's defaults if not set
    #     # max_tokens=LLM_MAX_NEW_TOKENS,
    #     # temperature=LLM_TEMPERATURE,
    #     # top_p=LLM_TOP_P,
    # )

    if LLM_BACKEND == "groq":

        api_key: str | None = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_BACKEND is 'groq' but GROQ_API_KEY is not set. "
                "Add it to your .env file or switch to LLM_BACKEND=ollama."
            )
        return Groq(
            api_key=api_key,
            model=GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
        )

    # Default: local Ollama. Requires the Ollama app running and the model
    # pulled (`ollama pull gemma3:4b`) gemma4:e2b
    print(f'Loading local LLM: {OLLAMA_MODEL} through Ollama 🏠')
    return Ollama(
        model=OLLAMA_MODEL,
        request_timeout=LLM_REQUEST_TIMEOUT,
        temperature=LLM_TEMPERATURE,
        context_window=OLLAMA_CONTEXT_WINDOW)


def get_embedding_model() -> HuggingFaceEmbedding:
    """Initialises and returns the HuggingFace embedding model"""

    # Create the cache directory if it doesn't exist
    EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix(),
    )

if __name__ == '__main__':
    print('this is the model loader file ⚠️')
    print(initialise_llm())
