from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.chat_engine import SimpleChatEngine

from src.config import (
    CHAT_MEMORY_TOKEN_LIMIT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    LLM_SYSTEM_PROMPT,
    SIMILARITY_TOP_K,
    VECTOR_STORE_PATH,
    RERANKER_TOP_N,
    RERANKER_MODEL_NAME,
)

from src.model_loader import initialise_llm, get_embedding_model#  , initialise_hyde_llm

# for HyDE
# from llama_index.core.query_engine import TransformQueryEngine, TransformRetriever  # <-- Add this line
# from llama_index.core.indices.query.query_transform import HyDEQueryTransform # <-- Add this line


def _create_new_vector_store(
        embed_model: HuggingFaceEmbedding
) -> VectorStoreIndex:
    """Creates, saves, and returns a new vector store from documents."""

    print(
        "Creating new vector store from all files in the 'data' directory..."
    )

    # This reads all the text files in the specified directory.
    documents: list[Document] = SimpleDirectoryReader(
        input_dir=DATA_PATH,
    ).load_data()

    if not documents:
        raise ValueError(
            f"No documents found in {DATA_PATH}. Cannot create vector store."
        )

    # This breaks the long document into smaller, more manageable chunks.
    text_splitter: SentenceSplitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # This is the core of the vector store. It takes the text chunks,
    # uses the embedding model to convert them to vectors, and stores them.
    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter],
        embed_model=embed_model,
    )

    # This saves the newly created index to disk for future use.
    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())
    print("Vector store created and saved.")
    return index


def get_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Loads the vector store from disk if it exists;
    otherwise, creates a new one.
    """

    # Create the parent directory if it doesn't exist.
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    # Check if the directory contains any files.
    if any(VECTOR_STORE_PATH.iterdir()):
        print("Loading existing vector store from disk...")
        storage_context: StorageContext = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix(),
        )
        # We must provide the embed_model when loading the index.
        return load_index_from_storage(
            storage_context,
            embed_model=embed_model
        )
    else:
        # If the directory is empty,
        # call our internal function to build the index.
        return _create_new_vector_store(embed_model)

def get_chat_engine(
        llm: Groq,
        embed_model: HuggingFaceEmbedding
) -> BaseChatEngine:
    """Initialises and returns the main conversational RAG chat engine."""

    vector_index = get_vector_store(embed_model)
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT
    )

    # Assemble the RAG chat engine
    chat_engine = vector_index.as_chat_engine(
        memory=memory,
        llm=llm,
        system_prompt=LLM_SYSTEM_PROMPT,
        similarity_top_k=SIMILARITY_TOP_K,
    )
    return chat_engine

# def get_chat_engine(
#         llm: Groq,
#         embed_model: HuggingFaceEmbedding
# ) -> CondensePlusContextChatEngine:
#     """Initialises and returns the main conversational RAG chat engine."""
#     vector_index: VectorStoreIndex = get_vector_store(embed_model)

#     # 1. Create the Retriever
#     retriever = vector_index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)

#     # 2. Setup the Reranker
#     reranker: SentenceTransformerRerank = SentenceTransformerRerank(
#         top_n=RERANKER_TOP_N,
#         model=RERANKER_MODEL_NAME,
#     )

#     # 3. Setup Memory
#     memory: ChatMemoryBuffer = ChatMemoryBuffer.from_defaults(
#         token_limit=CHAT_MEMORY_TOKEN_LIMIT,
#     )

#     # 4. Assemble the advanced Chat Engine
#     # 'contextual amnesia'
#     chat_engine = CondensePlusContextChatEngine(
#         retriever=retriever,
#         memory=memory,
#         llm=llm,
#         system_prompt=LLM_SYSTEM_PROMPT,
#         node_postprocessors=[reranker]
#     )

#     return chat_engine

#commented out recently
# def get_chat_engine(
#     llm: Groq,
#     embed_model: HuggingFaceEmbedding
# ) -> CondensePlusContextChatEngine:
#     """Initialises and returns the main conversational RAG chat engine with HyDE."""

#     vector_index: VectorStoreIndex = get_vector_store(embed_model)

#     base_retriever: BaseRetriever = vector_index.as_retriever(
#         similarity_top_k=SIMILARITY_TOP_K
#     )

#     # 'include_original=True' ensures the system searches with BOTH
#     # the hallucinated answer and the user's real question.
#     hyde = HyDEQueryTransform(
#         include_original=True,
#         llm=llm
#     )

#     # Now, any query passed to hyde_retriever is first 'imagined' by HyDE.
#     hyde_retriever = TransformRetriever(
#         retriever=base_retriever,
#         query_transform=hyde
#     )

#     reranker = SentenceTransformerRerank(
#         top_n=RERANKER_TOP_N,
#         model=RERANKER_MODEL_NAME
#     )


#     memory = ChatSummaryMemoryBuffer.from_defaults(
#         token_limit=CHAT_MEMORY_TOKEN_LIMIT,
#         llm=llm # potentially you can change this to initialise_hyde_llm() for faster retrieval
#     )

    # CondensePlusContextChatEngine does not replace HyDE. They actually solve two completely different problems in the RAG pipeline.
    # 1. What CondensePlusContext Solves: The "Memory" Problem
    # Condense engine fixes coreference resolution and conversational context.
    # It ensures the retriever understands what you are talking about by rewriting the query based on past messages.
    # Input: "At what age did he die?"
    # What the LLM does: Translates conversation into a standalone query.
    # Output: "At what age did Napoleon Bonaparte die?"
    # It keeps the query in the "Question Space". It just makes it a better, more specific question.

    # 2. What HyDE Solves: The "Vocabulary Mismatch" Problem
    # HyDE (Hypothetical Document Embeddings) solves a completely different issue:
    # the fact that questions look mathematically different from answers. When you embed a short question (like "At what age did Napoleon Bonaparte die?"),
    # its vector represents a question. But in your vector database, you are looking for an answer—a factual paragraph from a history book.
    # Sometimes, the embedding model struggles to match a question vector to a statement vector because the phrasing and vocabulary are so entirely different.
    # Input: "At what age did Napoleon Bonaparte die?"
    # What HyDE does: Asks the LLM to hallucinate/generate a fake, plausible document that answers the question before searching.
    # Output (Hypothetical Document): "Napoleon Bonaparte died on May 5, 1821. He was 51 years old at the time of his death on the island of Saint Helena..."
    # HyDE takes your query and forces it into the "Document/Answer Space".
    # The retriever then takes this fake paragraph, embeds it, and searches your vector database for chunks that look mathematically similar to that paragraph,
    # rather than chunks that look like the original question.

    #commented out recently
    # chat_engine = CondensePlusContextChatEngine(
    #     retriever=hyde_retriever, # Using our new HyDE retriever here
    #     llm=llm,
    #     memory=memory,
    #     system_prompt=LLM_SYSTEM_PROMPT,
    #     node_postprocessors=[reranker]
    # )

    # return chat_engine


def main_chat_loop() -> None:
    """Main application loop to run the RAG chatbot."""

    print("--- Initialising models... ---")
    llm: Groq = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()

    chat_engine: CondensePlusContextChatEngine = get_chat_engine(
        llm=llm,
        embed_model=embed_model
    )
    print("--- RAG Chatbot Initialised. ---")
    chat_engine.chat_repl()

def main_chat_loop_basic() -> None:
    """Main chat loop for a conversational chatbot."""
    llm = initialise_llm()

    # Create a chat engine using the system prompt
    conversation = SimpleChatEngine.from_defaults(
        llm=llm,
        system_prompt=LLM_SYSTEM_PROMPT
    )

    print("--- Chat Started ---")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("--------------------")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("\nBot: Goodbye!")
            break

        if not user_input:
            continue

        response = conversation.chat(user_input)
        print(f"\nBot: {response}")

if __name__ == '__main__':
    # main_chat_loop()
    main_chat_loop_basic()
