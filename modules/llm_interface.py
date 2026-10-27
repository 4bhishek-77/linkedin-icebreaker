"""Module for interfacing with local Ollama LLM and HuggingFace embeddings."""

import logging

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import config

logger = logging.getLogger(__name__)

# Module-level singletons so we don't recreate on every call
_embedding_model = None
_llm_instance = None


def create_embedding():
    """
    Creates a local HuggingFace embedding model (no API key needed).

    Returns:
        HuggingFaceEmbedding model.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    _embedding_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL_NAME,
    )

    logger.info(
        f"Created HuggingFace Embedding model: {config.EMBEDDING_MODEL_NAME}"
    )
    return _embedding_model


def create_llm(
    temperature: float = None,
    max_new_tokens: int = None,
) -> Ollama:
    """
    Creates an Ollama LLM instance for generating responses.

    Args:
        temperature: Temperature for controlling randomness (0.0-1.0).
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        Ollama LLM instance.
    """
    if temperature is None:
        temperature = config.TEMPERATURE
    if max_new_tokens is None:
        max_new_tokens = config.MAX_NEW_TOKENS

    llm = Ollama(
        model=config.LLM_MODEL_ID,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        request_timeout=120.0,
        additional_kwargs={
            "num_predict": max_new_tokens,
            "top_k": config.TOP_K,
            "top_p": config.TOP_P,
        },
    )

    logger.info(f"Created Ollama LLM: {config.LLM_MODEL_ID}")
    return llm


def change_llm_model(new_model_id: str) -> None:
    """
    Change the LLM model to use.

    Args:
        new_model_id: New Ollama model name to use (e.g. 'llama3.2', 'mistral').
    """
    global _llm_instance
    config.LLM_MODEL_ID = new_model_id
    _llm_instance = None  # Reset so next call creates a fresh instance
    logger.info(f"Changed LLM model to: {new_model_id}")
