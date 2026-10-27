"""Configuration settings for the Icebreaker Bot (Local Ollama Edition)."""

import os

# --- Ollama settings ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Default model – change to whatever you've pulled in Ollama
LLM_MODEL_ID = os.getenv("OLLAMA_MODEL", "llama3.2")

# Embedding model – runs locally via HuggingFace sentence-transformers (no API needed)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Local sample data ---
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
SAMPLE_DATA_FILE = os.path.join(SAMPLE_DATA_DIR, "linkedin_profiles.json")

# Available sample profiles
AVAILABLE_PROFILES = [
    "satya_nadella",
    "jensen_huang",
    "sam_altman",
    "andrej_karpathy",
    "demis_hassabis",
    "feifei_li",
    "leonkatsnelson",
    "sample_data_scientist",
]

# Human-readable profile labels for UI display
PROFILE_LABELS = {
    "satya_nadella": "Satya Nadella — Chairman & CEO, Microsoft",
    "jensen_huang": "Jensen Huang — Founder & CEO, NVIDIA",
    "sam_altman": "Sam Altman — CEO, OpenAI",
    "andrej_karpathy": "Andrej Karpathy — Founder Eureka Labs, Ex-Tesla AI / OpenAI",
    "demis_hassabis": "Sir Demis Hassabis — CEO Google DeepMind, 2024 Nobel Laureate",
    "feifei_li": "Dr. Fei-Fei Li — Professor at Stanford, Creator of ImageNet",
    "leonkatsnelson": "Leon Katsnelson — CTO, Former IBM Distinguished Engineer",
    "sample_data_scientist": "Priya Sharma — Senior Data Scientist, Google",
}

# Query settings
SIMILARITY_TOP_K = 5
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 500
TOP_K = 50
TOP_P = 0.9

# Node settings
CHUNK_SIZE = 500

# LLM prompt templates
INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides engaging, accurate, and detailed responses based strictly on the provided profile context.

Context information is below:
---------------------
{context_str}
---------------------

Based on the context provided, write 3 interesting and insightful facts about this person's career, education, or accomplishments.
Format them with clear bullet points and bold highlights.

Answer using only the information provided in the context.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed, factual answers to questions based on the provided profile context.

Context information is below:
---------------------
{context_str}
---------------------

Question: {query_str}

Answer thoroughly and directly, using only the information in the context. If the answer is not mentioned in the context, say: "I don't know based on the available profile information."
"""
