"""Main script for running the Icebreaker Bot (Local Ollama Edition)."""

import sys
import io
import time
import logging
import argparse

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from modules.data_extraction import extract_linkedin_profile, list_available_profiles
from modules.data_processing import (
    split_profile_data,
    create_vector_database,
    verify_embeddings,
)
from modules.query_engine import (
    generate_initial_facts,
    answer_user_query,
)

import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def process_linkedin(profile_key):
    """
    Process a LinkedIn profile, extract data, and interact with the user.

    Args:
        profile_key: Profile key name or LinkedIn URL.
    """
    try:
        # Extract the profile data
        profile_data = extract_linkedin_profile(profile_key)

        if not profile_data:
            logger.error("Failed to retrieve profile data.")
            return

        print(f"\n✅ Loaded profile: {profile_data.get('full_name', 'Unknown')}")
        print(f"   Headline: {profile_data.get('headline', 'N/A')}")

        # Split the data into nodes
        nodes = split_profile_data(profile_data)

        # Store in vector database
        vectordb_index = create_vector_database(nodes)

        if not vectordb_index:
            logger.error("Failed to create vector database.")
            return

        # Verify embeddings
        if not verify_embeddings(vectordb_index):
            logger.warning("Some embeddings may be missing or invalid.")

        # Generate and display initial facts
        print("\n⏳ Generating icebreaker facts (this may take a moment)...\n")
        initial_facts = generate_initial_facts(vectordb_index)

        print("Here are 3 interesting facts about this person:")
        print(initial_facts)

        # Start chatbot
        chatbot_interface(vectordb_index)

    except Exception as e:
        logger.error(f"Error occurred: {e}")


def chatbot_interface(index):
    """
    Simple CLI chatbot.

    Args:
        index: VectorStoreIndex containing profile data.
    """
    print(
        "\nYou can now ask more in-depth questions about this person."
    )
    print("Type 'exit', 'quit', or 'bye' to quit.\n")

    while True:
        user_query = input("You: ")

        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye!")
            break

        print("Bot is typing...", end="")
        sys.stdout.flush()

        response = answer_user_query(index, user_query)

        print("\r", end="")

        if hasattr(response, "response"):
            print(f"Bot: {response.response.strip()}\n")
        else:
            print(f"Bot: {response}\n")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Icebreaker Bot - LinkedIn Profile Analyzer (Local Ollama Edition)"
    )

    parser.add_argument(
        "--profile",
        type=str,
        help="Profile key name (e.g. 'leonkatsnelson', 'satya_nadella', 'sample_data_scientist')",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Ollama model to use (e.g. 'llama3.2', 'mistral', 'deepseek-coder:6.7b')",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available sample profiles",
    )

    args = parser.parse_args()

    # List profiles
    if args.list:
        profiles = list_available_profiles()
        print("\n📋 Available sample profiles:")
        for p in profiles:
            print(f"   - {p}")
        print(f"\nUsage: python main.py --profile {profiles[0]}")
        return

    # Change model if provided
    if args.model:
        from modules.llm_interface import change_llm_model
        change_llm_model(args.model)

    # Get profile key
    profile_key = args.profile
    if not profile_key:
        profiles = list_available_profiles()
        print("\n📋 Available sample profiles:")
        for i, p in enumerate(profiles, 1):
            print(f"   {i}. {p}")

        choice = input(
            "\nEnter profile name or number (or press Enter for default): "
        ).strip()

        if choice.isdigit() and 1 <= int(choice) <= len(profiles):
            profile_key = profiles[int(choice) - 1]
        elif choice:
            profile_key = choice
        else:
            profile_key = profiles[0]

    print(f"\n🚀 Using Ollama model: {config.LLM_MODEL_ID}")
    print(f"📂 Loading profile: {profile_key}")

    process_linkedin(profile_key)


if __name__ == "__main__":
    main()
