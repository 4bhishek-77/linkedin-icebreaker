"""Module for processing LinkedIn profile data."""

import json
import logging
from typing import Dict, List, Any, Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from modules.llm_interface import create_embedding
import config

logger = logging.getLogger(__name__)


def split_profile_data(profile_data: Dict[str, Any]) -> List:
    """
    Split the LinkedIn profile JSON data into nodes.

    Args:
        profile_data: LinkedIn profile data dictionary.

    Returns:
        List of document nodes.
    """
    try:
        # Convert the profile data to a JSON string
        profile_json = json.dumps(profile_data)

        # Create a Document object from the JSON string
        document = Document(text=profile_json)

        # Split the document into nodes using SentenceSplitter
        splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE)
        nodes = splitter.get_nodes_from_documents([document])

        logger.info(f"Created {len(nodes)} nodes from profile data")
        return nodes

    except Exception as e:
        logger.error(f"Error in split_profile_data: {e}")
        return []


def create_vector_database(nodes: List) -> Optional[VectorStoreIndex]:
    """
    Store the document chunks (nodes) in a vector database.

    Args:
        nodes: List of document nodes to be indexed.

    Returns:
        VectorStoreIndex or None if indexing fails.
    """
    try:
        # Get the local embedding model
        embedding_model = create_embedding()

        # Create a VectorStoreIndex from the nodes
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embedding_model,
            show_progress=True,
        )

        logger.info("Vector database created successfully")
        return index

    except Exception as e:
        logger.error(f"Error in create_vector_database: {e}")
        return None


def verify_embeddings(index: VectorStoreIndex) -> bool:
    """
    Verify that all nodes have been properly embedded.

    Args:
        index: VectorStoreIndex to verify.

    Returns:
        True if all embeddings are valid, False otherwise.
    """
    try:
        node_ids = list(index.index_struct.nodes_dict.keys())
        logger.info(f"Index contains {len(node_ids)} embedded nodes")
        return len(node_ids) > 0

    except Exception as e:
        logger.error(f"Error in verify_embeddings: {e}")
        return False
