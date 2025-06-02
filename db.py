from agno.agent import Agent
from agno.knowledge.arxiv import ArxivKnowledgeBase
from agno.vectordb.lancedb import LanceDb

from params import EMBEDDING_MODEL

def setup_lancedb_knowledge_base(queries: list[str], dataset_name: str, recreate_db: bool = True):
    """
    Initializes a LanceDB vector database and an ArxivKnowledgeBase.
    
    Args:
        queries: List of queries (keywords) for ArxivKnowledgeBase.
        dataset_name: Name for the LanceDB table (analogous to collection name).
        recreate_db: If True, forces recreation of the LanceDB table.
                       This is useful for ensuring a fresh state during KG creation.

    Returns:
        An initialized ArxivKnowledgeBase instance backed by LanceDB.
    """
    vector_db = LanceDb(
        table_name=dataset_name,
        uri="./lancedb_data",
        embedder=EMBEDDING_MODEL
    )

    # Knowledge Base
    knowledge_base = ArxivKnowledgeBase(
        queries=queries,
        vector_db=vector_db,
    )

    return knowledge_base