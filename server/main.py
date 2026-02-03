"""
MCP Server for Product Recommendation
Main entry point for the FastMCP server
"""

from fastmcp import FastMCP
from dotenv import load_dotenv

from config.settings import settings
from utils.logger import logger
from services.milvus_service import milvus_service
from services.ingestion_service import ingestion_service
from tools.product_tools import (
    search_product_documents,
    preety_print_product_metadata_response,
    return_ranked_product_response_from_ranked_index,
    product_metadata_analysis_for_refine_or_tuning_search_result
)

from prometheus_client import start_http_server, Counter, Histogram

# Load environment variables
load_dotenv()

# --- Prometheus Metrics ---
TOOL_USAGE_TOTAL = Counter('tool_usage_count', 'Tool Usage Count', ['tool_name'])
RAG_LATENCY = Histogram('rag_retrieval_latency_seconds', 'RAG Retrieval Latency', ['db_type'])
LLM_TOKEN_USAGE = Counter('llm_token_usage_total', 'LLM Token Usage', ['model', 'type']) # type=prompt/completion
# --------------------------

# Initialize FastMCP server
mcp = FastMCP(settings.SERVER_NAME)


def initialize_services():
    """Initialize all services and ensure data is ready"""
    try:
        logger.info("Initializing MCP Server services...")
                
        # Create collection if it doesn't exist
        milvus_service.create_collection()
        
        # Check if products need to be ingested
        status = ingestion_service.get_ingestion_status()
        logger.info(f"Ingestion status: {status}")
        
        # Force re-ingestion if cache says ingested but database is empty
        force_reingest = (status["ingested_files"] > 0 and status["entities_in_milvus"] == 0)
        
        if force_reingest:
            logger.warn("Cache shows ingested files but database is empty - forcing re-ingestion")
            ingested_count = ingestion_service.ingest_products(force_reingest=True)
            logger.success(f"Re-ingested {ingested_count} products")
        elif status["pending_files"] > 0:
            logger.info(f"Found {status['pending_files']} pending files, starting ingestion...")
            ingested_count = ingestion_service.ingest_products()
            logger.success(f"Ingested {ingested_count} products")
        else:
            logger.info("All products already ingested")
        
        logger.success("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise



# Register MCP tools
@mcp.tool()
def search_products(query: str, top_k: int = 5):
    """
    Based on the query, search for relevant products from the product documents.
    Return the top_k products.

    @param query: Search query string
    @param top_k: Number of top results to return (default: 5)
    @return: List of ProductResponse objects
    """
    TOOL_USAGE_TOTAL.labels(tool_name="search_products").inc()
    with RAG_LATENCY.labels(db_type="milvus").time():
        return search_product_documents(query, top_k)


@mcp.tool()
def format_product_metadata(product_response_list):
    """
    Pretty print a list of ProductResponse objects' metadata
    
    Args:
        product_response_list: Either a List[ProductResponse], a JSON string, or a list of JSON strings
    
    Returns:
        str: Formatted string of product metadata
    """
    TOOL_USAGE_TOTAL.labels(tool_name="format_product_metadata").inc()
    return preety_print_product_metadata_response(product_response_list)


@mcp.tool()
def rerank_products(product_responses, ranked_indices):
    """
    Reorders a list of ProductResponse objects based on a provided ranking of indices.

    This function is used when a model (e.g., an LLM or retrieval system) provides a ranked list 
    of indices indicating the relevance or order of product responses based on a user query.

    Args:
        product_responses: The original list of product responses
        ranked_indices: A list of indices representing the new ranked order

    Returns:
        List[ProductResponse]: A reordered list of product responses

    Example:
        If product_responses = [A, B, C] and ranked_indices = [2, 0], 
        the returned result will be [C, A]
    """
    TOOL_USAGE_TOTAL.labels(tool_name="rerank_products").inc()
    return return_ranked_product_response_from_ranked_index(product_responses, ranked_indices)


@mcp.tool()
def get_product_attributes():
    """
    Returns a dictionary of unique product attributes and sub-attributes that can be used by an LLM
    to refine, tune, or rerank search results based on user intent.

    This metadata can guide the LLM in determining which fields are important to filter or re-rank 
    products when the search_products() function does not return exact semantic matches.

    Returns:
        dict: A dictionary with keys representing attribute groups and values as lists of attribute names
    """
    TOOL_USAGE_TOTAL.labels(tool_name="get_product_attributes").inc()
    return product_metadata_analysis_for_refine_or_tuning_search_result()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting MCP Product Recommendation Server")
    logger.info("=" * 60)
    
    try:
        # Start Prometheus Metrics Server on Port 8000
        logger.info("Starting Prometheus Metrics Server on port 8000")
        start_http_server(8000)

        # Initialize services before starting server
        initialize_services()
        
        # Start the MCP server - this will block and handle stdio
        logger.info("MCP Server is ready to accept connections")
        mcp.run(transport="stdio")
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Cleanup
        try:
            milvus_service.disconnect()
        except:
            pass
        logger.info("Server stopped")
