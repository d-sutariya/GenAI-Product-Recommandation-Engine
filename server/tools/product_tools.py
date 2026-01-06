"""
MCP Tools for product recommendation
All tool functions that are exposed via FastMCP
"""

import json
import re
from typing import List, Optional

from models.products import (
    ProductResponse,
    ProductChunkTyped,
    ProductMetadata,
    ProductMetadataSubset
)
from services.milvus_service import milvus_service
from services.embedding_service import embedding_service
from utils.logger import logger


def search_product_documents(query: str, top_k: int = 5) -> List[ProductResponse]:
    """
    Based on the query, search for relevant products from the product documents.
    Return the top_k products.

    @param query: str
    @param top_k: int
    @return list[ProductResponse]
    """
    logger.info(f"Searching products with query: '{query}', top_k: {top_k}")
    
    try:
        # Generate query embedding
        query_embedding = embedding_service.get_embedding(query)
        
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search in Milvus
        search_results = milvus_service.search(query_embedding, top_k=top_k)
        
        if not search_results:
            logger.info("No results found")
            return []
        
        # Convert results to ProductResponse
        results = []
        for result in search_results:
            try:
                # Parse metadata JSON
                metadata_dict = json.loads(result["metadata"])
                
                # Create ProductChunkTyped
                product_chunk = ProductChunkTyped(
                    id=result["id"],
                    product_content=result["product_content"],
                    metadata=ProductMetadata(**metadata_dict)
                )
                
                # Convert to ProductResponse
                product_response = ProductResponse.from_product_chunk(product_chunk)
                results.append(product_response)
                
            except Exception as e:
                logger.error(f"Error processing search result: {e}")
                continue
        
        logger.info(f"Returning {len(results)} product results")
        return results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def preety_print_product_metadata_response(
    product_response_list: List[ProductResponse] | str | List[str]
) -> str:
    """
    Pretty print a list of ProductResponse objects' metadata
    
    Args:
        product_response_list: Either a List[ProductResponse], a JSON string, or a list of JSON strings
    
    Returns:
        str: Formatted string of product metadata
    """
    try:
        responses = []
        
        # Handle different input types
        if isinstance(product_response_list, str):
            # Single JSON string - Fix escaping before parsing
            try:
                # First try direct JSON parsing
                data = json.loads(product_response_list)
            except json.JSONDecodeError:
                # If that fails, try cleaning the string
                cleaned_json = product_response_list.replace("\\'", "'")
                cleaned_json = cleaned_json.replace('\\"', '"')
                cleaned_json = cleaned_json.replace('\\\\', '\\')
                data = json.loads(cleaned_json)
                
            if isinstance(data, list):
                responses.extend([ProductResponse(**item) for item in data])
            else:
                responses.append(ProductResponse(**data))
                
        elif isinstance(product_response_list, list):
            # List of JSON strings or ProductResponse objects
            for item in product_response_list:
                if isinstance(item, str):
                    try:
                        data = json.loads(item)
                    except json.JSONDecodeError:
                        cleaned_json = item.replace("\\'", "'")
                        cleaned_json = cleaned_json.replace('\\"', '"')
                        cleaned_json = cleaned_json.replace('\\\\', '\\')
                        data = json.loads(cleaned_json)
                    responses.append(ProductResponse(**data))
                elif isinstance(item, ProductResponse):
                    responses.append(item)
                else:
                    raise ValueError(f"Invalid item type: {type(item)}")
        else:
            raise ValueError(f"Invalid input type: {type(product_response_list)}")
            
        # Extract and format metadata for each product
        formatted_responses = []
        for product in responses:
            metadata = product.product_metadata
            metadata_subset = ProductMetadataSubset.from_product_metadata(metadata)
            formatted_responses.append(metadata_subset.model_dump_json(indent=2))
            
        return "\n\n".join(formatted_responses)
        
    except Exception as e:
        logger.error(f"Error formatting product response: {e}")
        return f"Error formatting product response: {str(e)}\nInput: {str(product_response_list)[:200]}..."


def return_ranked_product_response_from_ranked_index(
    product_responses: List[ProductResponse],
    ranked_indices: List[int]
) -> List[ProductResponse]:
    """
    Reorders a list of ProductResponse objects based on a provided ranking of indices.

    This function is used when a model (e.g., an LLM or retrieval system) provides a ranked list 
    of indices indicating the relevance or order of product responses based on a user query.
    The function maps those indices back to the original list of `ProductResponse` objects and 
    returns them in the ranked order.

    Args:
        product_responses (List[ProductResponse]): The original list of product responses.
        ranked_indices (List[int]): A list of indices representing the new ranked order.

    Returns:
        List[ProductResponse]: A reordered list of product responses based on the ranked indices.

    Example:
        If `product_responses = [A, B, C]` and `ranked_indices = [2, 0]`, 
        the returned result will be `[C, A]`.
    """
    try:
        return [product_responses[i] for i in ranked_indices]
    except IndexError as e:
        logger.error(f"Invalid index in ranked_indices: {e}")
        return []


def product_metadata_analysis_for_refine_or_tuning_search_result() -> dict:
    """
    Returns a dictionary of unique product attributes and sub-attributes that can be used by an LLM
    to refine, tune, or rerank search results based on user intent.

    This metadata can guide the LLM in determining which fields are important to filter or re-rank 
    products when the `search_product_documents(query: str, top_k: int = 5)` function does not return 
    exact semantic matches.

    For example:
    - In 'master_category', the important field is 'typeName' (e.g., Apparel, Accessories, Footwear).
    - In 'sub_category', the key 'typeName' holds values like Topwear, Bags, Shoes.
    - In 'article_type', 'typeName' includes values such as Tshirts, Backpacks, Water Bottles.
    - For brand, price, ageGroup, gender, season, etc., these fields can help refine or rerank results.

    Example use case:
    If a user searches for "Nike T-shirt for casual wear for men under 100 dollars", the LLM can refine 
    or rerank the results based on:
        - brandName = Nike
        - gender = Men
        - price < 100
        - article_type = T-shirt
        - usage = Casual

    Returns:
        dict: A dictionary with keys representing attribute groups (e.g., article_attributes, metadata)
              and values as lists of relevant attribute names.
    """
    
    article_attributes = [
        'Add-Ons', 'Ankle Height', 'Arch Type', 'Assorted', 'Back', 'Base Metal', 
        'Belt Width', 'Blouse', 'Blouse Fabric', 'Body or Garment Size', 'Border', 
        'Bottom Closure', 'Bottom Fabric', 'Bottom Pattern', 'Bottom Type', 'Brand', 
        'Brand Fit Name', 'Brick', 'Business Unit', 'Case', 'Character', 'Class', 
        'Cleats', 'Closure', 'Coin Pocket Type', 'Collar', 'Colour Family', 
        'Colour Hex Code', 'Colour Shade Name', 'Compartment Closure', 'Concern', 
        'Content', 'Coverage', 'Cuff', 'Cushioning', 'Design', 'Design Styling'
    ]
    
    master_category = ['typeName']
    sub_category = ['typeName']
    article_type = ['typeName']
    product_descriptors = ['description', 'materials_care_desc', 'size_fit_desc', 'style_note']
    metadata = [
        "id", "price", "discountedPrice", "styleType", "productTypeId", "articleNumber",
        "productDisplayName", "variantName", "myntraRating", "catalogAddDate", "brandName",
        "ageGroup", "gender", "baseColour", "colour1", "colour2", "fashionType",
        "season", "year", "usage", "vat", "displayCategories"
    ]
    
    return {
        "article_attributes": article_attributes,
        "master_category": master_category,
        "sub_category": sub_category,
        "article_type": article_type,
        "product_descriptors": product_descriptors,
        "metadata": metadata
    }
