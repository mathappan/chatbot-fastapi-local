"""
RAG-Based Search Flow for E-commerce Product Search

This module provides a complete RAG (Retrieval-Augmented Generation) pipeline for:
1. Converting customer intent to search queries
2. Vector-based product search using Redis and Voyage embeddings
3. Product reranking and relevance scoring

Key Features:
- Vector similarity search with Redis
- Product category filtering
- AI-powered search query optimization
- Complete search pipeline
"""

import os
from dotenv import load_dotenv, find_dotenv
import voyageai
import redis
from typing import List
from redis.commands.search.query import Query
import numpy as np
import asyncio
from redis_filter_utils import build_combined_filter
from product_validator import get_product_validator
import json

# Load environment variables once
_ = load_dotenv(find_dotenv('.env.txt'))

# Initialize clients as module-level singletons
VOYAGE_API_KEY = os.environ['VOYAGER_API_KEY']
voyageai.api_key = VOYAGE_API_KEY
vo_client = voyageai.AsyncClient()

# Use centralized Groq client from clients.py
from clients import groq_client

# Import unified product category extractor
from unified_product_category import unified_product_category_extractor

# Connection pool for Redis with optimized settings - Vector embeddings storage
from redis_client_manager import get_vector_redis_client
vector_embeddings_redis_client = get_vector_redis_client()

# Pre-compiled query object (reuse across calls)
REDIS_QUERY = (
    Query('(*)=>[KNN 50 @vector $query_vector AS vector_score]')
     .sort_by('vector_score')
     .return_fields('vector_score', 'internal_id', 'text_description', 'product_type')
     .paging(0, 25) 
     .dialect(2)
)

# Available product categories in the store
product_categories = ['Plus Size Suits', 'Sarees', 'Kurta Sets', 'Co-Ords', 'Unstitched Suits', 'Plus Size Kurtas', 'Loungewear', 'Dresses', 'Plus Size Kurta Sets', 
                      'Kurta', 'Suits', 'Plus Size Short Kurtis', 'Short Kurtis', 'Bottoms', 'Plus Size Kurtis', 'Kurtis', 'Plus Size Co-Ords', 'Lehengas', 'Shirts', 
                      'Unknown']

# Cached prompt template for search query creation
SEARCH_QUERY_CREATION_PROMPT = """
You are an assistant for an e-commerce clothing store. Your task is to convert the customer intent into a concise effective specific Google-style search query.

customer intent - {customer_intent}
Instructions:
- Do not include size and budget.
- Ensure to convert all negative preferences into specific positive alternatives.
- Use positive affirmation only.
- Do not use negative words at all.
- Think about how you would translate the users request into options in filters on an ecommerce website.

Your output should be only the query string. No explanation, quotes, or formatting.
"""


def get_formatted_prompt(customer_intent: str) -> str:
    """Format prompt template with customer intent"""
    return SEARCH_QUERY_CREATION_PROMPT.format(customer_intent=customer_intent)

# Product validation functions moved to shared product_validator.py module
# to eliminate code duplication with master_search.py

async def search_products(customer_intent: str, genders = None, excluded_product_ids: List[str] = None):
    """
    Main entry point for RAG-based product search
    
    Args:
        customer_intent: Customer's search intent/query
        genders: List of gender preferences (defaults to None)
        excluded_product_ids: List of product IDs to exclude from search
        
    Returns:
        List of dicts with product_id, relevance_score, and text_description
    """
    # Generate optimized search query using LLM
    search_query_response = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Strictly follow instructions."},
            {"role": "user", "content": get_formatted_prompt(customer_intent)}
        ],
        temperature=0.3,
        max_tokens=100,
    )
    
    search_query = search_query_response.choices[0].message.content.strip()
    print("search query - ", search_query)
    
    return await search_products_with_query(search_query, genders, excluded_product_ids)

async def search_products_with_query(search_query: str, genders = None, excluded_product_ids: List[str] = None, allowed_product_types: List[str] = None, sizes: List[str] = None, user_min_budget: float = None, user_max_budget: float = None):
    """
    RAG-based product search using pre-optimized query
    
    Args:
        search_query: Pre-optimized search query
        genders: List of gender preferences (defaults to None)
        excluded_product_ids: List of product IDs to exclude from search
        allowed_product_types: Pre-extracted product categories (optional)
        
    Returns:
        List of dicts with product_id, relevance_score, and text_description
    """
    
    print(f"Using product types: {allowed_product_types}")
    
    # Generate embedding for the search query
    embedding_response = await vo_client.embed([search_query], model="voyage-3-large", input_type="query")
    query_embedding = embedding_response.embeddings[0]

    # Build combined filter using the utility function
    combined_filter = build_combined_filter(
        allowed_product_types=allowed_product_types,
        sizes=sizes,
        user_min_budget=user_min_budget,
        user_max_budget=user_max_budget,
        excluded_ids=excluded_product_ids,
        gender=genders
    )
    
    print(f"🔍 Applied Redis filters: {combined_filter}")
    
    # Create filtered Redis query combining all filters and vector search
    filtered_redis_query = (
        Query(f'({combined_filter})=>[KNN 50 @vector $query_vector AS vector_score]')
        .sort_by('vector_score')
        .return_fields('vector_score', 'internal_id', 'text_description', 'product_type', 'sizes', 
                       'min_price', 'max_price')
        .paging(0, 50) 
        .dialect(2)
    )
    
    # Execute vector search in Redis
    query_vector_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
    search_results = vector_embeddings_redis_client.ft('idx:product_text_description_embedding').search(
        filtered_redis_query,
        {'query_vector': query_vector_bytes}
    ).docs
    
    # Rerank results using Voyage reranker if we have results
    if search_results:
        documents = [doc['text_description'] for doc in search_results]
        reranked_results = await vo_client.rerank(
            query=search_query, 
            documents=documents, 
            model="rerank-2-lite", 
            top_k=10
        )
        
        # Pre-extract internal_ids to avoid repeated dictionary lookups
        internal_ids = [doc.internal_id for doc in search_results]
        
        # Get reranked product IDs
        reranked_product_ids = [internal_ids[r.index] for r in reranked_results.results]
        
        # Fetch metadata (title, image_url, product_url) from Redis
        metadata_map = {}
        if reranked_product_ids:
            # Create OR query for all internal IDs
            query_string = " | ".join([f"@internal_id:{id_}" for id_ in reranked_product_ids])
            metadata_query = (
                Query(query_string)
                .return_fields("internal_id", "image_url", "product_url", "title", "max_price", "body_text")
            )
            
            try:
                metadata_results = vector_embeddings_redis_client.ft("idx:product_metadata").search(metadata_query)
                for doc in metadata_results.docs:
                    metadata_map[doc.internal_id] = {
                        'title': getattr(doc, 'title', f"Product {doc.internal_id}"),
                        'image_url': getattr(doc, 'image_url', ''),
                        'product_url': getattr(doc, 'product_url', f"/product/{doc.internal_id}"),
                        'max_price': getattr(doc, 'max_price', 0),
                        'body_text': getattr(doc, 'body_text', '')
                    }
            except Exception as e:
                from logger_config import log_detailed_error
                log_detailed_error(
                    e,
                    context="search_products_with_query.metadata_fetch",
                    local_vars={
                        "reranked_product_ids": reranked_product_ids[:5],  # First 5 IDs
                        "search_query": search_query,
                        "query_string": query_string[:200] if 'query_string' in locals() else None
                    }
                )
        
        # Prepare product candidates for matching validation
        product_candidates = [
            {
                'product_id': internal_ids[r.index],
                'relevance_score': r.relevance_score,
                'text_description': r.document,
                'title': metadata_map.get(internal_ids[r.index], {}).get('title', f"Product {internal_ids[r.index]}"),
                'image_url': metadata_map.get(internal_ids[r.index], {}).get('image_url', ''),
                'product_url': metadata_map.get(internal_ids[r.index], {}).get('product_url', f"/product/{internal_ids[r.index]}"),
                'max_price': metadata_map.get(internal_ids[r.index], {}).get('max_price', 0),
                'body_text': metadata_map.get(internal_ids[r.index], {}).get('body_text', '')
            }
            for r in reranked_results.results
        ]
        
        # Validate products against search query in parallel using shared validator
        validator = get_product_validator()
        validated_products = await validator.validate_product_matches(search_query, product_candidates)
        
        # Return all three sets: initial shortlisted, reranked, and validated
        # Create dictionaries mapping product_id -> relevance_score for consistency with master_search
        shortlisted_dict = {doc.internal_id: float(doc.vector_score) for doc in search_results}
        reranked_dict = {internal_ids[r.index]: r.relevance_score for r in reranked_results.results}
        validated_dict = {product['product_id']: product.get('relevance_score', 0) for product in validated_products}
        
        return {
            'validated_products': validated_products,
            'product_candidates': product_candidates,  # All reranked products (before validation)
            'shortlisted_products': shortlisted_dict,  # Initial vector search results
            'reranked_products': reranked_dict,        # Reranked products
            'validated_dict': validated_dict           # Final validated products as dict
        }
    
    # Return empty structure if no initial results
    return {
        'validated_products': [],
        'product_candidates': [],
        'shortlisted_products': {},
        'reranked_products': {},
        'validated_dict': {}
    }



async def search_and_pitch(optimized_query: str, genders = None, conversation_id: str = None, conversation_redis_client=None, allowed_product_types: List[str] = None, sizes: List[str] = None, user_min_budget: float = None, user_max_budget: float = None, excluded_ids: List[str] = None):
    """
    Complete RAG pipeline: search products with filtering (no pitching)
    
    This is the main entry point for the RAG-based search flow.
    It combines vector search and reranking.
    
    Args:
        optimized_query: Pre-optimized search query from groq_utils
        genders: List of gender preferences (defaults to None)
        conversation_id: Conversation ID to get excluded products
        allowed_product_types: List of product types to filter by
        sizes: List of sizes to filter by
        user_min_budget: Minimum price range
        user_max_budget: Maximum price range
        excluded_ids: List of product IDs to exclude
        conversation_redis_client: Redis client for conversation data
        
    Returns:
        Dict with search results (no pitches)
    """
    # Step 1: Get excluded product IDs from conversation history
    excluded_product_ids = []
    if conversation_id and conversation_redis_client:
        # Get both RAG and filter product IDs from conversation
        rag_key = f"conversation:{conversation_id}:search:rag:product_ids"
        filter_key = f"conversation:{conversation_id}:search:filter:product_ids"
        
        rag_ids = await conversation_redis_client.smembers(rag_key)
        filter_ids = await conversation_redis_client.smembers(filter_key)
        
        excluded_product_ids = list(set(list(rag_ids or []) + list(filter_ids or [])))
        if excluded_product_ids:
            print(f"Found {len(excluded_product_ids)} existing product IDs to exclude")
    
    # Step 2: Get search results using RAG-based vector search with filtering
    search_results_data = await search_products_with_query(
        optimized_query, 
        genders, 
        excluded_product_ids, 
        allowed_product_types, 
        sizes, 
        user_min_budget, 
        user_max_budget
    )
    
    # Extract validated products (final results)
    validated_products = search_results_data.get('validated_products', [])
    
    if not validated_products:
        return {
            "status": "success",
            "query": optimized_query,
            "products_found": 0,
            "product_recommendations": {},
            "ranked_products": search_results_data.get('shortlisted_products', {}),
            "formatted_response": {"search_results": []}
        }
    
    # Return same structure as master_search.search_products
    return {
        "status": "success",
        "query": optimized_query,
        "products_found": len(validated_products),
        "product_recommendations": search_results_data.get('validated_dict', {}),
        "ranked_products": search_results_data.get('shortlisted_products', {}),  # All initial vector search results for zero-results analysis
        "formatted_response": {"search_results": validated_products}
    }