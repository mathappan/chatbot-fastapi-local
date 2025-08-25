"""
RAG-Based Search Flow for E-commerce Product Search and Pitch Generation

This module provides a complete RAG (Retrieval-Augmented Generation) pipeline for:
1. Converting customer intent to search queries
2. Vector-based product search using Redis and Voyage embeddings
3. Product reranking and relevance scoring
4. AI-generated product pitches using Groq

Key Features:
- Vector similarity search with Redis
- Product category filtering
- AI-powered search query optimization
- Parallel product pitch generation
- Complete search-to-pitch pipeline
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
import json

# Load environment variables once
_ = load_dotenv(find_dotenv('.env.txt'))

# Initialize clients as module-level singletons
VOYAGE_API_KEY = os.environ['VOYAGER_API_KEY']
voyageai.api_key = VOYAGE_API_KEY
vo_client = voyageai.AsyncClient()

from groq import AsyncGroq
groq_client = AsyncGroq(api_key=os.environ.get('GROQ_API_KEY'))

# Import data loader for product images
from data_loader import data_loader
# Import unified product category extractor
from unified_product_category import unified_product_category_extractor

# Connection pool for Redis with optimized settings - Vector embeddings storage
vector_embeddings_redis_client = redis.Redis(
    host='redis-19985.c212.ap-south-1-1.ec2.redns.redis-cloud.com',
    port=19985,
    decode_responses=True,
    username="default",
    password="9sctWbao6E8VydXA2KlBTp1CTcOjAlko",
    socket_connect_timeout=5,
    socket_timeout=5,
    socket_keepalive=True,
    socket_keepalive_options={},
    max_connections=10,
)

# Pre-compiled query object (reuse across calls)
REDIS_QUERY = (
    Query('(*)=>[KNN 50 @vector $query_vector AS vector_score]')
     .sort_by('vector_score')
     .return_fields('vector_score', 'product_id', 'text_description', 'product_type')
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

async def search_products(customer_intent: str, gender: str = None, excluded_product_ids: List[str] = None):
    """
    Main entry point for RAG-based product search
    
    Args:
        customer_intent: Customer's search intent/query
        gender: Gender preference (defaults to None)
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
    
    return await search_products_with_query(search_query, gender, excluded_product_ids)

async def search_products_with_query(search_query: str, gender: str = None, excluded_product_ids: List[str] = None, allowed_product_types: List[str] = None, sizes: List[str] = None, user_min_budget: float = None, user_max_budget: float = None):
    """
    RAG-based product search using pre-optimized query
    
    Args:
        search_query: Pre-optimized search query
        gender: Gender preference (defaults to None)
        excluded_product_ids: List of product IDs to exclude from search
        allowed_product_types: Pre-extracted product categories (optional)
        
    Returns:
        List of dicts with product_id, relevance_score, and text_description
    """
    
    # Use pre-extracted categories if provided, otherwise extract them
    if allowed_product_types is None:
        category_response = await unified_product_category_extractor.get_product_category(search_query, gender or 'female')
        allowed_product_types = category_response.get('garment_type', [])
    
    # Generate embedding for the search query
    embedding_response = await vo_client.embed([search_query], model="voyage-3-large", input_type="query")
    query_embedding = embedding_response.embeddings[0]

    # Create filter for product types
    if len(allowed_product_types) == 1:
        # Use simple term search for single product type
        product_type_filter = f"@product_type:{allowed_product_types[0]}"
    else:
        # For multiple types, create OR filter
        filters = [f"@product_type:{pt}" for pt in allowed_product_types]
        product_type_filter = "(" + " | ".join(filters) + ")"

    # Create exclusion filter for already seen product IDs
    exclude_filter = ""
    if excluded_product_ids:
        exclude_ids_filter = " ".join([f"-@product_id:{pid}" for pid in excluded_product_ids])
        exclude_filter = f" {exclude_ids_filter}"
        print(f"Excluding {len(excluded_product_ids)} product IDs: {excluded_product_ids[:5]}...")

    # Build combined filter using the new utility function
    combined_filter = build_combined_filter(
        allowed_product_types=allowed_product_types,
        sizes=sizes,
        user_min_budget=user_min_budget,
        user_max_budget=user_max_budget,
        excluded_ids=excluded_product_ids
    )
    
    print(f"🔍 Applied Redis filters: {combined_filter}")
    
    # Create filtered Redis query combining all filters and vector search
    filtered_redis_query = (
        Query(f'({combined_filter})=>[KNN 50 @vector $query_vector AS vector_score]')
        .sort_by('vector_score')
        .return_fields('vector_score', 'product_id', 'text_description', 'product_type', 'size_options', 
                       'min_price', 'max_price', 'price1', 'price2', 'price3', 'price4', 'price5', 'price6')
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
        
        # Pre-extract product_ids to avoid repeated dictionary lookups
        product_ids = [doc['product_id'] for doc in search_results]
        
        return [
            {
                'product_id': product_ids[r.index],
                'relevance_score': r.relevance_score,
                'text_description': r.document
            }
            for r in reranked_results.results
        ]
    
    return []

async def fetch_product_pitch(customer_intent: str, product: dict):
    """
    Generate AI-powered product pitch for a single product
    
    Args:
        customer_intent: Original customer search intent
        product: Product dictionary with id, title, description
        
    Returns:
        Dict with product_pitch, match flag, and reasoning
    """
    product_pitch_agent_system_prompt = f'''
    You are an expert AI copywriter for a fashion e-commerce platform.

    Your job is to generate a short, catchy, benefit-focused **one-line pitch** for a given product based on a **user's intent**.

    Below is user intent
    - {customer_intent}
    Below is the product
    - {json.dumps(product)}

    Your task:
    1. Understand the user's intent based on the customer_intent. 
    2. Write short one liner pitches in the style of product taglines. Not more than twenty words.It should
      highlight the most appealing, relevant benefits of the product in relation to the query.
    3. Determine if this product is an **exact match** to the user's query. Set the flag to `true` if it directly fits the query intent and context; otherwise, `false`.

    Keep the pitch:
    - Short, friendly, and easy to scan.
    - Conversational and persuasive and creative.
    - Like a tag line instead of sales pitch.
    - If its not a match for the customer_intent , acknowledge that in the pitch.
    - Not misleading (don't make claims that the product doesn't support).
    - Use variety in your language and make it a creative pitch.
    - Pitch the product even if it is not a match by using sales psychology.
    - Never say the product is not the right match for the user.

    Also in the output json, identify if the product matches the customer intent or not.
    give reason for match or not

    Output format (JSON):
    {{
      "product_pitch": "Perfect party blouse with statement sleeves.",
      "match": true/false,
      "reason":..
    }}
    '''
    try:
        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": product_pitch_agent_system_prompt
                }
            ],
            temperature=1,
            response_format={"type": "json_object"},
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e), "product_pitch": None, "exact_match": False}

async def generate_product_pitches(customer_intent: str, search_results: list):
    """
    Generate product pitches for all search results in parallel
    
    Args:
        customer_intent: Original customer search intent
        search_results: List of dicts from search_products function
        
    Returns:
        Dict mapping product_id to pitch results
    """
    # Prepare products for pitch generation
    products = [
        {
            "id": result["product_id"],
            "title": f"Product {result['product_id']}",  # You can enhance this with actual titles
            "description": result["text_description"]
        }
        for result in search_results
    ]
    
    # Launch all pitch generation tasks in parallel for efficiency
    tasks = {
        product["id"]: asyncio.create_task(fetch_product_pitch(customer_intent, product))
        for product in products
    }
    
    # Gather results and return mapping by product ID
    results = {}
    for product_id, task in tasks.items():
        try:
            results[product_id] = await task
        except Exception as e:
            results[product_id] = {"error": str(e), "product_pitch": None, "exact_match": False}
    
    return results

async def search_and_pitch(optimized_query: str, gender: str = None, conversation_id: str = None, conversation_redis_client=None, allowed_product_types: List[str] = None, sizes: List[str] = None, user_min_budget: float = None, user_max_budget: float = None, excluded_ids: List[str] = None):
    """
    Complete RAG pipeline: search products and generate pitches with filtering
    
    This is the main entry point for the RAG-based search flow.
    It combines vector search, reranking, and AI-generated pitches.
    
    Args:
        optimized_query: Pre-optimized search query from groq_utils
        gender: Gender preference (defaults to None)
        conversation_id: Conversation ID to get excluded products
        allowed_product_types: List of product types to filter by
        sizes: List of sizes to filter by
        user_min_budget: Minimum price range
        user_max_budget: Maximum price range
        excluded_ids: List of product IDs to exclude
        conversation_redis_client: Redis client for conversation data
        
    Returns:
        Dict with pitches mapping (product_id -> pitch_data) including product images
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
    search_results = await search_products_with_query(
        optimized_query, 
        gender, 
        excluded_product_ids, 
        allowed_product_types, 
        sizes, 
        user_min_budget, 
        user_max_budget
    )
    
    if not search_results:
        return {"search_results": [], "pitches": {}}
    
    # Step 3: Generate AI-powered pitches for all results in parallel using unified generator
    from unified_pitch_generator import unified_pitch_generator
    pitches = await unified_pitch_generator.generate_pitches_from_search_results(optimized_query, search_results)
    
    # Step 4: Add product images to the pitch results
    for product_id, pitch in pitches.items():
        if product_id in data_loader.product_image_map:
            pitch['image_url'] = data_loader.product_image_map[product_id]
    
    return {
        "pitches": pitches
    }