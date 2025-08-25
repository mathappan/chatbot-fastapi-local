import os
from dotenv import load_dotenv, find_dotenv
import voyageai
import redis
from typing import List
from redis.commands.search.query import Query
import numpy as np
import asyncio
import json
from redis_filter_utils import build_combined_filter


# Load environment variables once
_ = load_dotenv(find_dotenv('.env.txt'))

# Initialize clients as module-level singletons
VOYAGE_API_KEY = os.environ['VOYAGER_API_KEY']
voyageai.api_key = VOYAGE_API_KEY
vo_client = voyageai.AsyncClient()

from groq import AsyncGroq
groq_client = AsyncGroq(api_key=os.environ.get('GROQ_API_KEY'))

# Connection pool for Redis with optimized settings
redis_client = redis.Redis(
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
product_categories = ['Plus Size Suits', 'Sarees', 'Kurta Sets', 'Co-Ords', 'Unstitched Suits', 'Plus Size Kurtas', 'Loungewear', 'Dresses', 'Plus Size Kurta Sets', 
                      'Kurta', 'Suits', 'Plus Size Short Kurtis', 'Short Kurtis', 'Bottoms', 'Plus Size Kurtis', 'Kurtis', 'Plus Size Co-Ords', 'Lehengas', 'Shirts', 
                      'Unknown']
# Cached prompt template
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

high_level_user_query_extraction_prompt = f'''
You are an expert apparel salesperson. Your task is to extract only one things from a user's natural language query:
"
1. `garment_type`: All possible types of clothing the user is referring to. 
This can be a single item or multiple. The types of apparel available in the store are {product_categories}. 
You will only choose from the above product types
Return as a list, even if only one is present.

Point to Note - The user might not specify exact type. Being a salesperson, you have to give the type of apparel that might fit the user query

Only return the extracted result in this JSON format:

{{
  "garment_type": [...]
}}

'''

async def get_product_category_rag(user_query, gender):
    
    completion = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": high_level_user_query_extraction_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_query},
                ]
            },
            {
                "role": "assistant",
                "content": "```json"
            }
        ],
        stop="```",
    )

    user_query_json_garment = json.loads(completion.choices[0].message.content)
    return user_query_json_garment

# Direct string formatting - simple and fast enough
def get_formatted_prompt_rag(customer_intent: str) -> str:
    """Format prompt template with customer intent"""
    return SEARCH_QUERY_CREATION_PROMPT.format(customer_intent=customer_intent)

async def rag_search_products(customer_intent: str, gender: str = None, sizes: List[str] = None, user_min_budget: float = None, user_max_budget: float = None, excluded_ids: List[str] = None):
    """
    RAG-based product search using vector embeddings with size and price filtering
    Args:
        customer_intent: Customer's search intent/query
        gender: User's gender preference
        sizes: List of sizes to filter by (e.g., ['s', 'm', 'l'])
        user_min_budget: Minimum price range
        user_max_budget: Maximum price range
        excluded_ids: List of product IDs to exclude from search
    Returns:
        List of dicts with product_id, relevance_score, and text_description
    """
    # Generate search query
    search_query_response = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Strictly follow instructions."},
            {"role": "user", "content": get_formatted_prompt_rag(customer_intent)}
        ],
        temperature=0.3,
        max_tokens=100,
    )
    
    search_query = search_query_response.choices[0].message.content.strip()
    print("RAG search query - ", search_query)
    category_response = await get_product_category_rag(customer_intent, gender or 'female')
    allowed_product_types = category_response.get('garment_type', [])
    
    # Generate embedding
    embedding_response = await vo_client.embed([search_query], model="voyage-3-large", input_type="query")
    query_embedding = embedding_response.embeddings[0]

    # Build combined filter using the new utility function
    combined_filter = build_combined_filter(
        allowed_product_types=allowed_product_types,
        sizes=sizes,
        user_min_budget=user_min_budget,
        user_max_budget=user_max_budget,
        excluded_ids=excluded_ids
    )
    
    print(f"🔍 Applied Redis filters: {combined_filter}")

    # According to Redis docs, try this combined query format:
    filtered_redis_query = (
        Query(f'({combined_filter})=>[KNN 50 @vector $query_vector AS vector_score]')
        .sort_by('vector_score')
        .return_fields('vector_score', 'product_id', 'text_description', 'product_type', 'size_options', 
                       'min_price', 'max_price', 'price1', 'price2', 'price3', 'price4', 'price5', 'price6')
        .paging(0, 50) 
        .dialect(2)
    )
    
    # Search Redis
    query_vector_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
    search_results = redis_client.ft('idx:product_text_description_embedding').search(
        filtered_redis_query,
        {'query_vector': query_vector_bytes}
    ).docs
    
 
    # Rerank if we have results
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




async def fetch_rag_product_pitch(customer_intent: str, product: dict):
    """Fetches product pitch from Groq API for a single product using RAG approach."""
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
                # {"role": "system", "content": product_pitch_agent_system_prompt},
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

async def generate_rag_product_pitches(customer_intent: str, search_results: list):
    """
    Generate product pitches for RAG search results
    Args:
        customer_intent: Original customer search intent
        search_results: List of dicts from rag_search_products function
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
    
    # Launch all pitch generation tasks in parallel
    tasks = {
        product["id"]: asyncio.create_task(fetch_rag_product_pitch(customer_intent, product))
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

# Combined function for complete RAG search + pitch generation
async def rag_search_and_pitch(customer_intent: str, gender: str = None, sizes: List[str] = None, user_min_budget: float = None, user_max_budget: float = None, excluded_ids: List[str] = None):
    """
    Complete RAG pipeline: search products and generate pitches with filtering
    Args:
        customer_intent: Customer's search intent/query
        gender: User's gender preference
        sizes: List of sizes to filter by
        user_min_budget: Minimum price range
        user_max_budget: Maximum price range
        excluded_ids: List of product IDs to exclude
    Returns:
        Dict with search_results and pitches
    """
    # Get search results with filtering
    search_results = await rag_search_products(
        customer_intent, 
        gender, 
        sizes, 
        user_min_budget, 
        user_max_budget, 
        excluded_ids
    )
    
    if not search_results:
        return {"search_results": [], "pitches": {}}
    
    # Generate pitches for all results
    pitches = await generate_rag_product_pitches(customer_intent, search_results)
    
    return {
        "pitches": pitches
    }