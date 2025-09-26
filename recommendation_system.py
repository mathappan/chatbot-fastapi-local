"""
Complete Recommendation System

This file contains everything needed to:
1. Define store details and customer types
2. Generate realistic customer messages using DeepSeek
3. Process messages through chatbot flow
4. Extract optimized search queries

FLOW EXPLANATION:
================
1. Store Details: Contains all information about the store (products, customer types, occasions)
2. Message Generation: Uses DeepSeek to generate realistic customer messages based on store details
3. Chatbot Processing: Runs each generated message through the existing chatbot flow
4. Search Query Extraction: Gets optimized search queries from chatbot responses
5. Results Storage: Saves everything to JSON for analysis

This system helps create a dataset of realistic customer interactions and their
corresponding optimized search queries for product recommendations.
"""

import json
import asyncio
from typing import List, Dict
from openai import AsyncOpenAI
from prompts import SEARCH_QUERY_CREATION_PROMPT
from groq_utils import generate_search_query
from search_engine import search_engine
from pinecone import Pinecone

# =============================================================================
# PINECONE CONFIGURATION
# =============================================================================
# Configuration for Pinecone vector database to store generated queries

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv('.env.txt')) # read local .env file

# DeepSeek API Configuration
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable is required")
deepseek_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

PINECONE_CONFIG = {
    "api_key": os.environ.get('PINECONE_API_KEY'),  # Replace with your actual API key
    "host": "https://user-search-queries-2xbbj27.svc.aped-4627-b74a.pinecone.io",
    "namespace": "user-search-queries"
}

# Configuration for Pinecone reranked results database
PINECONE_RERANKED_CONFIG = {
    "api_key": os.environ.get('PINECONE_API_KEY'),  # Same API key
    "host": "https://reranked-results-test-2xbbj27.svc.aped-4627-b74a.pinecone.io",
    "namespace": "reranked-results"
}

# Validate Pinecone API key
if not PINECONE_CONFIG["api_key"]:
    raise ValueError("PINECONE_API_KEY environment variable is required")

# =============================================================================
# STORE CONFIGURATION
# =============================================================================
# This section defines all the store details that will be used to generate
# realistic customer messages. It includes product categories, customer types,
# typical occasions, and common preferences.

STORE_DETAILS = {
    "store_name": "Fashion Store",
    "store_type": "Women's Ethnic & Fusion Wear",
    "brand_philosophy": "Stories over Seasons, personal style over trends, comfort over appearance",
    "target_demographic": "New age Indian women who are free-spirited, independent, and style-aware",
    
    # All product categories available in the store
    # These will be used to generate messages asking for specific product types
    "product_categories": [
        "kurta_sets", "kurtas", "co_ords", "dresses", "bottoms", 
        "lehengas", "sarees", "plus_sizes", "short_kurtis", "loungewear"
    ],
    
    # Different types of customers who would shop at this store
    # Each type has different preferences and shopping behaviors
    "customer_types": [
        "working_professional", "college_student", "festival_shopper", 
        "comfort_seeker", "style_experimenter", "budget_conscious_shopper",
        "occasion_shopper", "trendy_millennial", "traditional_buyer"
    ],
    
    # Common occasions customers shop for
    # These will influence the type of messages generated
    "typical_occasions": [
        "office wear", "casual outings", "festivals", "weddings", "parties",
        "daily wear", "family gatherings", "date nights", "travel", "comfort wear"
    ],
    
    # Common preferences customers express
    # These will be mixed into the generated messages
    "common_preferences": [
        "comfortable fabrics", "vibrant colors", "traditional prints", "modern cuts",
        "affordable prices", "versatile pieces", "trendy styles", "classic designs",
        "breathable materials", "easy maintenance", "flattering fits"
    ]
}

# =============================================================================
# DEEPSEEK PROMPT FOR MESSAGE GENERATION
# =============================================================================
# This prompt instructs DeepSeek to generate realistic customer messages that would
# trigger product recommendations. The prompt is designed to create diverse,
# natural-sounding messages that cover different customer types and scenarios.

CUSTOMER_MESSAGE_GENERATION_PROMPT = """
You are an expert at generating realistic customer messages for an e-commerce chatbot.

Based on the store details provided, generate realistic customer messages that would trigger product recommendations. These should be natural, conversational messages that real customers would send when looking for products.

Store Details:
{store_details}

Generate 50 diverse customer messages that cover:
1. Different customer types (working professional, college student, festival shopper, etc.)
2. Various occasions (office, casual, festivals, weddings, etc.)
3. Different specificity levels (vague to specific)
4. Various inquiry styles (direct, descriptive, question-based)

Make the messages sound natural and conversational, like real chat messages. Include:
- Casual language and typical chat abbreviations
- Different levels of product knowledge
- Various inquiry styles
- Occasion-based queries
- Style preferences
- Color and pattern preferences

DO NOT include:
- Size mentions (S, M, L, XL, etc.)
- Budget or price mentions
- Specific measurements

Return the result in this JSON format:
{{
    "customer_messages": [
        "hey, looking for something ethnic for office wear",
        "need a kurta set for my sister's wedding",
        "show me some casual dresses",
        "want something traditional for festivals"
    ]
}}

Make sure the messages are diverse, realistic, and would actually trigger the product recommendation flow in a chatbot.
"""

# =============================================================================
# DETAILED MESSAGE GENERATION PROMPT
# =============================================================================
# This prompt generates more detailed messages with specific attributes like
# sleeve types, necklines, fits, patterns, etc. for more comprehensive testing

DETAILED_CUSTOMER_MESSAGE_GENERATION_PROMPT = """
You are an expert at generating detailed, realistic customer messages for an e-commerce chatbot.

Based on the store details provided, generate realistic customer messages that include BOTH the product category AND specific clothing attributes. These should be natural, conversational messages that mention the product type along with detailed features.

Store Details:
{store_details}

Generate 50 diverse customer messages that include:

PRODUCT CATEGORIES (must include one): kurta_sets, kurtas, co_ords, dresses, bottoms, lehengas, sarees, short_kurtis, loungewear

ATTRIBUTES - Think broadly about ALL possible clothing attributes customers would ask for. The examples below are just EXAMPLES, not limitations. Use your knowledge of ethnic wear, fashion, and customer behavior to create diverse, realistic messages:

EXAMPLE ATTRIBUTES (not exhaustive):
- Occasions: office, wedding, festival, party, casual, daily wear, travel
- Fabrics: cotton, silk, chiffon, georgette, linen, crepe, velvet, muslin
- Colors: any color you can think of
- Patterns: floral, geometric, paisley, solid, printed, embroidered, bandhani, block print
- Fits: loose, tight, relaxed, oversized, fitted, flowy, A-line, straight
- Lengths: mini, midi, maxi, knee-length, ankle-length, floor-length
- Styles: casual, formal, traditional, contemporary, fusion, ethnic, western
- Necklines: V-neck, round neck, boat neck, high neck, square neck, off-shoulder
- Sleeves: sleeveless, short, long, 3/4, bell, puff, cap sleeves
- Embellishments: thread work, mirror work, zari work, sequins, stones, plain
- Bottom types: palazzo, churidar, straight pants, dhoti, sharara, gharara
- Seasonal: summer, monsoon, winter, breathable, warm, light
- Lifestyle: office-appropriate, travel-friendly, comfortable, easy care

DO NOT LIMIT YOURSELF TO THESE EXAMPLES. Think of ANY attribute a real customer would mention when shopping for ethnic wear. Be creative and diverse.

Make the messages sound natural and conversational. Examples (just examples, not templates):
- "looking for breathable kurtas for summer office wear"
- "need heavy work lehengas for my cousin's wedding"
- "want comfortable palazzo sets for daily wear"
- "show me festive sarees with traditional prints"

DO NOT include:
- Size mentions (S, M, L, XL, etc.)
- Budget or price mentions
- Specific measurements

Return the result in this JSON format:
{{
    "detailed_customer_messages": [
        "looking for sleeveless kurtas with floral print",
        "need cotton dresses with 3/4 sleeves for office",
        "want V-neck kurta sets in solid colors",
        "show me loose fit co-ords with geometric patterns"
    ]
}}

Make sure EVERY message includes both a product category AND specific attributes. Messages should be diverse, realistic, and test the chatbot's ability to understand detailed product requirements.
"""

# =============================================================================
# CUSTOM DEEPSEEK SEARCH QUERY GENERATION
# =============================================================================
# Custom implementation of generate_search_query functionality using DeepSeek
# instead of Groq. This function mirrors the behavior of the original function
# but uses DeepSeek's API for generating search queries.

async def generate_search_query_with_deepseek(user_messages: List[str], gender: str = "female", model: str = "deepseek-chat") -> dict:
    """
    Generate concise Google-style search queries from 20 user messages using DeepSeek.
    
    This function processes 20 user messages at once and returns all search queries
    in a JSON format using DeepSeek instead of Groq.
    
    Args:
        user_messages (List[str]): List of 20 user messages to process.
        gender (str): Gender of the user (e.g., 'female', 'male', 'unisex').
        model (str): DeepSeek model to use (defaults to "deepseek-chat").

    Returns:
        dict: JSON object containing all search queries with their corresponding messages.
    """
    
    # Validate input
    if len(user_messages) == 0 or len(user_messages) > 20:
        return {
            "error": f"Expected 1-20 user messages, but received {len(user_messages)}",
            "search_queries": []
        }
    
    # Create the batch processing prompt
    batch_prompt = f"""
You are an expert at generating concise Google-style search queries from customer messages for an ethnic wear e-commerce store.

Below are {len(user_messages)} customer messages. For each message, generate a concise, optimized search query that would help find the products the customer is looking for.

Customer Gender: {gender}
Store Type: Women's Ethnic & Fusion Wear (Fashion Store)

INSTRUCTIONS:
1. Generate exactly one search query per customer message
2. Make queries concise and Google-search style
3. Focus on product type, style, occasion, and key attributes
4. Remove unnecessary words and casual language
5. Keep queries under 10 words each

USER MESSAGES:
"""
    
    # Add numbered messages to the prompt
    for i, message in enumerate(user_messages, 1):
        batch_prompt += f"{i}. {message}\n"
    
    batch_prompt += """
Return the result in this exact JSON format:
{
    "search_queries": [
        {
            "message_number": 1,
            "original_message": "user message 1",
            "search_query": "generated search query 1"
        },
        {
            "message_number": 2,
            "original_message": "user message 2", 
            "search_query": "generated search query 2"
        }
        // ... continue for all {len(user_messages)} messages
    ]
}
"""
    
    try:
        # Call DeepSeek API
        response = await deepseek_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert at generating concise search queries from customer messages for an e-commerce store. Always return valid JSON."},
                {"role": "user", "content": batch_prompt}
            ],
            model=model,
            temperature=0.3,  # Lower temperature for more consistent JSON output
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Validate the response structure
        if "search_queries" not in result:
            return {
                "error": "Invalid response format: missing 'search_queries' key",
                "search_queries": []
            }
        
        # Ensure we have the correct number of search queries
        if len(result["search_queries"]) != len(user_messages):
            return {
                "error": f"Expected {len(user_messages)} search queries, but received {len(result['search_queries'])}",
                "search_queries": result["search_queries"]
            }
        
        return {
            "success": True,
            "total_processed": len(result["search_queries"]),
            "search_queries": result["search_queries"]
        }
        
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse JSON response: {e}",
            "search_queries": []
        }
    except Exception as e:
        return {
            "error": f"An error occurred while calling DeepSeek API: {e}",
            "search_queries": []
        }

# =============================================================================
# MESSAGE GENERATION FUNCTION
# =============================================================================
# This function uses DeepSeek to generate realistic customer messages based on the
# store details. It sends the store information and generation prompt to DeepSeek
# and expects back a JSON response with an array of customer messages.

async def generate_customer_messages():
    """
    Generate realistic customer messages for product recommendations
    
    This function:
    1. Takes the STORE_DETAILS configuration
    2. Formats it into the message generation prompt
    3. Sends it to DeepSeek for AI-powered message generation
    4. Returns a list of realistic customer messages
    
    Returns:
        dict: Contains 'customer_messages' array or 'error' if failed
    """
    
    # Convert store details to formatted string for the prompt
    store_details_str = json.dumps(STORE_DETAILS, indent=2)
    
    # Format the prompt with store details
    prompt = CUSTOMER_MESSAGE_GENERATION_PROMPT.format(store_details=store_details_str)
    
    try:
        # Call DeepSeek to generate messages
        resp = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert at generating realistic customer messages for an e-commerce chatbot."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        if not resp.choices:
            raise ValueError("No response from DeepSeek API")
        raw_response = resp.choices[0].message.content.strip()
        return json.loads(raw_response)
    except Exception as e:
        print(f"Error generating customer messages: {e}")
        return {"error": str(e)}

async def generate_detailed_customer_messages():
    """
    Generate detailed customer messages with specific attributes for product recommendations
    
    This function:
    1. Takes the STORE_DETAILS configuration
    2. Formats it into the detailed message generation prompt
    3. Sends it to DeepSeek for AI-powered detailed message generation
    4. Returns a list of detailed customer messages with specific attributes
    
    Returns:
        dict: Contains 'detailed_customer_messages' array or 'error' if failed
    """
    
    # Convert store details to formatted string for the prompt
    store_details_str = json.dumps(STORE_DETAILS, indent=2)
    
    # Format the prompt with store details
    prompt = DETAILED_CUSTOMER_MESSAGE_GENERATION_PROMPT.format(store_details=store_details_str)
    
    try:
        # Call DeepSeek to generate detailed messages
        resp = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert at generating detailed, realistic customer messages for an e-commerce chatbot."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        if not resp.choices:
            raise ValueError("No response from DeepSeek API")
        raw_response = resp.choices[0].message.content.strip()
        return json.loads(raw_response)
    except Exception as e:
        print(f"Error generating detailed customer messages: {e}")
        return {"error": str(e)}


# =============================================================================
# FULL PIPELINE PROCESSING FUNCTION
# =============================================================================
# This function processes messages through the complete pipeline using DeepSeek

async def process_messages_through_full_pipeline(messages, message_type="basic"):
    """
    Process customer messages through the full pipeline:
    1. Generate search queries using DeepSeek (in batches of 20)
    2. Extract attributes using get_search_attributes
    3. Run attribute search using run_attribute_search_for_all_product_types
    4. Map search values using map_search_values_to_catalog
    
    Args:
        messages (list): List of customer messages to process
        message_type (str): Type of messages - "basic" or "detailed"
    
    Returns:
        list: List of dictionaries containing complete processing results
    """
    
    results = []
    
    # Step 1: Generate search queries using DeepSeek batch processing
    print(f"   Step 1: Generating search queries for {len(messages)} {message_type} messages...")
    search_query_results = await process_messages_parallel_batches(messages, batch_size=20)
    
    # Step 2: Process each successful search query through the attribute pipeline
    successful_queries = [r for r in search_query_results if r['status'] == 'success']
    failed_queries = [r for r in search_query_results if r['status'] == 'error']
    
    print(f"   Step 2: Processing {len(successful_queries)} successful search queries through attribute pipeline...")
    
    for i, query_result in enumerate(successful_queries, 1):
        print(f"   Processing {message_type} query {i}/{len(successful_queries)}: {query_result['search_query']}")
        
        try:
            # Step 2a: Get search attributes from search query
            product_attributes_for_search = await search_engine.get_search_attributes(
                query_result['search_query'], "female"
            )
            
            # Step 2b: Map attributes to catalog attributes
            attribute_results_by_product_type = await search_engine.run_attribute_search_for_all_product_types(
                product_attributes_for_search
            )
            
            # Step 2c: Map search values to catalog values
            mapped_attributes_and_values = await search_engine.map_search_values_to_catalog(
                product_attributes_for_search,
                attribute_results_by_product_type
            )
            
            # Step 2d: Run shortlists (Redis-optimized, following master_search.py pattern)
            from product_matcher import product_matcher
            shortlist_results = await product_matcher.redis_run_all_shortlists(
                mapped_attributes_and_values,
                product_attributes_for_search,
                10
            )
            
            # Step 2e: Merge and sort shortlists
            sorted_shortlist_results = product_matcher.merge_and_sort_shortlists(shortlist_results)
            
            # Step 2f: Rerank shortlisted products
            reranked_shortlisted_results = await product_matcher.rerank_shortlisted_products(
                query_result['search_query'], sorted_shortlist_results
            )
            
            # Store successful result
            result = {
                "message_type": message_type,
                "original_message": query_result['original_message'],
                "search_query": query_result['search_query'],
                "product_attributes_for_search": product_attributes_for_search,
                "attribute_results_by_product_type": attribute_results_by_product_type,
                "mapped_attributes_and_values": mapped_attributes_and_values,
                "shortlist_results": shortlist_results,
                "sorted_shortlist_results": sorted_shortlist_results,
                "reranked_shortlisted_results": reranked_shortlisted_results,
                "status": "success"
            }
            
            results.append(result)
            print(f"   ✅ {message_type} query {i} completed successfully")
            
        except Exception as e:
            # Store failed result with error information
            result = {
                "message_type": message_type,
                "original_message": query_result['original_message'],
                "search_query": query_result['search_query'],
                "product_attributes_for_search": None,
                "attribute_results_by_product_type": None,
                "mapped_attributes_and_values": None,
                "shortlist_results": None,
                "sorted_shortlist_results": None,
                "reranked_shortlisted_results": None,
                "error": str(e),
                "status": "error"
            }
            results.append(result)
            print(f"   ❌ {message_type} query {i} failed: {e}")
    
    # Add failed search query results to the final results
    for failed_query in failed_queries:
        result = {
            "message_type": message_type,
            "original_message": failed_query['original_message'],
            "search_query": None,
            "product_attributes_for_search": None,
            "attribute_results_by_product_type": None,
            "mapped_attributes_and_values": None,
            "shortlist_results": None,
            "sorted_shortlist_results": None,
            "reranked_shortlisted_results": None,
            "error": failed_query.get('error', 'Search query generation failed'),
            "status": "error"
        }
        results.append(result)
    
    # Summary statistics
    successful = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] == 'error'])
    
    print(f"\n   📊 {message_type.title()} Pipeline Summary:")
    print(f"      Total messages: {len(messages)}")
    print(f"      Search queries generated: {len(successful_queries)}")
    print(f"      Successfully processed through full pipeline: {successful}")
    print(f"      Failed: {failed}")
    print(f"      Success rate: {(successful/len(messages)*100):.1f}%")
    
    return results

# =============================================================================
# CONTINUE PIPELINE FROM PREVIOUS RESULTS
# =============================================================================

async def continue_pipeline_to_rerank(processed_results, result_type="basic"):
    """
    Continue pipeline from previous results through to rerank_shortlisted_products
    
    Flow: mapped_attributes → run_all_shortlists → merge_and_sort_shortlists → rerank_shortlisted_products
    
    Args:
        processed_results (list): Previous pipeline results with mapped_attributes_and_values
        result_type (str): Type of results - "basic" or "detailed"
    
    Returns:
        list: Results with complete pipeline including rerank results
    """
    
    extended_results = []
    
    print(f"\n🔄 Continuing {result_type} pipeline to rerank_shortlisted_products...")
    print(f"   Processing {len([r for r in processed_results if r['status'] == 'success'])} successful results")
    
    for i, result in enumerate(processed_results, 1):
        if result['status'] != 'success' or not result.get('mapped_attributes_and_values'):
            # Keep failed results as-is
            extended_results.append(result)
            continue
            
        print(f"   Processing {result_type} result {i}: {result['original_message']}")
        
        try:
            # Step 1: Run shortlists (Redis-optimized, following master_search.py pattern)
            print(f"      → Step 1: Running Redis-optimized shortlists...")
            from product_matcher import product_matcher
            shortlist_results = await product_matcher.redis_run_all_shortlists(
                result['mapped_attributes_and_values'],
                result['product_attributes_for_search'],
                10
            )
            
            # Step 2: Merge and sort shortlists
            print(f"      → Step 2: Merging and sorting shortlists...")
            sorted_shortlist_results = product_matcher.merge_and_sort_shortlists(shortlist_results)
            
            # Step 3: Rerank shortlisted products
            print(f"      → Step 3: Reranking shortlisted products...")
            rerank_results = await product_matcher.rerank_shortlisted_products(
                result['search_query'], sorted_shortlist_results
            )
            
            # Create extended result with all pipeline steps
            extended_result = {
                **result,  # Include all previous data
                "shortlist_results": shortlist_results,
                "sorted_shortlist_results": sorted_shortlist_results,
                "rerank_results": rerank_results,
                "pipeline_complete": True,
                "final_products": rerank_results.get('products', []) if isinstance(rerank_results, dict) else rerank_results if isinstance(rerank_results, list) else []
            }
            
            extended_results.append(extended_result)
            print(f"      ✅ Completed full pipeline for {result_type} result {i}")
            
        except Exception as e:
            # Add error information but keep previous data
            error_result = {
                **result,  # Include all previous data
                "pipeline_error": str(e),
                "pipeline_complete": False,
                "shortlist_results": None,
                "sorted_shortlist_results": None,
                "rerank_results": None
            }
            extended_results.append(error_result)
            print(f"      ❌ Pipeline error for {result_type} result {i}: {e}")
    
    # Summary statistics
    successful_pipeline = len([r for r in extended_results if r.get('pipeline_complete')])
    failed_pipeline = len([r for r in extended_results if r.get('pipeline_error')])
    
    print(f"\n   📊 {result_type.title()} Pipeline Extension Summary:")
    print(f"      Total processed: {len(processed_results)}")
    print(f"      Successfully extended to rerank: {successful_pipeline}")
    print(f"      Failed in extension: {failed_pipeline}")
    print(f"      Success rate: {(successful_pipeline/len(processed_results)*100):.1f}%")
    
    return extended_results

async def load_and_continue_pipeline():
    """
    Load previous pipeline results and continue to rerank_shortlisted_products
    
    Returns:
        dict: Complete results with extended pipeline
    """
    
    print("🔄 Loading previous pipeline results...")
    
    # Check for existing result files
    import os
    basic_file = "basic_full_pipeline_results.json"
    detailed_file = "detailed_full_pipeline_results.json"
    
    basic_exists = os.path.exists(basic_file)
    detailed_exists = os.path.exists(detailed_file)
    
    if not basic_exists and not detailed_exists:
        print("❌ No previous pipeline results found!")
        print(f"   Looking for: {basic_file} or {detailed_file}")
        return None
    
    results = {}
    
    # Load and process basic results
    if basic_exists:
        print(f"📂 Loading basic results from {basic_file}...")
        try:
            with open(basic_file, 'r') as f:
                basic_data = json.load(f)
            
            basic_processed_results = basic_data.get('processed_results', [])
            print(f"   Found {len(basic_processed_results)} basic results")
            
            # Continue pipeline for basic results
            basic_extended_results = await continue_pipeline_to_rerank(basic_processed_results, "basic")
            
            # Create extended basic results object
            results['basic_extended_results'] = {
                "store_details": basic_data.get('store_details', {}),
                "basic_messages": basic_data.get('basic_messages', []),
                "processed_results": basic_extended_results,
                "summary": {
                    "basic_messages_generated": len(basic_data.get('basic_messages', [])),
                    "total_processed": len(basic_extended_results),
                    "successful_processing": len([r for r in basic_extended_results if r['status'] == 'success']),
                    "successful_pipeline_extension": len([r for r in basic_extended_results if r.get('pipeline_complete')]),
                    "failed_processing": len([r for r in basic_extended_results if r['status'] == 'error']),
                    "failed_pipeline_extension": len([r for r in basic_extended_results if r.get('pipeline_error')])
                }
            }
            
        except Exception as e:
            print(f"❌ Error loading basic results: {e}")
            results['basic_extended_results'] = None
    
    # Load and process detailed results
    if detailed_exists:
        print(f"📂 Loading detailed results from {detailed_file}...")
        try:
            with open(detailed_file, 'r') as f:
                detailed_data = json.load(f)
            
            detailed_processed_results = detailed_data.get('processed_results', [])
            print(f"   Found {len(detailed_processed_results)} detailed results")
            
            # Continue pipeline for detailed results
            detailed_extended_results = await continue_pipeline_to_rerank(detailed_processed_results, "detailed")
            
            # Create extended detailed results object
            results['detailed_extended_results'] = {
                "store_details": detailed_data.get('store_details', {}),
                "detailed_messages": detailed_data.get('detailed_messages', []),
                "processed_results": detailed_extended_results,
                "summary": {
                    "detailed_messages_generated": len(detailed_data.get('detailed_messages', [])),
                    "total_processed": len(detailed_extended_results),
                    "successful_processing": len([r for r in detailed_extended_results if r['status'] == 'success']),
                    "successful_pipeline_extension": len([r for r in detailed_extended_results if r.get('pipeline_complete')]),
                    "failed_processing": len([r for r in detailed_extended_results if r['status'] == 'error']),
                    "failed_pipeline_extension": len([r for r in detailed_extended_results if r.get('pipeline_error')])
                }
            }
            
        except Exception as e:
            print(f"❌ Error loading detailed results: {e}")
            results['detailed_extended_results'] = None
    
    # =============================================================================
    # UPLOAD RERANKED RESULTS TO PINECONE
    # =============================================================================
    print(f"\n📤 UPLOADING RERANKED RESULTS TO PINECONE:")
    print(f"   Host: {PINECONE_RERANKED_CONFIG['host']}")
    print(f"   Namespace: {PINECONE_RERANKED_CONFIG['namespace']}")
    
    # Ask user if they want to upload to Pinecone
    upload_choice = input("\n❓ Do you want to upload reranked results to Pinecone? (y/n): ").strip().lower()
    
    if upload_choice in ['y', 'yes']:
        # Ask user if they want to clear existing reranked records first
        clear_choice = input("\n❓ Do you want to clear all existing reranked records in Pinecone before uploading? (y/n): ").strip().lower()
        
        if clear_choice in ['y', 'yes']:
            print("🗑️ Clearing existing reranked records from Pinecone...")
            clear_result = await clear_pinecone_reranked_index()
            if clear_result["status"] == "success":
                print("✅ Pinecone reranked index cleared successfully")
            else:
                print(f"❌ Failed to clear Pinecone reranked index: {clear_result['error']}")
                print("⚠️ Proceeding with upload anyway...")
        else:
            print("📝 Keeping existing reranked records in Pinecone")
        
        upload_results = {}
        
        # Upload basic extended results
        if results.get('basic_extended_results'):
            print(f"\n📤 Uploading basic reranked results...")
            basic_extended_results = results['basic_extended_results']['processed_results']
            basic_upload_result = await upload_reranked_results_to_pinecone(basic_extended_results, "basic")
            upload_results['basic_upload'] = basic_upload_result
            
            # Add upload result to results
            results['basic_extended_results']['reranked_pinecone_upload'] = basic_upload_result
        
        # Upload detailed extended results
        if results.get('detailed_extended_results'):
            print(f"\n📤 Uploading detailed reranked results...")
            detailed_extended_results = results['detailed_extended_results']['processed_results']
            detailed_upload_result = await upload_reranked_results_to_pinecone(detailed_extended_results, "detailed")
            upload_results['detailed_upload'] = detailed_upload_result
            
            # Add upload result to results
            results['detailed_extended_results']['reranked_pinecone_upload'] = detailed_upload_result
        
        # Display upload summary
        print(f"\n📊 RERANKED RESULTS UPLOAD SUMMARY:")
        if upload_results.get('basic_upload'):
            basic_upload = upload_results['basic_upload']
            print(f"   Basic results uploaded: {basic_upload.get('uploaded_count', 0)}")
            print(f"   Basic upload status: {basic_upload.get('status', 'unknown')}")
        
        if upload_results.get('detailed_upload'):
            detailed_upload = upload_results['detailed_upload']
            print(f"   Detailed results uploaded: {detailed_upload.get('uploaded_count', 0)}")
            print(f"   Detailed upload status: {detailed_upload.get('status', 'unknown')}")
        
        print(f"   Upload host: {PINECONE_RERANKED_CONFIG['host']}")
        print(f"   Upload namespace: {PINECONE_RERANKED_CONFIG['namespace']}")
        
    else:
        print("📝 Skipping Pinecone upload for reranked results")
    
    return results

# =============================================================================
# PINECONE FUNCTIONS
# =============================================================================
# Functions for managing Pinecone vector database operations

async def clear_pinecone_index():
    """
    Clear all records from the Pinecone index in the specified namespace
    
    Returns:
        dict: Status of the clear operation
    """
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_CONFIG["api_key"])
        
        # Get the index
        index = pc.Index(host=PINECONE_CONFIG["host"])
        
        # Clear all records in the namespace
        index.delete(delete_all=True, namespace=PINECONE_CONFIG["namespace"])
        
        print(f"✅ Successfully cleared all records from namespace: {PINECONE_CONFIG['namespace']}")
        return {
            "status": "success",
            "message": f"All records cleared from namespace: {PINECONE_CONFIG['namespace']}"
        }
        
    except Exception as e:
        print(f"❌ Error clearing Pinecone index: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

async def clear_pinecone_reranked_index():
    """
    Clear all records from the Pinecone reranked results index in the specified namespace
    
    Returns:
        dict: Status of the clear operation
    """
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_RERANKED_CONFIG["api_key"])
        
        # Get the index
        index = pc.Index(host=PINECONE_RERANKED_CONFIG["host"])
        
        # Clear all records in the namespace
        index.delete(delete_all=True, namespace=PINECONE_RERANKED_CONFIG["namespace"])
        
        print(f"✅ Successfully cleared all reranked records from namespace: {PINECONE_RERANKED_CONFIG['namespace']}")
        return {
            "status": "success",
            "message": f"All reranked records cleared from namespace: {PINECONE_RERANKED_CONFIG['namespace']}"
        }
        
    except Exception as e:
        print(f"❌ Error clearing Pinecone reranked index: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

async def upload_reranked_results_to_pinecone(extended_results, result_type="general"):
    """
    Upload extended pipeline results with reranked products to Pinecone vector database
    
    Args:
        extended_results (list): List of extended results containing reranked products
        result_type (str): Type of results - "basic", "detailed", or "general"
    
    Returns:
        dict: Status of upload operation
    """
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_RERANKED_CONFIG["api_key"])
        
        # Get the index
        index = pc.Index(host=PINECONE_RERANKED_CONFIG["host"])
        
        # Prepare records for upsert
        records = []
        successful_uploads = 0
        
        for i, result in enumerate(extended_results):
            if result.get('pipeline_complete') and result.get('rerank_results'):
                # Get final products and pipeline results
                final_products = result.get('final_products', [])
                rerank_results = result.get('rerank_results', [])
                sorted_shortlist_results = result.get('sorted_shortlist_results', [])
                shortlist_results = result.get('shortlist_results', [])
                
                # Create main record with complete pipeline result
                record = {
                    "_id": f"{result_type}_reranked_{i+1}",
                    "text": result['search_query'],
                    "original_message": result['original_message'],
                    "search_query": result['search_query'],
                    "customer_type": "recommendation_system_generated",
                    "result_type": result_type,
                    "document_id": f"{result_type}_reranked_result_{i+1}",
                    "document_title": f"Generated {result_type.title()} Reranked Result",
                    "created_at": "2025-07-11",
                    "document_type": "reranked_result",
                    "pipeline_stage": "complete",
                    "mapped_attributes_and_values_json": json.dumps(result.get('mapped_attributes_and_values', {})),
                    "product_attributes_for_search_json": json.dumps(result.get('product_attributes_for_search', {})),
                    "shortlist_results_json": json.dumps(shortlist_results),
                    "sorted_shortlist_results_json": json.dumps(sorted_shortlist_results),
                    "rerank_results_json": json.dumps(rerank_results),
                    "total_products_found": len(final_products),
                    "total_reranked": len(rerank_results) if isinstance(rerank_results, list) else 0,
                    "total_shortlisted": len(sorted_shortlist_results) if isinstance(sorted_shortlist_results, list) else 0,
                    "has_products": len(final_products) > 0
                }
                
                # Add top 5 products as separate fields for easier querying
                if final_products:
                    record["top_products_json"] = json.dumps(final_products[:5])  # Store top 5 products as JSON
                    record["product_ids"] = [str(p.get('id', '')) for p in final_products[:5]]
                    record["product_names"] = [p.get('name', '') for p in final_products[:5]]
                
                records.append(record)
        
        if records:
            # Upload records to Pinecone in batches of 50
            batch_size = 50
            total_uploaded = 0
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                try:
                    print(f"📤 Uploading reranked results batch {batch_num} ({len(batch)} records)...")
                    
                    index.upsert_records(
                        PINECONE_RERANKED_CONFIG["namespace"],
                        batch
                    )
                    
                    total_uploaded += len(batch)
                    print(f"✅ Reranked results batch {batch_num} uploaded successfully ({len(batch)} records)")
                    
                except Exception as batch_error:
                    print(f"❌ Error uploading reranked results batch {batch_num}: {batch_error}")
                    continue
            
            print(f"🎉 Reranked results upload complete! Total uploaded: {total_uploaded}/{len(records)} results")
            return {
                "status": "success",
                "uploaded_count": total_uploaded,
                "total_processed": len(extended_results),
                "total_batches": (len(records) + batch_size - 1) // batch_size,
                "pinecone_host": PINECONE_RERANKED_CONFIG["host"],
                "pinecone_namespace": PINECONE_RERANKED_CONFIG["namespace"]
            }
        else:
            print("⚠️ No completed pipeline results found to upload")
            return {
                "status": "no_data",
                "uploaded_count": 0,
                "total_processed": len(extended_results)
            }
            
    except Exception as e:
        print(f"❌ Error uploading reranked results to Pinecone: {e}")
        return {
            "status": "error",
            "error": str(e),
            "uploaded_count": 0,
            "total_processed": len(extended_results)
        }

# =============================================================================
# PINECONE UPLOAD FUNCTION
# =============================================================================
# This function uploads generated queries to Pinecone vector database

async def upload_queries_to_pinecone(processed_results, query_type="general"):
    """
    Upload processed search queries to Pinecone vector database
    
    Args:
        processed_results (list): List of processed results containing search queries
        query_type (str): Type of queries - "basic", "detailed", or "general"
    
    Returns:
        dict: Status of upload operation
    """
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_CONFIG["api_key"])
        
        # Get the index
        index = pc.Index(host=PINECONE_CONFIG["host"])
        
        # Prepare records for upsert
        records = []
        successful_uploads = 0
        
        for i, result in enumerate(processed_results):
            if result['status'] == 'success' and result['search_query']:
                record = {
                    "_id": f"{query_type}_query_{i+1}",
                    "text": result['search_query'],
                    "original_message": result['original_message'],
                    "customer_type": "recommendation_system_generated",
                    "query_type": query_type,
                    "document_id": f"{query_type}_search_query_{i+1}",
                    "document_title": f"Generated {query_type.title()} Search Query",
                    "created_at": "2025-07-10",
                    "document_type": "search_query",
                    "mapped_attributes_and_values": result.get('mapped_attributes_and_values', {})
                }
                records.append(record)
        
        if records:
            # Upload records to Pinecone in batches of 50 (reduced from 96 to avoid API limit)
            batch_size = 50
            total_uploaded = 0
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                try:
                    print(f"📤 Uploading batch {batch_num} ({len(batch)} records)...")
                    
                    index.upsert_records(
                        PINECONE_CONFIG["namespace"],
                        batch
                    )
                    
                    total_uploaded += len(batch)
                    print(f"✅ Batch {batch_num} uploaded successfully ({len(batch)} records)")
                    
                except Exception as batch_error:
                    print(f"❌ Error uploading batch {batch_num}: {batch_error}")
                    continue
            
            print(f"🎉 Upload complete! Total uploaded: {total_uploaded}/{len(records)} queries")
            return {
                "status": "success",
                "uploaded_count": total_uploaded,
                "total_processed": len(processed_results),
                "total_batches": (len(records) + batch_size - 1) // batch_size
            }
        else:
            print("⚠️ No successful queries found to upload")
            return {
                "status": "no_data",
                "uploaded_count": 0,
                "total_processed": len(processed_results)
            }
            
    except Exception as e:
        print(f"❌ Error uploading to Pinecone: {e}")
        return {
            "status": "error",
            "error": str(e),
            "uploaded_count": 0,
            "total_processed": len(processed_results)
        }

# =============================================================================
# RESULTS STORAGE FUNCTION
# =============================================================================
# This function saves all the processed results to a JSON file for later analysis

async def save_results_to_file(results, filename="recommendation_system_results.json"):
    """
    Save processed results to a file
    
    Args:
        results (dict): Complete results including store details, messages, and processing results
        filename (str): Name of the file to save results to
    """
    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

# =============================================================================
# PARALLEL SEARCH QUERY GENERATION FUNCTION
# =============================================================================
# This function processes messages in parallel batches for search query generation only

def create_batches(messages, batch_size=20):
    """
    Create batches of messages
    
    Args:
        messages (list): List of customer messages
        batch_size (int): Size of each batch (default: 20)
    
    Returns:
        list: List of batches
    """
    batches = []
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        batches.append(batch)
    return batches

async def process_messages_parallel_batches(messages, batch_size=20):
    """
    Process customer messages in parallel batches using DeepSeek
    
    Args:
        messages (list): List of customer messages to process
        batch_size (int): Number of messages to process in each batch (default: 20)
    
    Returns:
        list: List of dictionaries containing original message, search query, and status
    """
    
    # Create batches
    batches = create_batches(messages, batch_size)
    total_messages = len(messages)
    total_batches = len(batches)
    
    print(f"🔄 Processing {total_messages} messages in {total_batches} batches of {batch_size}")
    
    # Process all batches in parallel
    batch_tasks = []
    for batch in batches:
        task = generate_search_query_with_deepseek(batch, "female")
        batch_tasks.append(task)
    
    # Wait for all batches to complete
    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
    
    # Process results
    all_results = []
    for batch_num, result in enumerate(batch_results):
        batch = batches[batch_num]
        print(f"\n📦 Processing batch {batch_num + 1}/{total_batches} ({len(batch)} messages)...")
        
        if isinstance(result, Exception):
            # Handle exception for entire batch
            for message in batch:
                error_result = {
                    "original_message": message,
                    "search_query": None,
                    "status": "error",
                    "error": str(result)
                }
                all_results.append(error_result)
            print(f"❌ Batch {batch_num + 1} failed: {result}")
        
        elif result.get("success") and result.get("search_queries"):
            # Handle successful batch
            for query_data in result["search_queries"]:
                batch_result = {
                    "original_message": query_data["original_message"],
                    "search_query": query_data["search_query"],
                    "status": "success"
                }
                all_results.append(batch_result)
            print(f"✅ Batch {batch_num + 1} completed")
        
        else:
            # Handle batch error
            error_msg = result.get("error", "Batch processing failed")
            for message in batch:
                error_result = {
                    "original_message": message,
                    "search_query": None,
                    "status": "error",
                    "error": error_msg
                }
                all_results.append(error_result)
            print(f"❌ Batch {batch_num + 1} failed: {error_msg}")
    
    # Summary statistics
    successful = len([r for r in all_results if r['status'] == 'success'])
    failed = len([r for r in all_results if r['status'] == 'error'])
    
    print(f"\n📊 PARALLEL PROCESSING COMPLETE:")
    print(f"   Total messages processed: {len(all_results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {(successful/len(all_results)*100):.1f}%")
    
    return all_results


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================
# This is the main orchestration function that runs the complete flow

async def main():
    """
    Main function to run the complete recommendation system
    
    COMPLETE FLOW:
    1. Display store configuration
    2. Generate realistic customer messages using DeepSeek
    3. Process messages through full pipeline (search queries + attribute extraction)
    4. Save results to JSON files
    5. Upload search queries to Pinecone (optional)
    6. Display summary statistics
    
    This creates complete datasets with:
    - Original customer messages
    - Generated search queries
    - Extracted attributes and mapped values
    - Processing status for each message
    """
    
    print("🎯 Starting Complete Recommendation System")
    print("=" * 50)
    
    # =============================================================================
    # INITIAL CHOICE: NEW RUN OR RESTART FROM PREVIOUS
    # =============================================================================
    
    print(f"\n🔄 INITIAL OPTIONS:")
    print("   1. Generate new messages and run full pipeline")
    print("   2. Generate messages and search queries only (no attribute extraction)")
    print("   3. Restart from previous pipeline results + continue to rerank_shortlisted_products")
    
    initial_choice = input("\n❓ Choose option (1/2/3): ").strip()
    
    if initial_choice == "2":
        print("🚀 Generating messages and search queries only...")
        # This will continue to the message generation and then automatically use option 3
        
    elif initial_choice == "3":
        print("🔄 Restarting from previous pipeline results...")
        
        # Load and continue pipeline from previous results
        extended_results = await load_and_continue_pipeline()
        
        if extended_results is None:
            print("❌ Cannot continue without previous results. Exiting...")
            return None
        
        # Save extended results to new files
        if extended_results.get('basic_extended_results'):
            await save_results_to_file(
                extended_results['basic_extended_results'], 
                "basic_extended_pipeline_results.json"
            )
            print("✅ Basic extended results saved to: basic_extended_pipeline_results.json")
        
        if extended_results.get('detailed_extended_results'):
            await save_results_to_file(
                extended_results['detailed_extended_results'], 
                "detailed_extended_pipeline_results.json"
            )
            print("✅ Detailed extended results saved to: detailed_extended_pipeline_results.json")
        
        # Display summary
        print(f"\n📊 PIPELINE EXTENSION SUMMARY:")
        
        if extended_results.get('basic_extended_results'):
            basic_summary = extended_results['basic_extended_results']['summary']
            print(f"   Basic Results:")
            print(f"      Total processed: {basic_summary['total_processed']}")
            print(f"      Successfully extended to rerank: {basic_summary['successful_pipeline_extension']}")
            print(f"      Failed in extension: {basic_summary['failed_pipeline_extension']}")
        
        if extended_results.get('detailed_extended_results'):
            detailed_summary = extended_results['detailed_extended_results']['summary']
            print(f"   Detailed Results:")
            print(f"      Total processed: {detailed_summary['total_processed']}")
            print(f"      Successfully extended to rerank: {detailed_summary['successful_pipeline_extension']}")
            print(f"      Failed in extension: {detailed_summary['failed_pipeline_extension']}")
        
        # Show sample results
        print(f"\n🎯 Sample Extended Results:")
        
        if extended_results.get('basic_extended_results'):
            basic_results = extended_results['basic_extended_results']['processed_results']
            complete_basic = [r for r in basic_results if r.get('pipeline_complete')]
            if complete_basic:
                sample = complete_basic[0]
                print(f"\n📝 Basic Sample:")
                print(f"   Original: {sample['original_message']}")
                print(f"   Search Query: {sample['search_query']}")
                print(f"   Final Products: {len(sample.get('final_products', []))} products")
        
        if extended_results.get('detailed_extended_results'):
            detailed_results = extended_results['detailed_extended_results']['processed_results']
            complete_detailed = [r for r in detailed_results if r.get('pipeline_complete')]
            if complete_detailed:
                sample = complete_detailed[0]
                print(f"\n📝 Detailed Sample:")
                print(f"   Original: {sample['original_message']}")
                print(f"   Search Query: {sample['search_query']}")
                print(f"   Final Products: {len(sample.get('final_products', []))} products")
        
        print(f"\n🎉 Pipeline extension to rerank_shortlisted_products complete!")
        print(f"💡 Check the extended results files for complete product recommendations!")
        
        return extended_results
    
    elif initial_choice != "1":
        print("❌ Invalid choice. Exiting...")
        return
    
    # Continue with message generation if user chose option 1
    print("\n🚀 Proceeding with new message generation and full pipeline...")
    
    # =============================================================================
    # STEP 1: DISPLAY STORE CONFIGURATION
    # =============================================================================
    # Show the user what store details are being used for message generation
    
    print("\n1️⃣ STORE DETAILS:")
    print(f"   Store: {STORE_DETAILS['store_name']}")
    print(f"   Type: {STORE_DETAILS['store_type']}")
    print(f"   Categories: {', '.join(STORE_DETAILS['product_categories'][:5])}...")
    print(f"   Customer Types: {', '.join(STORE_DETAILS['customer_types'][:3])}...")
    
    # =============================================================================
    # STEP 2: GENERATE CUSTOMER MESSAGES
    # =============================================================================
    # Use DeepSeek to generate both basic and detailed customer messages based on store details
    # Run 2 times each in parallel for maximum variety
    
    print("\n2️⃣ GENERATING CUSTOMER MESSAGES:")
    print("   Using DeepSeek to generate realistic customer messages...")
    print("   Running 20 parallel generations for each type to get maximum variety...")
    
    # Generate basic customer messages - 20 times in parallel
    print("   2a. Generating basic customer messages (20 parallel runs)...")
    basic_tasks = [generate_customer_messages() for _ in range(20)]
    basic_results = await asyncio.gather(*basic_tasks, return_exceptions=True)
    
    # Process basic message results
    basic_messages = []
    successful_basic = 0
    for i, result in enumerate(basic_results):
        if isinstance(result, Exception):
            print(f"   ❌ Basic generation {i+1} failed: {result}")
        elif "error" in result:
            print(f"   ❌ Basic generation {i+1} error: {result['error']}")
        else:
            messages = result.get('customer_messages', [])
            basic_messages.extend(messages)
            successful_basic += 1
            print(f"   ✅ Basic generation {i+1}: {len(messages)} messages")
    
    print(f"   📊 Basic messages: {successful_basic}/20 successful runs, {len(basic_messages)} total messages")
    
    # Generate detailed customer messages - 20 times in parallel
    print("   2b. Generating detailed customer messages (20 parallel runs)...")
    detailed_tasks = [generate_detailed_customer_messages() for _ in range(20)]
    detailed_results = await asyncio.gather(*detailed_tasks, return_exceptions=True)
    
    # Process detailed message results
    detailed_messages = []
    successful_detailed = 0
    for i, result in enumerate(detailed_results):
        if isinstance(result, Exception):
            print(f"   ❌ Detailed generation {i+1} failed: {result}")
        elif "error" in result:
            print(f"   ❌ Detailed generation {i+1} error: {result['error']}")
        else:
            messages = result.get('detailed_customer_messages', [])
            detailed_messages.extend(messages)
            successful_detailed += 1
            print(f"   ✅ Detailed generation {i+1}: {len(messages)} messages")
    
    print(f"   📊 Detailed messages: {successful_detailed}/20 successful runs, {len(detailed_messages)} total messages")
    
    print(f"   🎯 TOTAL MESSAGES: {len(basic_messages) + len(detailed_messages)} (Basic: {len(basic_messages)}, Detailed: {len(detailed_messages)})")
    
    # Check if we have any messages at all
    if not basic_messages and not detailed_messages:
        print("❌ No messages generated successfully. Exiting.")
        return
    
    # Show ALL messages to the user
    print(f"\n📝 ALL BASIC MESSAGES ({len(basic_messages)} total):")
    for i, msg in enumerate(basic_messages):
        print(f"   {i+1}. {msg}")
    
    print(f"\n📝 ALL DETAILED MESSAGES ({len(detailed_messages)} total):")
    for i, msg in enumerate(detailed_messages):
        print(f"   {i+1}. {msg}")
    
    # Ask user for confirmation before proceeding
    print(f"\n🤔 REVIEW COMPLETE:")
    print(f"   Total messages generated: {len(basic_messages) + len(detailed_messages)}")
    print(f"   Basic messages: {len(basic_messages)}")
    print(f"   Detailed messages: {len(detailed_messages)}")
    
    # Auto-select option based on initial choice
    if initial_choice == "2":
        choice = "3"  # Automatically use search queries only
        print(f"\n🎯 AUTO-SELECTED: Generate and save search queries only (based on initial choice)")
    else:
        print(f"\n📋 PROCESSING OPTIONS:")
        print("   1. Full processing (generate search queries + attribute extraction + upload search queries to Pinecone)")
        print("   2. Generate search queries separately for basic and detailed messages + upload to Pinecone")
        print("   3. Generate and save search queries only (no attribute extraction or upload)")
        print("   4. Save messages only (no processing)")
        print("   5. Restart from previous pipeline results + continue to rerank_shortlisted_products")
        
        choice = input("\n❓ Choose option (1/2/3/4/5): ").strip()
    
    if choice == "2":
        print("🚀 Generating search queries separately for basic and detailed messages...")
        print("   Processing basic and detailed messages separately...")
        
        # Process basic messages separately
        print(f"\n📦 Processing {len(basic_messages)} basic messages...")
        basic_results = await process_messages_parallel_batches(basic_messages, batch_size=20)
        
        # Process detailed messages separately
        print(f"\n📦 Processing {len(detailed_messages)} detailed messages...")
        detailed_results = await process_messages_parallel_batches(detailed_messages, batch_size=20)
        
        # Save basic messages results
        basic_search_queries_results = {
            "store_details": STORE_DETAILS,
            "basic_messages": basic_messages,
            "search_queries_results": basic_results,
            "summary": {
                "basic_messages_generated": len(basic_messages),
                "total_processed": len(basic_results),
                "successful_processing": len([r for r in basic_results if r['status'] == 'success']),
                "failed_processing": len([r for r in basic_results if r['status'] == 'error']),
                "success_rate": f"{(len([r for r in basic_results if r['status'] == 'success'])/len(basic_results)*100):.1f}%" if basic_results else "0%"
            }
        }
        
        # Save detailed messages results
        detailed_search_queries_results = {
            "store_details": STORE_DETAILS,
            "detailed_messages": detailed_messages,
            "search_queries_results": detailed_results,
            "summary": {
                "detailed_messages_generated": len(detailed_messages),
                "total_processed": len(detailed_results),
                "successful_processing": len([r for r in detailed_results if r['status'] == 'success']),
                "failed_processing": len([r for r in detailed_results if r['status'] == 'error']),
                "success_rate": f"{(len([r for r in detailed_results if r['status'] == 'success'])/len(detailed_results)*100):.1f}%" if detailed_results else "0%"
            }
        }
        
        # Save to separate files
        await save_results_to_file(basic_search_queries_results, "basic_search_queries_results.json")
        await save_results_to_file(detailed_search_queries_results, "detailed_search_queries_results.json")
        
        print(f"\n✅ SEPARATE PROCESSING COMPLETE:")
        print(f"   Basic messages results saved to: basic_search_queries_results.json")
        print(f"   Detailed messages results saved to: detailed_search_queries_results.json")
        print(f"   Basic success rate: {basic_search_queries_results['summary']['success_rate']}")
        print(f"   Detailed success rate: {detailed_search_queries_results['summary']['success_rate']}")
        
        # Ask if user wants to upload to Pinecone
        upload_choice = input("\n❓ Do you want to upload these search queries to Pinecone? (y/n): ").strip().lower()
        
        if upload_choice in ['y', 'yes']:
            print(f"\n📤 UPLOADING SEARCH QUERIES TO PINECONE:")
            
            # Ask user if they want to clear existing records first
            clear_choice = input("\n❓ Do you want to clear all existing records in Pinecone before uploading? (y/n): ").strip().lower()
            
            if clear_choice in ['y', 'yes']:
                print("🗑️ Clearing existing records from Pinecone...")
                clear_result = await clear_pinecone_index()
                if clear_result["status"] == "success":
                    print("✅ Pinecone index cleared successfully")
                else:
                    print(f"❌ Failed to clear Pinecone index: {clear_result['error']}")
                    print("⚠️ Proceeding with upload anyway...")
            else:
                print("📝 Keeping existing records in Pinecone")
            
            # Upload basic results
            print("\n📤 Uploading basic search queries...")
            basic_upload_result = await upload_queries_to_pinecone(basic_results, "basic")
            
            # Upload detailed results
            print("\n📤 Uploading detailed search queries...")
            detailed_upload_result = await upload_queries_to_pinecone(detailed_results, "detailed")
            
            # Add upload results to the results
            basic_search_queries_results["pinecone_upload"] = basic_upload_result
            detailed_search_queries_results["pinecone_upload"] = detailed_upload_result
            
            # Re-save with upload results
            await save_results_to_file(basic_search_queries_results, "basic_search_queries_results.json")
            await save_results_to_file(detailed_search_queries_results, "detailed_search_queries_results.json")
            
            print(f"\n📊 PINECONE UPLOAD SUMMARY:")
            print(f"   Basic search queries uploaded: {basic_upload_result.get('uploaded_count', 0)}")
            print(f"   Detailed search queries uploaded: {detailed_upload_result.get('uploaded_count', 0)}")
            print(f"   Basic upload status: {basic_upload_result.get('status', 'unknown')}")
            print(f"   Detailed upload status: {detailed_upload_result.get('status', 'unknown')}")
        
        return {
            "basic_results": basic_search_queries_results,
            "detailed_results": detailed_search_queries_results
        }
    
    elif choice == "3":
        print("🚀 Generating and saving search queries only...")
        print("   Processing basic and detailed messages to generate search queries...")
        
        # Process basic messages to generate search queries only
        print(f"\n📦 Processing {len(basic_messages)} basic messages for search queries...")
        basic_results = await process_messages_parallel_batches(basic_messages, batch_size=20)
        
        # Process detailed messages to generate search queries only
        print(f"\n📦 Processing {len(detailed_messages)} detailed messages for search queries...")
        detailed_results = await process_messages_parallel_batches(detailed_messages, batch_size=20)
        
        # Save basic search queries results
        basic_search_queries_only_results = {
            "store_details": STORE_DETAILS,
            "basic_messages": basic_messages,
            "search_queries_results": basic_results,
            "summary": {
                "basic_messages_generated": len(basic_messages),
                "total_processed": len(basic_results),
                "successful_processing": len([r for r in basic_results if r['status'] == 'success']),
                "failed_processing": len([r for r in basic_results if r['status'] == 'error']),
                "success_rate": f"{(len([r for r in basic_results if r['status'] == 'success'])/len(basic_results)*100):.1f}%" if basic_results else "0%",
                "processing_type": "search_queries_only"
            }
        }
        
        # Save detailed search queries results
        detailed_search_queries_only_results = {
            "store_details": STORE_DETAILS,
            "detailed_messages": detailed_messages,
            "search_queries_results": detailed_results,
            "summary": {
                "detailed_messages_generated": len(detailed_messages),
                "total_processed": len(detailed_results),
                "successful_processing": len([r for r in detailed_results if r['status'] == 'success']),
                "failed_processing": len([r for r in detailed_results if r['status'] == 'error']),
                "success_rate": f"{(len([r for r in detailed_results if r['status'] == 'success'])/len(detailed_results)*100):.1f}%" if detailed_results else "0%",
                "processing_type": "search_queries_only"
            }
        }
        
        # Save to separate files
        await save_results_to_file(basic_search_queries_only_results, "basic_search_queries_only.json")
        await save_results_to_file(detailed_search_queries_only_results, "detailed_search_queries_only.json")
        
        print(f"\n✅ SEARCH QUERIES ONLY PROCESSING COMPLETE:")
        print(f"   Basic search queries saved to: basic_search_queries_only.json")
        print(f"   Detailed search queries saved to: detailed_search_queries_only.json")
        print(f"   Basic success rate: {basic_search_queries_only_results['summary']['success_rate']}")
        print(f"   Detailed success rate: {detailed_search_queries_only_results['summary']['success_rate']}")
        
        # Display sample results
        basic_successful = len([r for r in basic_results if r['status'] == 'success'])
        detailed_successful = len([r for r in detailed_results if r['status'] == 'success'])
        
        if basic_successful > 0:
            print(f"\n🎯 Sample Basic Search Query:")
            basic_successful_results = [r for r in basic_results if r['status'] == 'success']
            result = basic_successful_results[0]
            print(f"   Original: {result['original_message']}")
            print(f"   Search Query: {result['search_query']}")
        
        if detailed_successful > 0:
            print(f"\n🎯 Sample Detailed Search Query:")
            detailed_successful_results = [r for r in detailed_results if r['status'] == 'success']
            result = detailed_successful_results[0]
            print(f"   Original: {result['original_message']}")
            print(f"   Search Query: {result['search_query']}")
        
        return {
            "basic_search_queries_only": basic_search_queries_only_results,
            "detailed_search_queries_only": detailed_search_queries_only_results
        }
    
    elif choice == "4":
        print("🛑 Saving messages only...")
        
        # Save just the generated messages
        all_messages = basic_messages + detailed_messages
        messages_only_results = {
            "store_details": STORE_DETAILS,
            "basic_messages": basic_messages,
            "detailed_messages": detailed_messages,
            "all_messages": all_messages,
            "summary": {
                "basic_messages_generated": len(basic_messages),
                "detailed_messages_generated": len(detailed_messages),
                "total_messages_generated": len(all_messages),
                "status": "messages_only"
            }
        }
        
        await save_results_to_file(messages_only_results, "generated_messages_only.json")
        print("✅ Generated messages saved to: generated_messages_only.json")
        return messages_only_results
    
    elif choice == "5":
        print("🔄 Restarting from previous pipeline results...")
        
        # Load and continue pipeline from previous results
        extended_results = await load_and_continue_pipeline()
        
        if extended_results is None:
            print("❌ Cannot continue without previous results. Exiting...")
            return None
        
        # Save extended results to new files
        if extended_results.get('basic_extended_results'):
            await save_results_to_file(
                extended_results['basic_extended_results'], 
                "basic_extended_pipeline_results.json"
            )
            print("✅ Basic extended results saved to: basic_extended_pipeline_results.json")
        
        if extended_results.get('detailed_extended_results'):
            await save_results_to_file(
                extended_results['detailed_extended_results'], 
                "detailed_extended_pipeline_results.json"
            )
            print("✅ Detailed extended results saved to: detailed_extended_pipeline_results.json")
        
        # Display summary
        print(f"\n📊 PIPELINE EXTENSION SUMMARY:")
        
        if extended_results.get('basic_extended_results'):
            basic_summary = extended_results['basic_extended_results']['summary']
            print(f"   Basic Results:")
            print(f"      Total processed: {basic_summary['total_processed']}")
            print(f"      Successfully extended to rerank: {basic_summary['successful_pipeline_extension']}")
            print(f"      Failed in extension: {basic_summary['failed_pipeline_extension']}")
        
        if extended_results.get('detailed_extended_results'):
            detailed_summary = extended_results['detailed_extended_results']['summary']
            print(f"   Detailed Results:")
            print(f"      Total processed: {detailed_summary['total_processed']}")
            print(f"      Successfully extended to rerank: {detailed_summary['successful_pipeline_extension']}")
            print(f"      Failed in extension: {detailed_summary['failed_pipeline_extension']}")
        
        # Show sample results
        print(f"\n🎯 Sample Extended Results:")
        
        if extended_results.get('basic_extended_results'):
            basic_results = extended_results['basic_extended_results']['processed_results']
            complete_basic = [r for r in basic_results if r.get('pipeline_complete')]
            if complete_basic:
                sample = complete_basic[0]
                print(f"\n📝 Basic Sample:")
                print(f"   Original: {sample['original_message']}")
                print(f"   Search Query: {sample['search_query']}")
                print(f"   Final Products: {len(sample.get('final_products', []))} products")
        
        if extended_results.get('detailed_extended_results'):
            detailed_results = extended_results['detailed_extended_results']['processed_results']
            complete_detailed = [r for r in detailed_results if r.get('pipeline_complete')]
            if complete_detailed:
                sample = complete_detailed[0]
                print(f"\n📝 Detailed Sample:")
                print(f"   Original: {sample['original_message']}")
                print(f"   Search Query: {sample['search_query']}")
                print(f"   Final Products: {len(sample.get('final_products', []))} products")
        
        print(f"\n🎉 Pipeline extension to rerank_shortlisted_products complete!")
        print(f"💡 Check the extended results files for complete product recommendations!")
        
        return extended_results
    
    elif choice != "1":
        print("❌ Invalid choice. Exiting...")
        return
    
    print("🚀 Proceeding with full processing flow...")
    
    # =============================================================================
    # STEP 3: PROCESS THROUGH FULL PIPELINE SEPARATELY
    # =============================================================================
    # Process basic and detailed messages separately through the full pipeline
    
    print(f"\n3️⃣ PROCESSING THROUGH FULL PIPELINE:")
    print("   Processing basic and detailed messages separately...")
    print("   Flow: Generate search queries → Extract attributes → Map values")
    
    # Process basic messages
    print(f"\n📦 Processing {len(basic_messages)} basic messages through full pipeline...")
    basic_processed_results = await process_messages_through_full_pipeline(basic_messages, "basic")
    
    # Process detailed messages  
    print(f"\n📦 Processing {len(detailed_messages)} detailed messages through full pipeline...")
    detailed_processed_results = await process_messages_through_full_pipeline(detailed_messages, "detailed")
    
    # =============================================================================
    # STEP 4: COMPILE AND SAVE RESULTS
    # =============================================================================
    # Create comprehensive results objects and save to separate files
    
    print(f"\n4️⃣ COMPILING AND SAVING RESULTS:")
    
    # Create basic results object
    basic_full_results = {
        "store_details": STORE_DETAILS,
        "basic_messages": basic_messages,
        "processed_results": basic_processed_results,
        "summary": {
            "basic_messages_generated": len(basic_messages),
            "total_processed": len(basic_processed_results),
            "successful_processing": len([r for r in basic_processed_results if r['status'] == 'success']),
            "failed_processing": len([r for r in basic_processed_results if r['status'] == 'error']),
            "success_rate": f"{(len([r for r in basic_processed_results if r['status'] == 'success'])/len(basic_processed_results)*100):.1f}%" if basic_processed_results else "0%"
        }
    }
    
    # Create detailed results object
    detailed_full_results = {
        "store_details": STORE_DETAILS,
        "detailed_messages": detailed_messages,
        "processed_results": detailed_processed_results,
        "summary": {
            "detailed_messages_generated": len(detailed_messages),
            "total_processed": len(detailed_processed_results),
            "successful_processing": len([r for r in detailed_processed_results if r['status'] == 'success']),
            "failed_processing": len([r for r in detailed_processed_results if r['status'] == 'error']),
            "success_rate": f"{(len([r for r in detailed_processed_results if r['status'] == 'success'])/len(detailed_processed_results)*100):.1f}%" if detailed_processed_results else "0%"
        }
    }
    
    # Save to separate files
    await save_results_to_file(basic_full_results, "basic_full_pipeline_results.json")
    await save_results_to_file(detailed_full_results, "detailed_full_pipeline_results.json")
    
    print(f"   ✅ Basic pipeline results saved to: basic_full_pipeline_results.json")
    print(f"   ✅ Detailed pipeline results saved to: detailed_full_pipeline_results.json")
    
    # =============================================================================
    # STEP 4B: UPLOAD TO PINECONE
    # =============================================================================
    # Upload successful search queries to Pinecone separately
    
    # Ask if user wants to upload to Pinecone
    upload_choice = input("\n❓ Do you want to upload these search queries to Pinecone? (y/n): ").strip().lower()
    
    if upload_choice in ['y', 'yes']:
        print(f"\n📤 UPLOADING SEARCH QUERIES TO PINECONE:")
        print("   Uploading generated search queries to Pinecone vector database...")
        
        # Ask user if they want to clear existing records first
        clear_choice = input("\n❓ Do you want to clear all existing records in Pinecone before uploading? (y/n): ").strip().lower()
        
        if clear_choice in ['y', 'yes']:
            print("🗑️ Clearing existing records from Pinecone...")
            clear_result = await clear_pinecone_index()
            if clear_result["status"] == "success":
                print("✅ Pinecone index cleared successfully")
            else:
                print(f"❌ Failed to clear Pinecone index: {clear_result['error']}")
                print("⚠️ Proceeding with upload anyway...")
        else:
            print("📝 Keeping existing records in Pinecone")
        
        # Convert results to search query format for upload
        basic_search_queries = []
        for result in basic_processed_results:
            if result['status'] == 'success' and result['search_query']:
                basic_search_queries.append({
                    "original_message": result['original_message'],
                    "search_query": result['search_query'],
                    "status": "success"
                })
        
        detailed_search_queries = []
        for result in detailed_processed_results:
            if result['status'] == 'success' and result['search_query']:
                detailed_search_queries.append({
                    "original_message": result['original_message'],
                    "search_query": result['search_query'],
                    "status": "success"
                })
        
        # Upload basic results
        print("\n📤 Uploading basic search queries...")
        basic_upload_result = await upload_queries_to_pinecone(basic_search_queries, "basic")
        
        # Upload detailed results
        print("\n📤 Uploading detailed search queries...")
        detailed_upload_result = await upload_queries_to_pinecone(detailed_search_queries, "detailed")
        
        # Add upload results to the results
        basic_full_results["pinecone_upload"] = basic_upload_result
        detailed_full_results["pinecone_upload"] = detailed_upload_result
        
        # Re-save with upload results
        await save_results_to_file(basic_full_results, "basic_full_pipeline_results.json")
        await save_results_to_file(detailed_full_results, "detailed_full_pipeline_results.json")
        
        print(f"\n📊 PINECONE UPLOAD SUMMARY:")
        print(f"   Basic search queries uploaded: {basic_upload_result.get('uploaded_count', 0)}")
        print(f"   Detailed search queries uploaded: {detailed_upload_result.get('uploaded_count', 0)}")
        print(f"   Basic upload status: {basic_upload_result.get('status', 'unknown')}")
        print(f"   Detailed upload status: {detailed_upload_result.get('status', 'unknown')}")
    else:
        print("📝 Skipping Pinecone upload")
    
    # =============================================================================
    # STEP 5: DISPLAY SUMMARY AND SAMPLE RESULTS
    # =============================================================================
    # Show processing statistics and sample results
    
    basic_successful = len([r for r in basic_processed_results if r['status'] == 'success'])
    basic_failed = len([r for r in basic_processed_results if r['status'] == 'error'])
    
    detailed_successful = len([r for r in detailed_processed_results if r['status'] == 'success'])
    detailed_failed = len([r for r in detailed_processed_results if r['status'] == 'error'])
    
    total_successful = basic_successful + detailed_successful
    total_failed = basic_failed + detailed_failed
    total_messages = len(basic_messages) + len(detailed_messages)
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Total messages generated: {total_messages}")
    print(f"   Basic messages: {len(basic_messages)} (Success: {basic_successful}, Failed: {basic_failed})")
    print(f"   Detailed messages: {len(detailed_messages)} (Success: {detailed_successful}, Failed: {detailed_failed})")
    print(f"   Overall success rate: {(total_successful/total_messages*100):.1f}%")
    print(f"   Basic success rate: {(basic_successful/len(basic_messages)*100):.1f}%")
    print(f"   Detailed success rate: {(detailed_successful/len(detailed_messages)*100):.1f}%")
    print(f"   Results saved to: basic_full_pipeline_results.json & detailed_full_pipeline_results.json")
    
    # Show sample successful results
    if total_successful > 0:
        print(f"\n🎯 Sample successful results:")
        
        # Show basic sample
        if basic_successful > 0:
            print(f"\n📝 Basic Message Sample:")
            basic_successful_results = [r for r in basic_processed_results if r['status'] == 'success']
            result = basic_successful_results[0]
            print(f"   Original: {result['original_message']}")
            print(f"   Search Query: {result['search_query']}")
            print(f"   Attributes: {list(result['product_attributes_for_search'].keys())}")
        
        # Show detailed sample
        if detailed_successful > 0:
            print(f"\n📝 Detailed Message Sample:")
            detailed_successful_results = [r for r in detailed_processed_results if r['status'] == 'success']
            result = detailed_successful_results[0]
            print(f"   Original: {result['original_message']}")
            print(f"   Search Query: {result['search_query']}")
            print(f"   Attributes: {list(result['product_attributes_for_search'].keys())}")
    
    # Show sample failed results if any
    if total_failed > 0:
        print(f"\n❌ Sample failed results:")
        if basic_failed > 0:
            basic_failed_results = [r for r in basic_processed_results if r['status'] == 'error']
            result = basic_failed_results[0]
            print(f"\n   Basic Failed: {result['original_message']}")
            print(f"   Error: {result['error']}")
        
        if detailed_failed > 0:
            detailed_failed_results = [r for r in detailed_processed_results if r['status'] == 'error']
            result = detailed_failed_results[0]
            print(f"\n   Detailed Failed: {result['original_message']}")
            print(f"   Error: {result['error']}")
    
    print(f"\n🎉 Complete full pipeline processing finished!")
    print(f"💡 You now have complete attribute extraction data from realistic customer messages!")
    print(f"📁 Check the JSON files for the complete datasets with all pipeline steps.")
    
    return {
        "basic_results": basic_processed_results,
        "detailed_results": detailed_processed_results
    }

# =============================================================================
# EXECUTION ENTRY POINT
# =============================================================================
# Run the main function when script is executed directly

if __name__ == "__main__":
    # Run the complete recommendation system
    asyncio.run(main())