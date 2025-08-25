"""
Pinecone Search Module

This module provides functionality to search for similar user queries in Pinecone
vector database. It takes a search query and returns ranked results based on
semantic similarity.

USAGE:
======
1. Search for similar queries: search_similar_queries("looking for kurtas")
2. Get top results: get_top_similar_queries("need dresses for party", top_k=5)
3. With reranking: search_with_reranking("ethnic wear for office")
"""

import os
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone

# Load environment variables
_ = load_dotenv(find_dotenv('.env.txt'))

# =============================================================================
# PINECONE CONFIGURATION
# =============================================================================
# Configuration for Pinecone vector database search

PINECONE_CONFIG = {
    "api_key": os.environ['PINECONE_API_KEY'],
    "host": "https://user-search-queries-2xbbj27.svc.aped-4627-b74a.pinecone.io",
    "namespace": "user-search-queries"
}

# Configuration for Pinecone reranked results database
PINECONE_RERANKED_CONFIG = {
    "api_key": os.environ['PINECONE_API_KEY'],
    "host": "https://reranked-results-test-2xbbj27.svc.aped-4627-b74a.pinecone.io",
    "namespace": "reranked-results"
}

# =============================================================================
# PINECONE CLIENT INITIALIZATION
# =============================================================================
# Initialize Pinecone client and index

def get_pinecone_index():
    """
    Initialize and return Pinecone index for search queries
    
    Returns:
        Pinecone Index object
    """
    pc = Pinecone(api_key=PINECONE_CONFIG["api_key"])
    index = pc.Index(host=PINECONE_CONFIG["host"])
    return index

def get_pinecone_reranked_index():
    """
    Initialize and return Pinecone index for reranked results
    
    Returns:
        Pinecone Index object
    """
    pc = Pinecone(api_key=PINECONE_RERANKED_CONFIG["api_key"])
    index = pc.Index(host=PINECONE_RERANKED_CONFIG["host"])
    return index

# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

async def search_similar_queries(search_query, top_k=5, include_metadata=True):
    """
    Search for similar queries in Pinecone vector database
    
    Args:
        search_query (str): The query to search for similar matches
        top_k (int): Number of top results to return (default: 5)
        include_metadata (bool): Whether to include metadata in results
    
    Returns:
        dict: Search results with matches and metadata
    """
    
    try:
        # Get Pinecone index
        index = get_pinecone_index()
        
        # Prepare search parameters
        search_params = {
            "namespace": PINECONE_CONFIG["namespace"],
            "query": {
                "inputs": {"text": search_query},
                "top_k": top_k
            }
        }
        
        # Add fields to retrieve if metadata is requested
        if include_metadata:
            search_params["fields"] = [
                "text", 
                "original_message", 
                "customer_type", 
                "query_type",
                "document_id", 
                "document_title", 
                "created_at", 
                "document_type",
                "mapped_attributes_and_values"
            ]
        
        # Perform search
        results = index.search(**search_params)
        
        # Process results - actual response structure has result.hits
        hits = results.get("result", {}).get("hits", [])
        processed_results = {
            "query": search_query,
            "total_matches": len(hits),
            "matches": []
        }
        
        # Extract match information from hits
        for hit in hits:
            fields = hit.get("fields", {})
            match_info = {
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "text": fields.get("text", ""),
                "original_message": fields.get("original_message", ""),
                "customer_type": fields.get("customer_type", ""),
                "query_type": fields.get("query_type", ""),
                "document_type": fields.get("document_type", ""),
                "document_id": fields.get("document_id", ""),
                "created_at": fields.get("created_at", ""),
                "mapped_attributes_and_values": fields.get("mapped_attributes_and_values", {})
            }
            processed_results["matches"].append(match_info)
        
        return {
            "status": "success",
            "results": processed_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "query": search_query
        }

async def search_with_reranking(search_query, top_k=10, top_n=3, rerank_model="pinecone-rerank-v0"):
    """
    Search with reranking for more precise results
    
    Args:
        search_query (str): The query to search for
        top_k (int): Initial number of results to retrieve
        top_n (int): Number of results to return after reranking
        rerank_model (str): Reranking model to use
    
    Returns:
        dict: Reranked search results
    """
    
    try:
        # Get Pinecone index
        index = get_pinecone_index()
        
        # Perform search with reranking
        ranked_results = index.search(
            namespace=PINECONE_CONFIG["namespace"],
            query={
                "inputs": {"text": search_query},
                "top_k": top_k
            },
            rerank={
                "model": rerank_model,
                "top_n": top_n,
                "rank_fields": ["text"]
            },
            fields=[
                "text", 
                "original_message", 
                "customer_type", 
                "query_type",
                "document_id", 
                "document_title", 
                "created_at", 
                "document_type",
                "mapped_attributes_and_values"
            ]
        )
        
        # Process reranked results - actual response structure has result.hits
        hits = ranked_results.get("result", {}).get("hits", [])
        processed_results = {
            "query": search_query,
            "reranked": True,
            "total_matches": len(hits),
            "matches": []
        }
        
        # Extract match information from hits
        for hit in hits:
            fields = hit.get("fields", {})
            match_info = {
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "rerank_score": hit.get("_rerank_score"),  # May not exist in basic search
                "text": fields.get("text", ""),
                "original_message": fields.get("original_message", ""),
                "customer_type": fields.get("customer_type", ""),
                "query_type": fields.get("query_type", ""),
                "document_type": fields.get("document_type", ""),
                "document_id": fields.get("document_id", ""),
                "created_at": fields.get("created_at", ""),
                "mapped_attributes_and_values": fields.get("mapped_attributes_and_values", {})
            }
            processed_results["matches"].append(match_info)
        
        return {
            "status": "success",
            "results": processed_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "query": search_query
        }

async def get_top_similar_queries(search_query, top_k=5):
    """
    Get top similar queries (simplified version)
    
    Args:
        search_query (str): The query to search for
        top_k (int): Number of top results to return
    
    Returns:
        list: List of similar queries with scores
    """
    
    result = await search_similar_queries(search_query, top_k, include_metadata=True)
    
    if result["status"] == "success":
        similar_queries = []
        for match in result["results"]["matches"]:
            similar_queries.append({
                "query": match["text"],
                "original_message": match["original_message"],
                "similarity_score": match["score"],
                "query_type": match.get("query_type", ""),
                "mapped_attributes_and_values": match.get("mapped_attributes_and_values", {})
            })
        return similar_queries
    else:
        return []

async def filter_relevant_results_with_groq(user_query: str, retrieved_matches: list) -> dict:
    """
    Use Groq to determine which retrieved search queries are good substitutes for the user's query
    
    Args:
        user_query (str): The original user query
        retrieved_matches (list): List of matches from Pinecone search
    
    Returns:
        dict: Results with filtered matches and corresponding rerank results
    """
    
    try:
        from groq import AsyncGroq
        import os
        import json
        
        # Initialize Groq client
        groq_client = AsyncGroq(api_key=os.environ.get('GROQ_API_KEY'))
        
        if not retrieved_matches:
            return {
                "status": "no_matches",
                "filtered_matches": [],
                "groq_reasoning": "No matches to filter"
            }
        
        # Prepare the search queries for Groq evaluation
        candidate_queries = []
        for i, match in enumerate(retrieved_matches):
            candidate_queries.append({
                "index": i,
                "search_query": match.get("search_query", ""),
                "original_message": match.get("original_message", "")
            })
        
        # Create the prompt for Groq
        groq_prompt = f"""
You are an expert at evaluating semantic similarity for e-commerce product searches.

User's Original Query: "{user_query}"

Below are {len(candidate_queries)} search queries retrieved from a database. Evaluate each one to determine if it could be a good substitute for the user's original query in finding similar products.

Search Queries:
"""
        
        for candidate in candidate_queries:
            groq_prompt += f"Index {candidate['index']}: \"{candidate['search_query']}\"\n"
        
        groq_prompt += """
EVALUATION CRITERIA:
- Consider product type similarity (kurtas, dresses, sarees, etc.)
- Consider occasion similarity (office, party, wedding, casual, etc.)  
- Consider style attributes (traditional, modern, ethnic, fusion, etc.)
- Consider fabric preferences (cotton, silk, etc.)
- Consider functionality (comfort, formal, festive, etc.)

For each search query, determine if it could lead to products that would satisfy the user's intent.

Return your response in this exact JSON format:
{
    "selected_indices": [0, 2, 4],

}

Only include indices of search queries that have strong semantic similarity for the user's shopping intent.
"""

        # Call Groq
        response = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert e-commerce search analyst. Always return valid JSON responses."
                },
                {
                    "role": "user", 
                    "content": groq_prompt
                }
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        # Parse Groq response
        groq_result = json.loads(response.choices[0].message.content)
        selected_indices = groq_result.get("selected_indices", [])
        
        # Filter matches and extract corresponding rerank results
        filtered_matches = []
        all_rerank_results = []
        
        for index in selected_indices:
            if 0 <= index < len(retrieved_matches):
                match = retrieved_matches[index]
                filtered_matches.append(match)
                
                # Extract rerank results for this match
                rerank_results = match.get("rerank_results", [])
                if rerank_results:
                    all_rerank_results.extend(rerank_results)
        
        return {
            "status": "success",
            "original_matches_count": len(retrieved_matches),
            "filtered_matches_count": len(filtered_matches),
            "filtered_matches": filtered_matches,
            "combined_rerank_results": all_rerank_results,
            "groq_analysis": groq_result.get("analysis", ""),
            "groq_reasoning": groq_result.get("reasoning", ""),
            "selected_indices": selected_indices
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "filtered_matches": retrieved_matches,  # Return all if filtering fails
            "combined_rerank_results": [],
            "groq_reasoning": f"Groq filtering failed: {str(e)}"
        }

async def search_reranked_results(search_query, top_k=5, include_metadata=True):
    """
    Search for reranked results in Pinecone vector database
    
    Args:
        search_query (str): The query to search for similar matches
        top_k (int): Number of top results to return (default: 5)
        include_metadata (bool): Whether to include metadata in results
    
    Returns:
        dict: Search results with reranked matches and metadata
    """
    
    try:
        # Get Pinecone reranked index
        index = get_pinecone_reranked_index()
        
        # Prepare search parameters
        search_params = {
            "namespace": PINECONE_RERANKED_CONFIG["namespace"],
            "query": {
                "inputs": {"text": search_query},
                "top_k": top_k
            }
        }
        
        # Add fields to retrieve if metadata is requested
        if include_metadata:
            search_params["fields"] = [
                "text", 
                "original_message", 
                "search_query",
                "customer_type", 
                "result_type",
                "document_id", 
                "document_title", 
                "created_at", 
                "document_type",
                "pipeline_stage",
                "mapped_attributes_and_values_json",
                "product_attributes_for_search_json",
                "shortlist_results_json",
                "sorted_shortlist_results_json",
                "rerank_results_json",
                "top_products_json",
                "total_products_found",
                "total_reranked",
                "total_shortlisted",
                "has_products",
                "product_ids",
                "product_names"
            ]
        
        # Perform search
        results = index.search(**search_params)
        
        # Process results - actual response structure has result.hits
        hits = results.get("result", {}).get("hits", [])
        processed_results = {
            "query": search_query,
            "total_matches": len(hits),
            "matches": []
        }
        
        # Extract match information from hits
        for hit in hits:
            fields = hit.get("fields", {})
            
            # Parse JSON fields back to objects
            import json
            mapped_attributes_and_values = {}
            product_attributes_for_search = {}
            shortlist_results = []
            sorted_shortlist_results = []
            rerank_results = []
            top_products = []
            
            try:
                mapped_attributes_and_values = json.loads(fields.get("mapped_attributes_and_values_json", "{}"))
                product_attributes_for_search = json.loads(fields.get("product_attributes_for_search_json", "{}"))
                shortlist_results = json.loads(fields.get("shortlist_results_json", "[]"))
                sorted_shortlist_results = json.loads(fields.get("sorted_shortlist_results_json", "[]"))
                rerank_results = json.loads(fields.get("rerank_results_json", "[]"))
                top_products = json.loads(fields.get("top_products_json", "[]"))
            except json.JSONDecodeError:
                pass  # Keep empty defaults if JSON parsing fails
            
            match_info = {
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "text": fields.get("text", ""),
                "original_message": fields.get("original_message", ""),
                "search_query": fields.get("search_query", ""),
                "customer_type": fields.get("customer_type", ""),
                "result_type": fields.get("result_type", ""),
                "document_type": fields.get("document_type", ""),
                "pipeline_stage": fields.get("pipeline_stage", ""),
                "document_id": fields.get("document_id", ""),
                "created_at": fields.get("created_at", ""),
                "mapped_attributes_and_values": mapped_attributes_and_values,
                "product_attributes_for_search": product_attributes_for_search,
                "shortlist_results": shortlist_results,
                "sorted_shortlist_results": sorted_shortlist_results,
                "rerank_results": rerank_results,
                "final_products": top_products,
                "total_products_found": fields.get("total_products_found", 0),
                "total_reranked": fields.get("total_reranked", 0),
                "total_shortlisted": fields.get("total_shortlisted", 0),
                "has_products": fields.get("has_products", False),
                "product_ids": fields.get("product_ids", []),
                "product_names": fields.get("product_names", [])
            }
            processed_results["matches"].append(match_info)
        
        return {
            "status": "success",
            "results": processed_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "query": search_query
        }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_search_results(results):
    """
    Format search results for display
    
    Args:
        results (dict): Search results from Pinecone
    
    Returns:
        str: Formatted string representation of results
    """
    
    if results["status"] != "success":
        return f"Error: {results['error']}"
    
    output = []
    output.append(f"Query: {results['results']['query']}")
    output.append(f"Total matches: {results['results']['total_matches']}")
    output.append("-" * 50)
    
    for i, match in enumerate(results["results"]["matches"], 1):
        output.append(f"{i}. Score: {match['score']:.4f}")
        output.append(f"   Text: {match['text']}")
        output.append(f"   Original: {match['original_message']}")
        output.append(f"   Type: {match['customer_type']}")
        output.append(f"   Query Type: {match.get('query_type', 'N/A')}")
        
        # Display mapped attributes if available
        mapped_attrs = match.get("mapped_attributes_and_values", {})
        if mapped_attrs:
            output.append(f"   Mapped Attributes:")
            for product_type, attrs in mapped_attrs.items():
                if attrs:
                    output.append(f"     {product_type}: {list(attrs.keys())}")
        else:
            output.append(f"   Mapped Attributes: None")
        output.append("")
    
    return "\n".join(output)

# =============================================================================
# PERFORMANCE COMPARISON FUNCTIONS
# =============================================================================

async def method1_direct_chatbot_flow(query: str) -> dict:
    """
    Method 1: Process query through the complete chatbot pipeline
    
    Flow: query → generate_search_query → get_search_attributes → 
          run_attribute_search → map_search_values_to_catalog →
          run_all_shortlists → merge_and_sort_shortlists → rerank_shortlisted_products
    
    Args:
        query (str): User query to process
        
    Returns:
        dict: Result containing complete pipeline results including reranked products
    """
    import time
    from groq_utils import generate_search_query
    from search_engine import search_engine
    from product_matcher import product_matcher
    
    start_time = time.time()
    
    try:
        # Step 1: Generate search query
        step1_start = time.time()
        search_query = await generate_search_query(query, "female")
        step1_time = time.time() - step1_start
        
        # Step 2: Get search attributes
        step2_start = time.time()
        product_attributes_for_search = await search_engine.get_search_attributes(search_query, "female")
        step2_time = time.time() - step2_start
        
        # Step 3: Run attribute search
        step3_start = time.time()
        attribute_results_by_product_type = await search_engine.run_attribute_search_for_all_product_types(
            product_attributes_for_search
        )
        step3_time = time.time() - step3_start
        
        # Step 4: Map search values to catalog
        step4_start = time.time()
        mapped_attributes_and_values = await search_engine.map_search_values_to_catalog(
            product_attributes_for_search,
            attribute_results_by_product_type
        )
        step4_time = time.time() - step4_start
        
        # Step 5: Run shortlists (following master_search.py pattern)
        step5_start = time.time()
        shortlist_results = await product_matcher.run_all_shortlists(
            mapped_attributes_and_values,
            product_attributes_for_search,
            10
        )
        step5_time = time.time() - step5_start
        
        # Step 6: Merge and sort shortlists
        step6_start = time.time()
        sorted_shortlist_results = product_matcher.merge_and_sort_shortlists(shortlist_results)
        step6_time = time.time() - step6_start
        
        # Step 7: Rerank shortlisted products
        step7_start = time.time()
        reranked_shortlisted_results = await product_matcher.rerank_shortlisted_products(
            search_query, sorted_shortlist_results
        )
        step7_time = time.time() - step7_start
        
        total_time = time.time() - start_time
        
        # Extract final products
        final_products = reranked_shortlisted_results if isinstance(reranked_shortlisted_results, list) else reranked_shortlisted_results.get('products', []) if isinstance(reranked_shortlisted_results, dict) else []
        
        return {
            "status": "success",
            "method": "direct_chatbot_flow",
            "original_query": query,
            "search_query": search_query,
            "product_attributes_for_search": product_attributes_for_search,
            "attribute_results_by_product_type": attribute_results_by_product_type,
            "mapped_attributes_and_values": mapped_attributes_and_values,
            "shortlist_results": shortlist_results,
            "sorted_shortlist_results": sorted_shortlist_results,
            "reranked_shortlisted_results": reranked_shortlisted_results,
            "final_products": final_products,
            "total_products_found": len(final_products),
            "total_time": total_time,
            "step_times": {
                "generate_search_query": step1_time,
                "get_search_attributes": step2_time,
                "run_attribute_search": step3_time,
                "map_search_values": step4_time,
                "run_all_shortlists": step5_time,
                "merge_and_sort_shortlists": step6_time,
                "rerank_shortlisted_products": step7_time
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "method": "direct_chatbot_flow",
            "original_query": query,
            "error": str(e),
            "total_time": time.time() - start_time
        }

async def method2_pinecone_search(query: str) -> dict:
    """
    Method 2: Search Pinecone for pre-computed reranked results with Groq filtering
    
    Flow: query → search_reranked_results → filter_relevant_results_with_groq → extract filtered rerank results
    
    Args:
        query (str): User query to search for
        
    Returns:
        dict: Result containing filtered reranked products from relevant search queries
    """
    import time
    
    start_time = time.time()
    
    try:
        # Step 1: Search for reranked results in Pinecone
        step1_start = time.time()
        search_results = await search_reranked_results(query, top_k=10, include_metadata=True)  # Get more results for filtering
        step1_time = time.time() - step1_start
        
        if search_results["status"] != "success":
            return {
                "status": "error",
                "method": "pinecone_reranked_search_with_groq",
                "original_query": query,
                "error": search_results.get("error", "Pinecone search failed"),
                "total_time": time.time() - start_time
            }
        
        matches = search_results["results"]["matches"]
        
        if not matches:
            return {
                "status": "no_matches",
                "method": "pinecone_reranked_search_with_groq",
                "original_query": query,
                "total_time": time.time() - start_time,
                "matches_found": 0
            }
        
        # Step 2: Use Groq to filter relevant results
        step2_start = time.time()
        filter_results = await filter_relevant_results_with_groq(query, matches)
        step2_time = time.time() - step2_start
        
        total_time = time.time() - start_time
        
        if filter_results["status"] == "success" and filter_results["filtered_matches"]:
            # Get the best filtered match for primary data
            best_filtered_match = filter_results["filtered_matches"][0]
            combined_rerank_results = filter_results["combined_rerank_results"]
            
            return {
                "status": "success",
                "method": "pinecone_reranked_search_with_groq",
                "original_query": query,
                "search_query": best_filtered_match.get("search_query", ""),
                "groq_filtering": {
                    "analysis": filter_results.get("groq_analysis", ""),
                    "reasoning": filter_results.get("groq_reasoning", ""),
                    "selected_indices": filter_results.get("selected_indices", []),
                    "original_matches": filter_results.get("original_matches_count", 0),
                    "filtered_matches": filter_results.get("filtered_matches_count", 0)
                },
                "best_match": {
                    "score": best_filtered_match.get("score", 0),
                    "text": best_filtered_match.get("text", ""),
                    "original_message": best_filtered_match.get("original_message", ""),
                    "result_type": best_filtered_match.get("result_type", ""),
                    "pipeline_stage": best_filtered_match.get("pipeline_stage", "")
                },
                "product_attributes_for_search": best_filtered_match.get("product_attributes_for_search", {}),
                "mapped_attributes_and_values": best_filtered_match.get("mapped_attributes_and_values", {}),
                "shortlist_results": best_filtered_match.get("shortlist_results", []),
                "sorted_shortlist_results": best_filtered_match.get("sorted_shortlist_results", []),
                "reranked_shortlisted_results": combined_rerank_results,  # Combined from all relevant matches
                "final_products": combined_rerank_results,  # Use combined rerank results as final products
                "total_products_found": len(combined_rerank_results),
                "total_reranked": len(combined_rerank_results),
                "total_shortlisted": best_filtered_match.get("total_shortlisted", 0),
                "has_products": len(combined_rerank_results) > 0,
                "total_time": total_time,
                "step_times": {
                    "pinecone_search": step1_time,
                    "groq_filtering": step2_time
                },
                "matches_found": len(matches),
                "matches_after_filtering": len(filter_results["filtered_matches"])
            }
        else:
            # If Groq filtering fails or returns no results, fall back to best match
            best_match = matches[0]
            
            return {
                "status": "success",
                "method": "pinecone_reranked_search_with_groq",
                "original_query": query,
                "search_query": best_match.get("search_query", ""),
                "groq_filtering": {
                    "analysis": "Groq filtering failed, using fallback",
                    "error": filter_results.get("error", "No relevant matches found"),
                    "fallback_used": True
                },
                "best_match": {
                    "score": best_match.get("score", 0),
                    "text": best_match.get("text", ""),
                    "original_message": best_match.get("original_message", ""),
                    "result_type": best_match.get("result_type", ""),
                    "pipeline_stage": best_match.get("pipeline_stage", "")
                },
                "product_attributes_for_search": best_match.get("product_attributes_for_search", {}),
                "mapped_attributes_and_values": best_match.get("mapped_attributes_and_values", {}),
                "shortlist_results": best_match.get("shortlist_results", []),
                "sorted_shortlist_results": best_match.get("sorted_shortlist_results", []),
                "reranked_shortlisted_results": best_match.get("rerank_results", []),
                "final_products": best_match.get("final_products", []),
                "total_products_found": best_match.get("total_products_found", 0),
                "total_reranked": best_match.get("total_reranked", 0),
                "total_shortlisted": best_match.get("total_shortlisted", 0),
                "has_products": best_match.get("has_products", False),
                "total_time": total_time,
                "step_times": {
                    "pinecone_search": step1_time,
                    "groq_filtering": step2_time
                },
                "matches_found": len(matches),
                "matches_after_filtering": 0
            }
            
    except Exception as e:
        return {
            "status": "error",
            "method": "pinecone_reranked_search_with_groq",
            "original_query": query,
            "error": str(e),
            "total_time": time.time() - start_time
        }

async def method3_pinecone_basic_search(query: str) -> dict:
    """
    Method 3: Search original Pinecone index for mapped attributes without reranking
    
    Flow: query → search_similar_queries → extract mapped_attributes_and_values (no reranking)
    
    Args:
        query (str): User query to search for
        
    Returns:
        dict: Result containing mapped attributes from similar queries (no reranked products)
    """
    import time
    
    start_time = time.time()
    
    try:
        # Search original Pinecone index for similar queries
        search_results = await search_similar_queries(query, top_k=5, include_metadata=True)
        
        total_time = time.time() - start_time
        
        if search_results["status"] == "success":
            matches = search_results["results"]["matches"]
            
            if matches:
                best_match = matches[0]  # Highest scoring match
                mapped_attributes = best_match.get("mapped_attributes_and_values", {})
                
                return {
                    "status": "success",
                    "method": "pinecone_basic_search",
                    "original_query": query,
                    "search_query": best_match.get("text", ""),
                    "best_match": {
                        "score": best_match.get("score", 0),
                        "text": best_match.get("text", ""),
                        "original_message": best_match.get("original_message", ""),
                        "query_type": best_match.get("query_type", ""),
                        "customer_type": best_match.get("customer_type", "")
                    },
                    "mapped_attributes_and_values": mapped_attributes,
                    "total_time": total_time,
                    "matches_found": len(matches),
                    # No reranked products - this method stops at mapped attributes
                    "final_products": [],
                    "total_products_found": 0,
                    "note": "Method 3 provides mapped attributes only, no product reranking"
                }
            else:
                return {
                    "status": "no_matches",
                    "method": "pinecone_basic_search",
                    "original_query": query,
                    "total_time": total_time,
                    "matches_found": 0
                }
        else:
            return {
                "status": "error",
                "method": "pinecone_basic_search",
                "original_query": query,
                "error": search_results.get("error", "Unknown error"),
                "total_time": total_time
            }
            
    except Exception as e:
        return {
            "status": "error",
            "method": "pinecone_basic_search",
            "original_query": query,
            "error": str(e),
            "total_time": time.time() - start_time
        }

async def compare_methods(query: str) -> dict:
    """
    Compare all three methods for getting search results
    
    Args:
        query (str): Query to test all methods
        
    Returns:
        dict: Comparison results with timing and performance metrics
    """
    
    print(f"\n🔍 Comparing methods for query: {query}")
    
    # Test Method 1: Direct Chatbot Flow (complete pipeline through reranking)
    print("   → Method 1: Direct Chatbot Flow (Complete Pipeline)...")
    method1_result = await method1_direct_chatbot_flow(query)
    
    # Test Method 2: Pinecone Reranked Results Search with Groq Filtering
    print("   → Method 2: Pinecone Reranked Results + Groq Filtering...")
    method2_result = await method2_pinecone_search(query)
    
    # Test Method 3: Basic Pinecone Search (mapped attributes only)
    print("   → Method 3: Basic Pinecone Search (No Reranking)...")
    method3_result = await method3_pinecone_basic_search(query)
    
    # Calculate comparison metrics
    method1_time = method1_result.get("total_time", 0)
    method2_time = method2_result.get("total_time", 0)
    method3_time = method3_result.get("total_time", 0)
    
    # Calculate speedup factors
    speedup_2vs1 = method1_time / method2_time if method2_time > 0 else 0
    speedup_3vs1 = method1_time / method3_time if method3_time > 0 else 0
    speedup_3vs2 = method2_time / method3_time if method3_time > 0 else 0
    
    # Calculate time savings
    savings_2vs1 = ((method1_time - method2_time) / method1_time) * 100 if method1_time > 0 else 0
    savings_3vs1 = ((method1_time - method3_time) / method1_time) * 100 if method1_time > 0 else 0
    savings_3vs2 = ((method2_time - method3_time) / method2_time) * 100 if method2_time > 0 else 0
    
    comparison = {
        "query": query,
        "method1_result": method1_result,
        "method2_result": method2_result,
        "method3_result": method3_result,
        "performance_metrics": {
            "method1_time": method1_time,
            "method2_time": method2_time,
            "method3_time": method3_time,
            "speedup_2vs1": speedup_2vs1,
            "speedup_3vs1": speedup_3vs1,
            "speedup_3vs2": speedup_3vs2,
            "savings_2vs1": savings_2vs1,
            "savings_3vs1": savings_3vs1,
            "savings_3vs2": savings_3vs2
        }
    }
    
    # Display results
    print(f"   ✅ Method 1 (Direct Flow - Complete Pipeline): {method1_time:.3f}s")
    print(f"   ✅ Method 2 (Pinecone + Groq Filtering): {method2_time:.3f}s")
    print(f"   ✅ Method 3 (Basic Pinecone Search): {method3_time:.3f}s")
    
    # Show Method 2 timing breakdown
    if method2_result.get("status") == "success" and "step_times" in method2_result:
        step_times = method2_result["step_times"]
        pinecone_time = step_times.get("pinecone_search", 0)
        groq_time = step_times.get("groq_filtering", 0)
        print(f"      ↳ Pinecone Search: {pinecone_time:.3f}s")
        print(f"      ↳ Groq Filtering: {groq_time:.3f}s")
    
    # Display product counts
    if method1_result.get("status") == "success":
        print(f"   🎯 Method 1 Products Found: {method1_result.get('total_products_found', 0)}")
    if method2_result.get("status") == "success":
        print(f"   🎯 Method 2 Products Found: {method2_result.get('total_products_found', 0)}")
        if "groq_filtering" in method2_result:
            groq_info = method2_result["groq_filtering"]
            if not groq_info.get("fallback_used"):
                print(f"   🤖 Groq Filtered: {groq_info.get('original_matches', 0)} → {groq_info.get('filtered_matches', 0)} matches")
    if method3_result.get("status") == "success":
        print(f"   🎯 Method 3 Products Found: {method3_result.get('total_products_found', 0)} (attributes only)")
    
    # Display speedup comparisons
    print(f"   📊 Speedups:")
    print(f"      Method 2 vs Method 1: {speedup_2vs1:.2f}x ({savings_2vs1:.1f}% faster)")
    print(f"      Method 3 vs Method 1: {speedup_3vs1:.2f}x ({savings_3vs1:.1f}% faster)")
    print(f"      Method 3 vs Method 2: {speedup_3vs2:.2f}x ({savings_3vs2:.1f}% faster)")
    
    return comparison

async def run_performance_comparison(test_queries: list = None) -> dict:
    """
    Run performance comparison across multiple queries
    
    Args:
        test_queries (list): List of queries to test. If None, uses default queries.
        
    Returns:
        dict: Complete comparison results with statistics
    """
    
    if test_queries is None:
        test_queries = [
            "looking for kurtas for office",
            "need dresses for party",
            "ethnic wear for festivals",
            "comfortable loungewear",
            "cotton kurta sets for summer",
            "formal sarees for weddings",
            "casual co-ords for daily wear",
            "traditional lehengas for ceremonies"
        ]
    
    print("🚀 Starting Performance Comparison")
    print("=" * 60)
    print(f"📋 Testing {len(test_queries)} queries")
    
    all_results = []
    method1_times = []
    method2_times = []
    method3_times = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}/{len(test_queries)}")
        
        result = await compare_methods(query)
        all_results.append(result)
        
        # Collect timing data
        if result["method1_result"].get("status") == "success":
            method1_times.append(result["method1_result"]["total_time"])
        if result["method2_result"].get("status") == "success":
            method2_times.append(result["method2_result"]["total_time"])
        if result["method3_result"].get("status") == "success":
            method3_times.append(result["method3_result"]["total_time"])
    
    # Calculate statistics
    from statistics import mean, median, stdev
    
    # Collect timing breakdown for Method 2
    pinecone_times = []
    groq_times = []
    
    for result in all_results:
        if (result["method2_result"].get("status") == "success" and 
            "step_times" in result["method2_result"]):
            step_times = result["method2_result"]["step_times"]
            pinecone_times.append(step_times.get("pinecone_search", 0))
            groq_times.append(step_times.get("groq_filtering", 0))
    
    stats = {
        "total_queries": len(test_queries),
        "successful_method1": len(method1_times),
        "successful_method2": len(method2_times),
        "successful_method3": len(method3_times),
        "method1_stats": {
            "avg_time": mean(method1_times) if method1_times else 0,
            "median_time": median(method1_times) if method1_times else 0,
            "min_time": min(method1_times) if method1_times else 0,
            "max_time": max(method1_times) if method1_times else 0,
            "std_dev": stdev(method1_times) if len(method1_times) > 1 else 0
        },
        "method2_stats": {
            "avg_time": mean(method2_times) if method2_times else 0,
            "median_time": median(method2_times) if method2_times else 0,
            "min_time": min(method2_times) if method2_times else 0,
            "max_time": max(method2_times) if method2_times else 0,
            "std_dev": stdev(method2_times) if len(method2_times) > 1 else 0,
            "pinecone_avg": mean(pinecone_times) if pinecone_times else 0,
            "groq_avg": mean(groq_times) if groq_times else 0,
            "pinecone_min": min(pinecone_times) if pinecone_times else 0,
            "pinecone_max": max(pinecone_times) if pinecone_times else 0,
            "groq_min": min(groq_times) if groq_times else 0,
            "groq_max": max(groq_times) if groq_times else 0
        },
        "method3_stats": {
            "avg_time": mean(method3_times) if method3_times else 0,
            "median_time": median(method3_times) if method3_times else 0,
            "min_time": min(method3_times) if method3_times else 0,
            "max_time": max(method3_times) if method3_times else 0,
            "std_dev": stdev(method3_times) if len(method3_times) > 1 else 0
        }
    }
    
    # Calculate overall performance metrics
    if method1_times and method2_times and method3_times:
        avg_speedup_2vs1 = mean(method1_times) / mean(method2_times)
        avg_speedup_3vs1 = mean(method1_times) / mean(method3_times)
        avg_speedup_3vs2 = mean(method2_times) / mean(method3_times)
        
        avg_savings_2vs1 = ((mean(method1_times) - mean(method2_times)) / mean(method1_times)) * 100
        avg_savings_3vs1 = ((mean(method1_times) - mean(method3_times)) / mean(method1_times)) * 100
        avg_savings_3vs2 = ((mean(method2_times) - mean(method3_times)) / mean(method2_times)) * 100
        
        stats["overall_performance"] = {
            "speedup_2vs1": avg_speedup_2vs1,
            "speedup_3vs1": avg_speedup_3vs1,
            "speedup_3vs2": avg_speedup_3vs2,
            "savings_2vs1": avg_savings_2vs1,
            "savings_3vs1": avg_savings_3vs1,
            "savings_3vs2": avg_savings_3vs2
        }
    
    # Display summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"\n⚡ Method 1 - Direct Chatbot Flow (Complete Pipeline):")
    print(f"   Average Time: {stats['method1_stats']['avg_time']:.3f}s")
    print(f"   Min/Max: {stats['method1_stats']['min_time']:.3f}s / {stats['method1_stats']['max_time']:.3f}s")
    
    print(f"\n🔍 Method 2 - Pinecone + Groq Filtering:")
    print(f"   Average Time: {stats['method2_stats']['avg_time']:.3f}s")
    print(f"   Min/Max: {stats['method2_stats']['min_time']:.3f}s / {stats['method2_stats']['max_time']:.3f}s")
    # Calculate percentages
    total_avg = stats['method2_stats']['avg_time']
    pinecone_pct = (stats['method2_stats']['pinecone_avg'] / total_avg * 100) if total_avg > 0 else 0
    groq_pct = (stats['method2_stats']['groq_avg'] / total_avg * 100) if total_avg > 0 else 0
    
    print(f"   ├─ Pinecone Avg: {stats['method2_stats']['pinecone_avg']:.3f}s ({pinecone_pct:.1f}%) [{stats['method2_stats']['pinecone_min']:.3f}s - {stats['method2_stats']['pinecone_max']:.3f}s]")
    print(f"   └─ Groq Avg: {stats['method2_stats']['groq_avg']:.3f}s ({groq_pct:.1f}%) [{stats['method2_stats']['groq_min']:.3f}s - {stats['method2_stats']['groq_max']:.3f}s]")
    
    print(f"\n⚡ Method 3 - Basic Pinecone Search (No Reranking):")
    print(f"   Average Time: {stats['method3_stats']['avg_time']:.3f}s")
    print(f"   Min/Max: {stats['method3_stats']['min_time']:.3f}s / {stats['method3_stats']['max_time']:.3f}s")
    
    if "overall_performance" in stats:
        perf = stats["overall_performance"]
        print(f"\n🏆 Overall Performance:")
        print(f"   Method 2 vs Method 1: {perf['speedup_2vs1']:.2f}x faster ({perf['savings_2vs1']:.1f}% time savings)")
        print(f"   Method 3 vs Method 1: {perf['speedup_3vs1']:.2f}x faster ({perf['savings_3vs1']:.1f}% time savings)")
        print(f"   Method 3 vs Method 2: {perf['speedup_3vs2']:.2f}x faster ({perf['savings_3vs2']:.1f}% time savings)")
    
    return {
        "test_queries": test_queries,
        "results": all_results,
        "statistics": stats,
        "timestamp": __import__('time').strftime("%Y-%m-%d %H:%M:%S")
    }

# =============================================================================
# MAIN EXECUTION FOR TESTING
# =============================================================================

async def main():
    """
    Test function to demonstrate search functionality
    """
    
    print("=Testing Pinecone Search Functionality")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "looking for kurtas for office",
        "need dresses for party",
        "ethnic wear for festivals",
        "comfortable loungewear"
    ]
    
    for query in test_queries:
        print(f"\n= Searching for: {query}")
        
        # Basic search
        results = await search_similar_queries(query, top_k=3)
        print(format_search_results(results))
        
        # Search with reranking
        print(f"\n🔄 Reranked results for: {query}")
        reranked_results = await search_with_reranking(query, top_k=5, top_n=2)
        print(format_search_results(reranked_results))
        
        print("-" * 50)

# Remove the duplicate main function above and keep only the performance comparison one

async def main():
    """
    Main function to run performance comparison and basic search tests
    """
    
    print("🎯 Pinecone Search Performance Comparison")
    print("=" * 50)
    
    # Run performance comparison
    comparison_results = await run_performance_comparison()
    
    # Save results to file
    try:
        import json
        with open('pinecone_performance_comparison.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\n💾 Results saved to pinecone_performance_comparison.json")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
    
    print("\n" + "=" * 50)
    print("🔍 Basic Search Test")
    print("=" * 50)
    
    # Basic search test
    test_query = "looking for kurtas for office"
    print(f"\n🔍 Testing basic search for: {test_query}")
    
    results = await search_similar_queries(test_query, top_k=3)
    print(format_search_results(results))

if __name__ == "__main__":
    import asyncio
    # Run performance comparison by default
    asyncio.run(main())