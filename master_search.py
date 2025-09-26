import json
from typing import Dict, List
from search_engine import search_engine
from product_matcher import product_matcher
from logger_config import logger, log_error, error_handler, log_search_query
from redis.commands.search.query import Query
from product_validator import get_product_validator
import asyncio

class MasterSearchFunction:
    """Orchestrates the complete product search pipeline."""
    
    def __init__(self):
        self.search_engine = search_engine
        self.product_matcher = product_matcher
    
    # Product validation methods moved to shared product_validator.py module
    # to eliminate code duplication across master_search.py and rag_search_flow.py
    
    async def _get_filtered_product_ids_from_redis(self, sizes: list = None, user_min_budget: float = None, user_max_budget: float = None) -> list:
        """
        Get product IDs that match size and price criteria from Redis.
        
        Args:
            sizes: List of sizes to filter by (e.g., ['s', 'm', 'l'])
            user_min_budget: Minimum price range
            user_max_budget: Maximum price range
            
        Returns:
            List of product IDs that match the criteria
        """
        try:
            # Import Redis utilities
            from redis_filter_utils import build_combined_filter
            from redis_client_manager import get_redis_client
            from redis.commands.search.query import Query
            
            # Use centralized Redis client
            redis_client = get_redis_client()
            
            # Build filter using existing utility
            combined_filter = build_combined_filter(
                allowed_product_types=None,  # No product type restriction
                sizes=sizes,
                user_min_budget=user_min_budget,
                user_max_budget=user_max_budget,
                excluded_ids=None
            )
            
            # If no specific filters, return empty list (let Redis shortlisting handle everything)
            if combined_filter == "*":
                return []
            
            # Query Redis for matching products
            query = Query(combined_filter).return_fields("internal_id")
            results = redis_client.ft("idx:product_metadata").search(query)
            
            matching_products = [doc.internal_id for doc in results.docs if hasattr(doc, 'internal_id')]
            print(f"🔍 Redis filtering found {len(matching_products)} matching products")
            return matching_products
            
        except Exception as e:
            print(f"⚠️ Error filtering products with Redis: {e}")
            return []  # Return empty list, let shortlisting handle everything
    
    @error_handler("Product Search")
    async def search_products(self, user_message: str, genders = ["female"], conversation_id: str = None, conversation_redis_client=None, allowed_product_types=None, sizes: list = None, user_min_budget: float = None, user_max_budget: float = None, max_results: int = 10) -> Dict:
        """
        Master search function that handles the complete product search pipeline with size and price filtering.
        
        Args:
            user_message (str): User's search query
            genders: List of user's gender preferences
            conversation_id (str): Conversation ID to get excluded products
            conversation_redis_client: Redis client for conversation data
            allowed_product_types: List of product types to filter by
            sizes: List of sizes to filter by (e.g., ['s', 'm', 'l'])
            user_min_budget: Minimum price range
            user_max_budget: Maximum price range
            
        Returns:
            Dict: Product recommendations
        """
        try:
            logger.info(f"Starting product search for query: {user_message[:100]}")
            
            # Step 1: Get search attributes from user query
            logger.info("🔍 Step 1: Extracting search attributes...")
            product_attributes_for_search = await self.search_engine.get_search_attributes(
                user_message, genders, allowed_product_types
            )
            logger.info(f"Extracted attributes: {list(product_attributes_for_search.keys())}")
            
            # Step 2: Map attributes to existing catalog attributes
            print("🔍 Step 2: Mapping attributes to catalog...")
            attribute_results_by_product_type = await self.search_engine.run_attribute_search_for_all_product_types(
                product_attributes_for_search
            )
            
            # Step 3: Map search values to catalog values
            logger.info("🔍 Step 3: Mapping values to catalog...")
            mapped_attributes_and_values = await self.search_engine.map_search_values_to_catalog(
                product_attributes_for_search,
                attribute_results_by_product_type
            )
            
            # Step 3.5: Get excluded product IDs from conversation history
            excluded_product_ids = []
            if conversation_id and conversation_redis_client:
                # Get both RAG and filter product IDs from conversation
                rag_key = f"conversation:{conversation_id}:search:rag:product_ids"
                filter_key = f"conversation:{conversation_id}:search:filter:product_ids"
                
                rag_ids = await conversation_redis_client.smembers(rag_key)
                filter_ids = await conversation_redis_client.smembers(filter_key)
                
                excluded_product_ids = list(set(list(rag_ids or []) + list(filter_ids or [])))
                if excluded_product_ids:
                    print(f"🔍 Found {len(excluded_product_ids)} existing product IDs to exclude from filter search")

            # Step 3.7: Size and price filtering is now handled directly by Redis shortlisting
            # No need for separate pre-filtering since redis_run_all_shortlists handles all filtering
            if sizes or user_min_budget or user_max_budget:
                print(f"🔍 Step 3.7: Size/price filtering will be applied in Redis shortlisting (sizes: {sizes}, budget: {user_min_budget}-{user_max_budget})")

            # Step 4: Shortlist products (Redis-optimized)
            # Handle both single gender (string) and multiple genders (list) for backward compatibility  
            if isinstance(genders, str):
                genders = [genders]
            elif not isinstance(genders, list) or not genders:
                genders = ["female"]  # fallback
            
            print(f"🔍 Step 4: Shortlisting products with Redis (genders: {genders})...")
            print("mapped_attributes_and_values", json.dumps(mapped_attributes_and_values, indent=4))
            print("product_attributes_for_search", json.dumps(product_attributes_for_search, indent=4))
            shortlist_results = await self.product_matcher.redis_run_all_shortlists(
                mapped_attributes_and_values,
                product_attributes_for_search,
                10,
                excluded_product_ids,
                genders
            )
            
            # Step 5: Merge and sort shortlists
            print("🔍 Step 5: Merging and sorting results...")
            sorted_shortlist_results = self.product_matcher.merge_and_sort_shortlists(shortlist_results)
            print("sorted_shortlist_results - ", sorted_shortlist_results)
            # Step 6: Rerank using semantic similarity
            print("🔍 Step 6: Reranking with semantic similarity...")
            reranked_shortlisted_results = await self.product_matcher.rerank_shortlisted_products(
                user_message, sorted_shortlist_results
            )
            print('reranked_shortlisted_results - ', json.dumps(reranked_shortlisted_results, indent=4))
            # Step 7: Limit results to max_results
            print(f"🔍 Step 7: Limiting results to {max_results}...")
            limited_results = dict(list(reranked_shortlisted_results.items())[:max_results])
            print(f"🔍 Final results: {len(limited_results)} products")
            
            # Step 8: Fetch metadata for unified format (same as RAG search)
            print(f"🔍 Step 8: Fetching product metadata for unified format...")
            unified_search_results = []
            if limited_results:
                from redis_client_manager import get_redis_client
                redis_client = get_redis_client()
                
                product_ids = list(limited_results.keys())
                query_string = " | ".join([f"@internal_id:{id_}" for id_ in product_ids])
                metadata_query = (
                    Query(query_string)
                    .return_fields("internal_id", "image_url", "product_url", "title", "max_price", "body_text")
                )
                
                try:
                    metadata_results = redis_client.ft("idx:product_metadata").search(metadata_query)
                    metadata_map = {}
                    for doc in metadata_results.docs:
                        metadata_map[doc.internal_id] = {
                            'title': getattr(doc, 'title', f"Product {doc.internal_id}"),
                            'image_url': getattr(doc, 'image_url', ''),
                            'product_url': getattr(doc, 'product_url', f"/product/{doc.internal_id}"),
                            'max_price': getattr(doc, 'max_price', 0),
                            'body_text': getattr(doc, 'body_text', '')
                        }
                    
                    # Create unified format matching RAG search
                    for product_id, relevance_score in limited_results.items():
                        metadata = metadata_map.get(product_id, {})
                        unified_search_results.append({
                            'product_id': product_id,
                            'relevance_score': relevance_score,
                            'text_description': metadata.get('body_text', ''),
                            'title': metadata.get('title', f"Product {product_id}"),
                            'image_url': metadata.get('image_url', ''),
                            'product_url': metadata.get('product_url', f"/product/{product_id}"),
                            'max_price': metadata.get('max_price', 0),
                            'body_text': metadata.get('body_text', '')
                        })
                    
                    print(f"✅ Created unified format with {len(unified_search_results)} products with metadata")
                    
                    # Step 9: Validate products against customer intent (same as RAG search)
                    print(f"🔍 Step 9: Validating {len(unified_search_results)} products against customer intent...")
                    validator = get_product_validator()
                    validated_search_results = await validator.validate_product_matches(user_message, unified_search_results)
                    unified_search_results = validated_search_results
                    print(f"✅ After validation: {len(unified_search_results)} products match customer intent")
                    
                except Exception as e:
                    print(f"⚠️ Error fetching metadata: {e}, using limited format")
                    # Fallback to basic format without metadata
                    unified_search_results = [
                        {
                            'product_id': product_id,
                            'relevance_score': relevance_score,
                            'text_description': '',
                            'title': f"Product {product_id}",
                            'image_url': '',
                            'product_url': f"/product/{product_id}",
                            'max_price': 0,
                            'body_text': ''
                        }
                        for product_id, relevance_score in limited_results.items()
                    ]
                    
                    # Step 9: Validate products against customer intent (fallback case)
                    print(f"🔍 Step 9: Validating {len(unified_search_results)} products against customer intent (fallback)...")
                    validator = get_product_validator()
                    validated_search_results = await validator.validate_product_matches(user_message, unified_search_results)
                    unified_search_results = validated_search_results
                    print(f"✅ After validation: {len(unified_search_results)} products match customer intent (fallback)")
            
            # Log successful search
            log_search_query("system", user_message, len(unified_search_results), True)
            logger.info(f"Product search completed successfully. Found {len(unified_search_results)} products.")
            
            return {
                "status": "success",
                "query": user_message,
                "products_found": len(unified_search_results),
                "product_recommendations": limited_results,
                "ranked_products": reranked_shortlisted_results,
                "formatted_response": {"search_results": unified_search_results}
            }
            
        except Exception as e:
            # Log failed search
            log_search_query("system", user_message, 0, False)
            log_error(e, "Master search function error", {"query": user_message, "genders": genders})
            return {
                "status": "error",
                "query": user_message,
                "error": str(e),
                "products_found": 0,
                "formatted_response": "I'm sorry, I encountered an issue while searching for products. Please try rephrasing your request."
            }

# Global master search instance
master_search = MasterSearchFunction()