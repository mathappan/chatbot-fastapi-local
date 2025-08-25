import json
from typing import Dict
from search_engine import search_engine
from product_matcher import product_matcher
from product_pitch import product_pitch_generator
from data_loader import data_loader
from logger_config import logger, log_error, error_handler, log_search_query

class MasterSearchFunction:
    """Orchestrates the complete product search pipeline."""
    
    def __init__(self):
        self.search_engine = search_engine
        self.product_matcher = product_matcher
        self.pitch_generator = product_pitch_generator
    
    def _filter_products_by_size_and_price(self, sizes: list = None, user_min_budget: float = None, user_max_budget: float = None) -> list:
        """
        Filter products based on size_options and prices from the enhanced JSON descriptions.
        
        Args:
            sizes: List of sizes to filter by (e.g., ['s', 'm', 'l'])
            user_min_budget: Minimum price range
            user_max_budget: Maximum price range
            
        Returns:
            List of product IDs that match the size and price criteria
        """
        matching_products = []
        
        for product_id, product_data in data_loader.standardised_jsons.items():
            # Check if product matches size criteria
            size_match = True
            if sizes:
                product_sizes = product_data.get('size_options', [])
                if not product_sizes:
                    size_match = False
                else:
                    # Check if any of the user's preferred sizes are available
                    size_match = any(size.lower() in [ps.lower() for ps in product_sizes] for size in sizes)
            
            # Check if product matches price criteria
            price_match = True
            if user_min_budget is not None or user_max_budget is not None:
                product_prices = product_data.get('prices', [])
                if not product_prices:
                    price_match = False
                else:
                    # Check if any price falls within the user's budget range
                    min_price = user_min_budget if user_min_budget is not None else 0
                    max_price = user_max_budget if user_max_budget is not None else float('inf')
                    
                    price_match = any(min_price <= price <= max_price for price in product_prices)
            
            # Include product if it matches both size and price criteria
            if size_match and price_match:
                matching_products.append(product_id)
        
        return matching_products
    
    @error_handler("Product Search")
    async def search_products(self, user_message: str, gender: str = "female", conversation_id: str = None, conversation_redis_client=None, allowed_product_types=None, sizes: list = None, user_min_budget: float = None, user_max_budget: float = None, max_results: int = 10) -> Dict:
        """
        Master search function that handles the complete product search pipeline with size and price filtering.
        
        Args:
            user_message (str): User's search query
            gender (str): User's gender preference
            conversation_id (str): Conversation ID to get excluded products
            conversation_redis_client: Redis client for conversation data
            allowed_product_types: List of product types to filter by
            sizes: List of sizes to filter by (e.g., ['s', 'm', 'l'])
            user_min_budget: Minimum price range
            user_max_budget: Maximum price range
            
        Returns:
            Dict: Product recommendations with pitches
        """
        try:
            logger.info(f"Starting product search for query: {user_message[:100]}")
            
            # Step 1: Get search attributes from user query
            logger.info("🔍 Step 1: Extracting search attributes...")
            product_attributes_for_search = await self.search_engine.get_search_attributes(
                user_message, gender, allowed_product_types
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

            # Step 3.7: Apply size and price filtering before shortlisting
            if sizes or user_min_budget or user_max_budget:
                print("🔍 Step 3.7: Applying size and price pre-filtering...")
                size_price_filtered_products = self._filter_products_by_size_and_price(
                    sizes, user_min_budget, user_max_budget
                )
                print(f"🔍 Size/price filtering: {len(size_price_filtered_products)} products match criteria")
                
                # Add filtered products to exclusion list (exclude products that don't match)
                all_product_ids = set(data_loader.standardised_jsons.keys())
                size_price_excluded = list(all_product_ids - set(size_price_filtered_products))
                excluded_product_ids.extend(size_price_excluded)
                print(f"🔍 Total exclusions after size/price filtering: {len(excluded_product_ids)}")

            # Step 4: Shortlist products
            print("🔍 Step 4: Shortlisting products...")
            shortlist_results = await self.product_matcher.run_all_shortlists(
                mapped_attributes_and_values,
                product_attributes_for_search,
                10,
                excluded_product_ids
            )
            
            # Step 5: Merge and sort shortlists
            print("🔍 Step 5: Merging and sorting results...")
            sorted_shortlist_results = self.product_matcher.merge_and_sort_shortlists(shortlist_results)
            
            # Step 6: Rerank using semantic similarity
            print("🔍 Step 6: Reranking with semantic similarity...")
            reranked_shortlisted_results = await self.product_matcher.rerank_shortlisted_products(
                user_message, sorted_shortlist_results
            )
            
            # Step 7: Generate product pitches
            print("🔍 Step 7: Generating product pitches...")
            pitches = await self.pitch_generator.generate_top_ten_pitches(
                user_message, reranked_shortlisted_results
            )
            
            # Step 8: Add product images to pitches
            for product_id, pitch in pitches.items():
                if product_id in data_loader.product_image_map:
                    pitch['image_url'] = data_loader.product_image_map[product_id]
            
            # Step 9: Prioritize and limit results (exact matches first, then non-exact, limited to max_results)
            if pitches:
                exact_matches = {}
                non_exact_matches = {}
                
                for product_id, pitch in pitches.items():
                    match_value = str(pitch.get('match', False)).lower()
                    exact_match_value = str(pitch.get('exact_match', False)).lower()
                    
                    if match_value == 'true' or exact_match_value == 'true':
                        exact_matches[product_id] = pitch
                    else:
                        non_exact_matches[product_id] = pitch
                
                # Combine in priority order: exact matches first (limited to max_results), then non-exact
                final_pitches = {}
                
                # Add exact matches first, but respect max_results limit
                added_count = 0
                for product_id, pitch in exact_matches.items():
                    if added_count < max_results:
                        final_pitches[product_id] = pitch
                        added_count += 1
                
                # Add non-exact matches if there are remaining slots
                remaining_slots = max_results - len(final_pitches)
                if remaining_slots > 0:
                    for product_id, pitch in non_exact_matches.items():
                        if len(final_pitches) < max_results:
                            final_pitches[product_id] = pitch
                
                pitches = final_pitches
                exact_count = min(len(exact_matches), max_results)
                non_exact_count = len(pitches) - exact_count
                print(f"🔍 Final results: {exact_count} exact matches + {non_exact_count} non-exact = {len(pitches)} total (limited to {max_results})")
            
            # Log successful search
            log_search_query("system", user_message, len(pitches), True)
            logger.info(f"Product search completed successfully. Found {len(pitches)} products.")
            
            return {
                "status": "success",
                "query": user_message,
                "products_found": len(pitches),
                "product_recommendations": pitches,
                "ranked_products": reranked_shortlisted_results,
                # "formatted_response": self.pitch_generator.format_product_recommendations(
                #     pitches, reranked_shortlisted_results
                "formatted_response": {"pitches": pitches}
            }
            
        except Exception as e:
            # Log failed search
            log_search_query("system", user_message, 0, False)
            log_error(e, "Master search function error", {"query": user_message, "gender": gender})
            return {
                "status": "error",
                "query": user_message,
                "error": str(e),
                "products_found": 0,
                "formatted_response": "I'm sorry, I encountered an issue while searching for products. Please try rephrasing your request."
            }

# Global master search instance
master_search = MasterSearchFunction()