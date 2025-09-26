import asyncio
import uuid
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict

from clients import voyageai_client
from redis.commands.search.query import Query

class ProductMatcher:
    """Handles product shortlisting and ranking."""
    
    def __init__(self):
        # Redis client configuration - use centralized client
        from redis_client_manager import get_async_redis_client
        self.redis_client = get_async_redis_client()
    
    def shortlist_products(
        self,
        mapped_attributes_and_values: Dict,
        product_attributes_for_search: Dict,
        category: str,
        n: int = 10,
        excluded_product_ids: list = None,
    ) -> Dict:
        """Shortlist products based on search criteria."""
        # This method is deprecated - use redis_shortlist_products instead
        raise NotImplementedError("Use redis_shortlist_products instead of shortlist_products")
        
        # Remove excluded product IDs
        if excluded_product_ids:
            excluded_set = set(str(pid) for pid in excluded_product_ids)
            original_count = len(valid_pids)
            valid_pids = valid_pids - excluded_set
            print(f"Excluded {original_count - len(valid_pids)} products from {len(excluded_product_ids)} excluded IDs for category {category}")
        
    
    async def shortlist_products_async(
        self,
        mapped_attributes_and_values: Dict,
        product_attributes_for_search: Dict,
        category: str,
        n: int = 10,
        excluded_product_ids: list = None,
    ) -> Tuple:
        """Async wrapper for shortlist_products."""
        result = self.shortlist_products(
            mapped_attributes_and_values,
            product_attributes_for_search,
            category,
            n,
            excluded_product_ids
        )
        return category, result
    
    async def run_all_shortlists(
        self,
        mapped_attributes_and_values: Dict,
        product_attributes_for_search: Dict,
        n: int = 10,
        excluded_product_ids: list = None,
    ) -> Dict:
        """Run shortlisting for all product categories."""
        categories = product_attributes_for_search.keys()
        
        tasks = [
            asyncio.create_task(
                self.shortlist_products_async(
                    mapped_attributes_and_values,
                    product_attributes_for_search,
                    category,
                    n,
                    excluded_product_ids
                )
            )
            for category in categories
        ]
        
        results = await asyncio.gather(*tasks)
        return {category: shortlist for category, shortlist in results}
    
    def merge_and_sort_shortlists(self, shortlist_results: Dict) -> OrderedDict:
        """Merge and sort shortlists from multiple categories."""
        combined_scores = {}
        
        for category_scores in shortlist_results.values():
            for pid, score in category_scores.items():
                combined_scores[pid] = score
        
        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return OrderedDict(sorted_scores)
    
    async def redis_shortlist_products(
        self,
        mapped_attributes_and_values: Dict,
        product_attributes_for_search: Dict,
        category: str,
        n: int = 10,
        excluded_product_ids: List[str] = None,
        genders = ["female"],
    ) -> Dict[str, float]:
        """
        Redis-optimized product shortlisting with weighted scoring.
        Exact functional replacement for the original shortlist_products method.

        Returns:
            Dict mapping internal_id to weighted score (same format as original)
        """
        
        search_attrs = product_attributes_for_search.get(category, {}).get("attributes", [])
        if not search_attrs:
            return {}

        # Generate unique session ID for temporary keys
        session_id = uuid.uuid4().hex[:8]
        temp_keys = []

        try:
            # Handle both list and string input for backward compatibility
            if isinstance(genders, str):
                genders = [genders]
            
            # Create list of all genders to search (including unisex)
            genders_to_search = list(genders) if genders else ["female"]
            if "unisex" not in genders_to_search and not (len(genders_to_search) == 1 and genders_to_search[0] == "unisex"):
                genders_to_search.append("unisex")
                
            print(f"🚀 Redis shortlisting for category '{category}' with {len(search_attrs)} attributes (genders: {genders_to_search})")

            # Process each search attribute
            for attr_idx, search_attr in enumerate(search_attrs):
                desc = search_attr["attribute_description"]
                possible_vals = search_attr["possible_values"]
                weight = search_attr["weight"]

                print(f"  Processing attribute '{desc}' (weight: {weight}) with values: {possible_vals}")

                mapping_info = mapped_attributes_and_values.get(category, {}).get(desc)
                if not mapping_info:
                    print(f"    No mapping info found for '{desc}', skipping")
                    continue

                # Collect all Redis keys for this attribute
                redis_keys_for_attribute = []
                value_mapping = mapping_info["value_mapping"]

                for query_val in possible_vals:
                    mapped_dict = value_mapping.get(query_val, {})
                    for mapped_attr, details in mapped_dict.items():
                        for canonical_value in details.get("values", []):
                            print("canonical value - ", canonical_value)
                            # Search all specified genders
                            for search_gender in genders_to_search:
                                redis_key = f"attr_score:{search_gender}:{category}:{mapped_attr}:{canonical_value.lower().strip()}"
                                print("redis key - ", redis_key)
                                # Check if key exists and has products
                                key_size = await self.redis_client.zcard(redis_key)
                                if key_size > 0:
                                    redis_keys_for_attribute.append(redis_key)
                                    print(f"    Found {key_size} products for {redis_key}")

                # Union this attribute's keys with weight applied
                if redis_keys_for_attribute:
                    attr_temp_key = f"temp_attr_{attr_idx}_{session_id}"
                    temp_keys.append(attr_temp_key)

                    # Apply weight to all scores for this attribute
                    await self.redis_client.zunionstore(
                        attr_temp_key,
                        {key: weight for key in redis_keys_for_attribute}
                    )

                    # Set expiration for cleanup
                    await self.redis_client.expire(attr_temp_key, 300)  # 5 minutes

                    attr_size = await self.redis_client.zcard(attr_temp_key)
                    print(f"    Created weighted attribute set '{attr_temp_key}' with {attr_size} products")
                else:
                    print(f"    No Redis keys found for attribute '{desc}'")

            if not temp_keys:
                print("  No matching attributes found, returning empty results")
                return {}

            # Final union: combine all weighted attributes
            final_key = f"final_scores_{session_id}"
            await self.redis_client.zunionstore(
                final_key,
                {key: 1.0 for key in temp_keys}
            )

            final_size = await self.redis_client.zcard(final_key)
            print(f"  Combined all attributes into '{final_key}' with {final_size} scored products")

            # Handle exclusions if provided
            if excluded_product_ids:
                for internal_id in excluded_product_ids:
                    await self.redis_client.zrem(final_key, internal_id)
                print(f"  Excluded {len(excluded_product_ids)} products from results")

            # Get top-N results with scores
            results = await self.redis_client.zrevrange(
                final_key, 0, n-1, withscores=True
            )

            result_dict = dict(results)
            print(f"  Returning top {len(result_dict)} products with scores: {list(result_dict.keys())[:3]}...")

            return result_dict

        except Exception as e:
            print(f"❌ Redis shortlisting failed: {str(e)}")
            # Fallback to original method
            print("  Falling back to original shortlisting method...")
            return self.shortlist_products(
                mapped_attributes_and_values,
                product_attributes_for_search,
                category,
                n,
                excluded_product_ids
            )

        finally:
            # Cleanup: Delete all temporary keys
            cleanup_keys = temp_keys[:]
            if 'final_key' in locals():
                cleanup_keys.append(final_key)

            if cleanup_keys:
                try:
                    await self.redis_client.delete(*cleanup_keys)
                    print(f"  Cleaned up {len(cleanup_keys)} temporary Redis keys")
                except Exception as cleanup_error:
                    print(f"  Warning: Failed to cleanup keys: {cleanup_error}")

    async def redis_run_all_shortlists(
        self,
        mapped_attributes_and_values: Dict,
        product_attributes_for_search: Dict,
        n: int = 10,
        excluded_product_ids: List[str] = None,
        genders = ["female"],
    ) -> Dict[str, Dict[str, float]]:
        """
        Redis-optimized version of run_all_shortlists.
        Exact functional replacement for the original run_all_shortlists method.

        Returns:
            Dict mapping category to {internal_id: score} (same format as original)
        """
        categories = product_attributes_for_search.keys()
        print(f"🚀 Redis shortlisting for {len(categories)} categories: {list(categories)} (genders: {genders})")

        # Run all categories in parallel
        tasks = []
        for category in categories:
            task = asyncio.create_task(
                self.redis_shortlist_products(
                    mapped_attributes_and_values,
                    product_attributes_for_search,
                    category,
                    n,
                    excluded_product_ids,
                    genders
                )
            )
            tasks.append((category, task))

        # Gather results
        results = {}
        for category, task in tasks:
            try:
                category_results = await task
                if category_results:
                    results[category] = category_results
                    print(f"✅ Category '{category}': {len(category_results)} products")
                else:
                    print(f"⚠️  Category '{category}': No products found")
            except Exception as e:
                print(f"❌ Category '{category}' failed: {str(e)}")
                results[category] = {}

        total_products = sum(len(cat_results) for cat_results in results.values())
        print(f"🎯 Redis shortlisting completed: {total_products} total products across {len(results)} categories")

        return results

    async def get_products_from_redis(
        self,
        internal_ids: List[str],
        index_name: str = "idx:product_metadata"
    ) -> List[Dict]:
        """
        Retrieve product details from Redis using FT.SEARCH.
        
        Args:
            internal_ids: List of internal product IDs
            index_name: Redis search index name
            
        Returns:
            List of product dictionaries with metadata
        """
        if not internal_ids:
            return []
            
        try:
            # Create OR query for all internal IDs
            query_string = " | ".join([f"@internal_id:{id_}" for id_ in internal_ids])
            query = (
                Query(query_string)
                .return_fields("internal_id", "title", "image_url", "product_url", "max_price", "store_name", "body_text")
            )
            
            # Execute search
            search_results = await self.redis_client.ft(index_name).search(query)
            
            # Convert results to list of dictionaries
            products = []
            for doc in search_results.docs:
                product_dict = {key: value for key, value in doc.__dict__.items()}
                products.append(product_dict)
                
            print(f"Retrieved {len(products)} products from Redis index '{index_name}'")
            return products
            
        except Exception as e:
            print(f"Error retrieving products from Redis: {str(e)}")
            return []

    async def rerank_shortlisted_products(
        self,
        user_message: str,
        sorted_shortlist_results: OrderedDict
    ) -> Dict:
        """Rerank shortlisted products using semantic similarity."""
        product_ids = list(sorted_shortlist_results.keys())
        
        if not product_ids:
            return {}
        
        # Get product descriptions from Redis instead of data_loader
        try:
            products = await self.get_products_from_redis(product_ids)
            
            documents = []
            description_to_id = {}
            
            for product in products:
                pid = product.get("internal_id")
                description = product.get("body_text") or product.get("title", "")
                if description and pid:
                    documents.append(description)
                    description_to_id[description] = pid
            
            if not documents:
                print(f"No descriptions found for {len(product_ids)} products")
                return {}
            
            print(f"Reranking {len(documents)} products with descriptions")
            
            # Rerank using Voyage AI
            reranking = await voyageai_client.rerank(
                user_message,
                documents,
                model="rerank-2-lite",
                top_k=min(len(documents), 10)
            )
            
            return {
                description_to_id[result.document]: result.relevance_score
                for result in reranking.results
            }
            
        except Exception as e:
            print(f"Error in rerank_shortlisted_products: {e}")
            return {}

# Global product matcher instance
product_matcher = ProductMatcher()