import asyncio
from typing import Dict, Tuple
from collections import OrderedDict

from clients import voyageai_client
from data_loader import data_loader

class ProductMatcher:
    """Handles product shortlisting and ranking."""
    
    def __init__(self):
        self.data = data_loader
    
    def shortlist_products(
        self,
        mapped_attributes_and_values: Dict,
        product_attributes_for_search: Dict,
        category: str,
        n: int = 10,
        excluded_product_ids: list = None,
    ) -> Dict:
        """Shortlist products based on search criteria."""
        # Filter products by category
        valid_pids = {
            str(item["id"]) for item in self.data.cleaned_results_with_type
            if item.get("product_type") == category
        }
        
        # Remove excluded product IDs
        if excluded_product_ids:
            excluded_set = set(str(pid) for pid in excluded_product_ids)
            original_count = len(valid_pids)
            valid_pids = valid_pids - excluded_set
            print(f"Excluded {original_count - len(valid_pids)} products from {len(excluded_product_ids)} excluded IDs for category {category}")
        
        filtered_standardised_jsons = {
            pid: data for pid, data in self.data.standardised_jsons.items()
            if pid in valid_pids
        }
        
        print(f"Found {len(filtered_standardised_jsons)} products for category {category}")
        
        # Get search attributes for the category
        search_attrs = product_attributes_for_search.get(category, {}).get("attributes", [])
        
        # Initialize scores
        scores = {pid: 0 for pid in filtered_standardised_jsons}
        
        # Scoring loop
        for pid, pdata in filtered_standardised_jsons.items():
            for search_attr in search_attrs:
                desc = search_attr["attribute_description"]
                possible_vals = search_attr["possible_values"]
                weight = search_attr["weight"]
                
                mapping_info = mapped_attributes_and_values.get(category, {}).get(desc)
                if not mapping_info:
                    continue
                
                value_mapping = mapping_info["value_mapping"]
                
                matched = False
                for query_val in possible_vals:
                    mapped_dict = value_mapping.get(query_val, {})
                    
                    for mapped_attr, details in mapped_dict.items():
                        prod_vals = pdata.get(mapped_attr, [])
                        if isinstance(prod_vals, str):
                            prod_vals = [prod_vals]
                        
                        if prod_vals and any(v in prod_vals for v in details.get("values", [])):
                            scores[pid] += weight
                            matched = True
                            break
                    if matched:
                        break
        
        # Return top-n
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return dict(ranked[:n])
    
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
    
    async def rerank_shortlisted_products(
        self,
        user_message: str,
        sorted_shortlist_results: OrderedDict
    ) -> Dict:
        """Rerank shortlisted products using semantic similarity."""
        product_ids = list(sorted_shortlist_results.keys())
        
        documents = []
        description_to_id = {}
        
        for pid in product_ids:
            description = self.data.text_descriptions.get(pid)
            if description:
                documents.append(description)
                description_to_id[description] = pid
        
        if not documents:
            return {}
        
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

# Global product matcher instance
product_matcher = ProductMatcher()