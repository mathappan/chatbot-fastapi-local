"""
Unified Product Category Extraction

This module provides a centralized product category extraction function used by both:
- RAG-based search flow 
- Filter-based search flow

Uses the RAG-based methodology for consistent category extraction across all search types.
"""

import json
from typing import Dict
from clients import groq_client
from data_loader import data_loader

# Prompt for extracting product categories from user queries (from RAG methodology)
high_level_user_query_extraction_prompt = '''
You are an expert apparel salesperson. Your task is to extract only one things from a user's natural language query:

1. `garment_type`: All possible types of clothing the user is referring to. 
This can be a single item or multiple. You will only choose from the product types mentioned in the user message.
Return as a list, even if only one is present.

Point to Note - The user might not specify exact type. Being a salesperson, you have to give the type of apparel that might fit the user query

Only return the extracted result in this JSON format:

{
  "garment_type": [...]
}

'''

class UnifiedProductCategoryExtractor:
    """Unified product category extraction for both RAG and filter-based searches"""
    
    def __init__(self):
        self.data = data_loader
    
    async def get_product_category(self, user_query: str, gender: str) -> Dict:
        """
        Extract product categories from user query using RAG methodology with validation
        
        Args:
            user_query: Natural language user query
            gender: User's gender preference
            
        Returns:
            Dict with extracted garment_type list
        """
        max_retries = 3
        valid_categories = set(self.data.product_categories)
        last_response = None
        
        for attempt in range(max_retries):
            try:
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
                                        {
                                        "type": "text",
                                        "text": f"""User query: {user_query}

                                    Extract the `garment_type` mentioned or implied in the query. Choose only from this list: {self.data.product_categories}  
                                    Return as JSON: {{ "garment_type": [...] }}
                                    Use exact spelling and casing from the list.  
                                    If the type isn’t explicit, infer the most likely options.  
                                    Do not include anything not in the list."""
                                        }
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
                last_response = user_query_json_garment
                garment_types = user_query_json_garment.get('garment_type', [])
                
                # Validate that all garment types are in the valid categories
                if all(garment_type in valid_categories for garment_type in garment_types):
                    print("user_query_json_garment-", user_query_json_garment)
                    return user_query_json_garment
                else:
                    invalid_types = [gt for gt in garment_types if gt not in valid_categories]
                    print(f"Attempt {attempt + 1}: Invalid garment types found: {invalid_types}")
                    if attempt < max_retries - 1:
                        continue
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for category extraction: {str(e)}")
                if attempt < max_retries - 1:
                    continue
        
        # After all retries failed, filter to only valid categories
        if last_response:
            garment_types = last_response.get('garment_type', [])
            valid_garment_types = [gt for gt in garment_types if gt in valid_categories]
            
            if valid_garment_types:
                print(f"Filtered to valid categories: {valid_garment_types}")
                return {"garment_type": valid_garment_types}
        
        # If no valid categories found, raise error
        raise ValueError(f"No valid product categories found for query: {user_query}")

# Global unified product category extractor instance
unified_product_category_extractor = UnifiedProductCategoryExtractor()