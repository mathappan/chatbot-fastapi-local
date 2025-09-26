"""
Unified Product Category Extraction

This module provides a centralized product category extraction function used by both:
- RAG-based search flow 
- Filter-based search flow

Uses the RAG-based methodology for consistent category extraction across all search types.
"""

import json
from typing import Dict, List
from clients import groq_client
from prompts import UNIFIED_PRODUCT_CATEGORY_EXTRACTION_PROMPT

from global_metadata import get_global_metadata


class UnifiedProductCategoryExtractor:
    """Unified product category extraction for both RAG and filter-based searches"""
    
    def __init__(self):
        pass
    
    async def get_product_category(self, user_query: str, genders: List[str]) -> Dict:
        """
        Extract product categories from user query using RAG methodology with validation
        
        Args:
            user_query: Natural language user query
            genders: List of user's gender preferences
            
        Returns:
            Dict with extracted garment_type list
        """
        max_retries = 3
        
        # Get product categories from global metadata
        group_descriptions, _, _ = get_global_metadata()
        valid_categories = set(group_descriptions.keys()) if group_descriptions else set()
        
        # Store conversation history for retry attempts
        messages = [
            {
                "role": "system",
                "content": UNIFIED_PRODUCT_CATEGORY_EXTRACTION_PROMPT
            },
            {
                "role": "user",
                "content": f"""User query: {user_query}
Gender preferences: {genders}

Extract the `garment_type` mentioned or implied in the query. Consider the gender preferences when determining appropriate product categories. Choose only from this list: {list(valid_categories)}  
Return as JSON: {{ "garment_type": [...] }}
Use exact spelling and casing from the list.  
If the type isn't explicit, infer the most likely options.  
Do not include anything not in the list."""
            }
        ]
        
        for attempt in range(max_retries):
            try:
                # Add assistant role for continuation
                current_messages = messages + [{"role": "assistant", "content": "```json"}]
                
                completion = await groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=current_messages,
                    stop="```",
                )
                response_content = completion.choices[0].message.content
                user_query_json_garment = json.loads(response_content)
                garment_types = user_query_json_garment.get('garment_type', [])
                print("garment_types - ", garment_types)
                
                # Validate that all garment types are in the valid categories
                invalid_types = [gt for gt in garment_types if gt not in valid_categories]
                
                if not invalid_types:
                    # All types are valid
                    print("user_query_json_garment-", user_query_json_garment)
                    return user_query_json_garment
                else:
                    # Some types are invalid - add feedback for retry
                    print(f"Attempt {attempt + 1}: Invalid garment types found: {invalid_types}")
                    
                    if attempt < max_retries - 1:
                        # Add the assistant's response to conversation history
                        messages.append({
                            "role": "assistant", 
                            "content": f"```json\n{response_content}\n```"
                        })
                        
                        # Add feedback message for retry
                        feedback_message = f"""Your previous response contained invalid product types: {invalid_types}
These types are not in our valid product catalog.

Valid product types are: {list(valid_categories)}

Please correct your response and only use the exact product types from the valid list. 
Retry extracting garment types from the original query: "{user_query}"

Return corrected JSON: {{ "garment_type": [...] }}"""
                        
                        messages.append({
                            "role": "user",
                            "content": feedback_message
                        })
                    else:
                        # Last attempt failed, filter to valid categories
                        valid_garment_types = [gt for gt in garment_types if gt in valid_categories]
                        if valid_garment_types:
                            print(f"Final attempt: Filtered to valid categories: {valid_garment_types}")
                            return {"garment_type": valid_garment_types}
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for category extraction: {str(e)}")
                if attempt < max_retries - 1:
                    # Add error feedback for retry
                    if len(messages) > 2:  # If there was a previous assistant response
                        messages.append({
                            "role": "user",
                            "content": f"""There was an error parsing your previous response: {str(e)}
Please provide a valid JSON response with the correct format:
{{ "garment_type": [...] }}

Only use product types from this valid list: {list(valid_categories)}"""
                        })
                    continue
        
        # If all attempts failed, raise error
        raise ValueError(f"No valid product categories found for query: {user_query} after {max_retries} attempts")

# Global unified product category extractor instance
unified_product_category_extractor = UnifiedProductCategoryExtractor()