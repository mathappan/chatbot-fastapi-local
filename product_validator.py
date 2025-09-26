"""
Product Validation Module

Centralized product validation logic to eliminate code duplication.
This module provides functions for validating product matches against customer intent using LLM.
"""

import json
import asyncio
from typing import List, Dict, Any
from logger_config import log_detailed_error


class ProductValidator:
    """Handles product validation against customer intent using LLM."""
    
    def __init__(self, groq_client):
        self.groq_client = groq_client
    
    async def validate_single_product_match(self, customer_intent: str, product: dict) -> Dict[str, Any]:
        """
        Validate if a single product matches the customer intent using LLM
        
        Args:
            customer_intent: Original customer search query
            product: Product dictionary with id, title, description
            
        Returns:
            Dict with match flag and reasoning
        """
        product_validation_prompt = f'''
        You are an expert AI product matcher for a fashion e-commerce platform.

        Your job is to determine if a product is a good match for a customer's search intent.

        Customer Intent: {customer_intent}
        Product Details: {json.dumps({
            "id": product.get("product_id"),
            "title": product.get("title"),
            "description": product.get("text_description")
        })}

        Your task:
        1. Analyze if this product directly matches the customer's intent
        2. Consider product type, style, features, and attributes mentioned in the intent
        3. Be strict - only return true if it's a genuine match
        4. Provide clear reasoning for your decision

        Output format (JSON):
        {{
          "match": true/false,
          "reason": "Brief explanation of why it matches or doesn't match"
        }}
        '''
        
        try:
            response = await self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": product_validation_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            log_detailed_error(
                e,
                context="validate_single_product_match",
                local_vars={
                    "customer_intent": customer_intent,
                    "product_id": product.get("product_id"),
                    "product_title": product.get("title"),
                    "product_description": product.get("text_description", "")[:200]
                }
            )
            return {"match": True, "reason": "Validation error - assuming match"}

    async def validate_product_matches(self, customer_intent: str, product_candidates: List[dict]) -> List[dict]:
        """
        Validate all product candidates against customer intent in parallel
        
        Args:
            customer_intent: Original customer search query
            product_candidates: List of product dictionaries to validate
            
        Returns:
            List of products that match the customer intent
        """
        if not product_candidates:
            return []
            
        print(f"🔍 Validating {len(product_candidates)} products against customer intent...")
        
        # Launch all validation tasks in parallel for efficiency
        validation_tasks = {
            product["product_id"]: asyncio.create_task(
                self.validate_single_product_match(customer_intent, product)
            )
            for product in product_candidates
        }
        
        # Create product lookup for easy access
        product_lookup = {p["product_id"]: p for p in product_candidates}
        
        # Gather validation results
        try:
            validation_results = await asyncio.gather(*validation_tasks.values(), return_exceptions=True)
            
            # Process results and filter matching products
            matching_products = []
            for i, (product_id, task) in enumerate(validation_tasks.items()):
                try:
                    result = validation_results[i]
                    if isinstance(result, Exception):
                        print(f"❌ Validation failed for product {product_id}: {result}")
                        # Include product with error - assume match for user experience
                        matching_products.append(product_lookup[product_id])
                    elif result.get("match", False):
                        product = product_lookup[product_id]
                        product["llm_filter_reason"] = result.get("reason", "Matches customer intent")
                        matching_products.append(product)
                        print(f"✅ Product {product_id} matches: {result.get('reason', 'No reason provided')}")
                    else:
                        print(f"❌ Product {product_id} doesn't match: {result.get('reason', 'No reason provided')}")
                
                except Exception as e:
                    print(f"❌ Error processing validation result for product {product_id}: {e}")
                    # Include product with error - assume match for user experience
                    matching_products.append(product_lookup[product_id])
            
            print(f"✅ Product validation complete: {len(matching_products)}/{len(product_candidates)} products match")
            return matching_products
            
        except Exception as e:
            log_detailed_error(
                e,
                context="validate_product_matches",
                local_vars={
                    "customer_intent": customer_intent,
                    "product_count": len(product_candidates)
                }
            )
            print(f"❌ Error in batch validation: {e}")
            # Return all products if validation fails
            return product_candidates


# Global instance to be used across modules
product_validator = None

def get_product_validator():
    """Get the global product validator instance"""
    global product_validator
    if product_validator is None:
        # Import here to avoid circular imports
        from clients import groq_client
        product_validator = ProductValidator(groq_client)
    return product_validator