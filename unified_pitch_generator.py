"""
Unified Product Pitch Generator

This module provides a centralized pitch generation system used by both:
- RAG-based search flow 
- Filter-based search flow

Uses the exact same prompt and methodology from rag_search_flow.py
"""

import json
import asyncio
from typing import Dict, List
from clients import groq_client
from data_loader import data_loader

class UnifiedPitchGenerator:
    """Unified pitch generator for both RAG and filter-based searches"""
    
    def __init__(self):
        self.data = data_loader
    
    async def fetch_product_pitch(self, customer_intent: str, product: Dict) -> Dict:
        """
        Generate a product pitch for a single product using RAG methodology
        
        Args:
            customer_intent: Customer's search intent/query
            product: Product dictionary with id, title, description
            
        Returns:
            Dict with product_pitch, match flag, and reasoning
        """
        # Use the exact same prompt format as rag_search_flow.py
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
            # Use the exact same Groq call format as rag_search_flow.py
            response = await groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
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
    
    async def generate_pitches_for_products(self, customer_intent: str, products: List[Dict]) -> Dict:
        """
        Generate pitches for multiple products in parallel
        
        Args:
            customer_intent: Customer's search intent/query
            products: List of product dictionaries with id, title, description
            
        Returns:
            Dict mapping product_id to pitch results
        """
        # Launch all pitch generation tasks in parallel
        tasks = {
            product["id"]: asyncio.create_task(
                self.fetch_product_pitch(customer_intent, product)
            )
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
    
    async def generate_pitches_from_search_results(self, customer_intent: str, search_results: List[Dict]) -> Dict:
        """
        Generate pitches from RAG search results format
        
        Args:
            customer_intent: Customer's search intent/query
            search_results: List of dicts with product_id, relevance_score, text_description
            
        Returns:
            Dict mapping product_id to pitch results
        """
        # Convert search results to product format (same as rag_search_flow.py)
        products = [
            {
                "id": result["product_id"],
                "title": f"Product {result['product_id']}",  # You can enhance this with actual titles
                "description": result["text_description"]
            }
            for result in search_results
        ]
        
        return await self.generate_pitches_for_products(customer_intent, products)
    
    async def generate_pitches_from_product_ids(self, customer_intent: str, product_ids: List[str]) -> Dict:
        """
        Generate pitches from product IDs (filter search format)
        
        Args:
            customer_intent: Customer's search intent/query
            product_ids: List of product IDs
            
        Returns:
            Dict mapping product_id to pitch results
        """
        # Convert product IDs to product format using data_loader
        products = []
        for product_id in product_ids:
            products.append({
                "id": product_id,
                "title": self.data.product_titles.get(product_id, f"Product {product_id}"),
                "description": self.data.text_descriptions.get(product_id, "No description available.")
            })
        
        return await self.generate_pitches_for_products(customer_intent, products)

# Global unified pitch generator instance
unified_pitch_generator = UnifiedPitchGenerator()