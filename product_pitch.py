import json
import asyncio
from typing import Dict, List

from clients import groq_client
from data_loader import data_loader

class ProductPitchGenerator:
    """Generates product pitches and recommendations."""
    
    def __init__(self):
        self.data = data_loader
        self.product_pitch_system_prompt = '''
You are an expert AI copywriter for a fashion e-commerce platform.

Your job is to generate a short, catchy, benefit-focused **one-line pitch** for a given product based on a **user's search query**.

You will be provided with:
- The user's search query (e.g., "need blouse for party").
- A product, which includes:
    - `title`: Name of the product.
    - `description`: Description of the product.

Your task:
1. Understand the user's intent based on the search query.
2. Write a **one-line product pitch** that highlights the most appealing, relevant benefits of the product in relation to the query.
3. Determine if this product is an **exact match** to the user's query. Set the flag to `true` if it directly fits the query intent and context; otherwise, `false`.

Keep the pitch:
- Short, friendly, and easy to scan.
- Conversational and persuasive and creative.
- Not misleading (don't make claims that the product doesn't support).
- Use variety in your language and make it a creative pitch.

Output format (JSON):
{
  "product_pitch": "Perfect party blouse with statement sleeves.",
  "match": true/false
}
'''
    
    async def fetch_product_pitch(self, search_query: str, product: Dict) -> Dict:
        """Generate a product pitch for a single product."""
        try:
            response = await groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": self.product_pitch_system_prompt},
                    {
                        "role": "user",
                        "content": f"Search query - {search_query}\nProduct - {json.dumps(product)}. Write short one liner pitches. Be creative in your language."
                    },
                ],
                temperature=1,
                response_format={"type": "json_object"},
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "error": str(e),
                "product_pitch": "Great product for your needs!",
                "exact_match": False
            }
    
    async def generate_top_ten_pitches(
        self,
        search_query: str,
        top_ten: Dict
    ) -> Dict:
        """Generate pitches for top 10 products using unified generator."""
        from unified_pitch_generator import unified_pitch_generator
        
        # Use unified pitch generator with product IDs
        product_ids = list(top_ten.keys())
        return await unified_pitch_generator.generate_pitches_from_product_ids(search_query, product_ids)
    
    def format_product_recommendations(
        self,
        pitches: Dict,
        top_products: Dict,
        max_products: int = 5
    ) -> str:
        """Format product recommendations into a user-friendly response."""
        if not pitches:
            return "I'm sorry, I couldn't find any products that match your search. Would you like to try different criteria?"
        
        response = "I found some great options for you:\n\n"
        
        count = 0
        for product_id, score in list(top_products.items())[:max_products]:
            if product_id in pitches:
                pitch_data = pitches[product_id]
                title = data_loader.product_titles.get(product_id, f"Product {product_id}")
                pitch = pitch_data.get("product_pitch", "Great product for your needs!")
                
                count += 1
                response += f"{count}. **{title}**\n   {pitch}\n\n"
        
        if count == 0:
            return "I'm sorry, I couldn't find any products that match your search. Would you like to try different criteria?"
        
        response += "Would you like more details about any of these items, or shall I search for something else?"
        return response

# Global product pitch generator instance
product_pitch_generator = ProductPitchGenerator()