import os
import json
import base64
from typing import Dict, List, Any, Optional
from groq import AsyncGroq
from fastapi import HTTPException, UploadFile

class ImageProcessor:
    """Handles image processing and analysis for fashion recommendations."""
    
    def __init__(self):
        self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.supported_formats = {"jpg", "jpeg", "png", "webp", "gif"}
        self.max_size_mb = 10
    
    async def validate_image_file(self, file: UploadFile) -> bytes:
        """Validates and returns image file content."""
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in self.supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported image format. Supported formats: {', '.join(self.supported_formats)}"
            )
        
        # Read and check file size
        content = await file.read()
        max_size_bytes = self.max_size_mb * 1024 * 1024
        if len(content) > max_size_bytes:
            raise HTTPException(
                status_code=400, 
                detail=f"File size ({len(content) / (1024*1024):.2f}MB) exceeds maximum allowed size ({self.max_size_mb}MB)"
            )
        
        print(f"Image validated: {file.filename}, size: {len(content) / (1024*1024):.2f}MB")
        return content
    
    def encode_image_to_base64(self, image_content: bytes) -> str:
        """Converts image bytes to base64 string."""
        return base64.b64encode(image_content).decode('utf-8')
    
    async def analyze_fashion_image(self, image_content: bytes, user_text: Optional[str] = None) -> str:
        """
        Analyzes fashion image to extract style attributes and generate recommendations.
        
        Returns: JSON string with recommendations and analysis
        """
        base64_image = self.encode_image_to_base64(image_content)
        
        # Create comprehensive analysis prompt
        analysis_prompt = self._create_fashion_analysis_prompt(user_text)
        
        try:
            response = await self.groq_client.chat.completions.create(
                model="llava-v1.5-7b-4096-preview",  # Vision model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            print("Image analysis completed successfully")
            return result
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            raise HTTPException(status_code=500, detail=f"Error analyzing image: {e}")
    
    def _create_fashion_analysis_prompt(self, user_text: Optional[str] = None) -> str:
        """Creates a comprehensive prompt for fashion image analysis."""
        
        base_prompt = """You are an expert fashion stylist analyzing a clothing item or outfit. 

Analyze this image and provide recommendations for complementary items that would go well with what you see.

Your analysis should include:

1. ITEM IDENTIFICATION: What clothing items do you see? (kurta, dress, saree, lehenga, etc.)

2. STYLE ANALYSIS:
   - Colors (primary and accent colors)
   - Patterns and textures
   - Silhouette and cut
   - Occasion suitability (casual, formal, festive, party)
   - Cultural style (traditional, modern, fusion)

3. COMPLEMENTARY RECOMMENDATIONS:
   - 3-5 specific items that would complement this outfit
   - Consider color coordination, style matching, and occasion appropriateness
   - Focus on items that would enhance the overall look

4. STYLING SUGGESTIONS:
   - How to style these complementary pieces
   - Occasion-appropriate combinations

Provide your response in this JSON format:
{
    "identified_items": ["item1", "item2"],
    "style_analysis": {
        "colors": ["primary_color", "accent_color"],
        "patterns": ["pattern1", "pattern2"],
        "silhouette": "description",
        "occasion": "occasion_type",
        "cultural_style": "style_type"
    },
    "complementary_items": [
        "complementary_item_1",
        "complementary_item_2", 
        "complementary_item_3"
    ],
    "styling_notes": "Brief styling suggestions for the recommended pieces"
}"""

        if user_text:
            base_prompt += f"\n\nADDITIONAL USER CONTEXT: {user_text}\nConsider this context when making your recommendations."
        
        return base_prompt
    
    async def generate_search_query_for_intent(self, user_text: str, image_content: bytes = None, intent: str = "SEARCH") -> str:
        """
        Generate search query based on user text, optional image, and intent.
        
        Args:
            user_text: User's text input
            image_content: Optional image bytes
            intent: SEARCH or COMPLEMENT
            
        Returns:
            Search query string for SEARCH intent or JSON string with multiple queries for COMPLEMENT
        """
        try:
            # Build conversation messages based on whether image is provided
            messages = self._build_search_conversation(user_text, image_content, intent)
            
            # Use vision model if image provided, otherwise use text model
            model = "llava-v1.5-7b-4096-preview" if image_content else "llama-3.3-70b-versatile"
            
            response = await self.groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"} if intent == "COMPLEMENT" else None
            )
            
            result = response.choices[0].message.content
            print(f"Generated {intent} query: {result[:100]}")
            return result
            
        except Exception as e:
            print(f"Error generating {intent} query: {e}")
            if intent == "SEARCH":
                return user_text  # Fallback to user text
            else:
                return '{"complementary_searches": ["complementary fashion items"]}'
    
    def _build_search_conversation(self, user_text: str, image_content: bytes = None, intent: str = "SEARCH") -> list:
        """Build conversation for Groq API with optional image and user text."""
        
        if intent == "SEARCH":
            if image_content:
                # Image + text for search
                base64_image = self.encode_image_to_base64(image_content)
                prompt = f"""You are a fashion search assistant. The user wants to SEARCH for specific items.

User text: {user_text}

Analyze the image and user text to generate ONE search query that captures what the user is looking for.

Return only a single search query string, no JSON, no extra text.

Example: "black formal kurta for women"""
                
                return [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ]
            else:
                # Text-only search
                prompt = f"""You are a fashion search assistant. Generate ONE search query based on the user's request.

User request: {user_text}

Return only a single search query string that captures what the user is looking for.

Example: "black formal kurta for women"""
                
                return [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
        
        else:  # COMPLEMENT
            if image_content:
                # Image + text for complement
                base64_image = self.encode_image_to_base64(image_content)
                prompt = f"""You are a fashion stylist. The user wants COMPLEMENTARY items that go well with what's in the image.

User text: {user_text}

Analyze the image and user text to suggest 3-5 complementary items that would go well with the outfit/item shown.

Return a JSON object with multiple search queries:
{{
    "complementary_searches": [
        "search query 1",
        "search query 2", 
        "search query 3"
    ]
}}"""
                
                return [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ]
            else:
                # Text-only complement
                prompt = f"""You are a fashion stylist. Based on the user's description, suggest complementary items.

User description: {user_text}

Generate 3-5 complementary items that would go well with what the user described.

Return a JSON object with multiple search queries:
{{
    "complementary_searches": [
        "search query 1",
        "search query 2", 
        "search query 3"
    ]
}}"""
                
                return [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

# Initialize the image processor
image_processor = ImageProcessor()