import os
import json
from typing import Dict, Optional
from groq import AsyncGroq
from fastapi import HTTPException

class IntentClassifier:
    """Classifies user intent for image and text inputs."""
    
    def __init__(self):
        self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    
    async def classify_user_intent(self, user_text: str, has_image: bool = False) -> Dict[str, str]:
        """
        Classify user intent into COMPLEMENT, SEARCH, or AMBIGUOUS.
        
        Args:
            user_text: The user's text input
            has_image: Whether the user uploaded an image
            
        Returns:
            Dict with 'intent' and 'reason'
        """
        system_prompt = f'''
        You are an expert fashion assistant that classifies user intent.

        User Input: "{user_text}"
        Has Image: {has_image}

        Classify the user's intent into exactly ONE of these categories:

        COMPLEMENT - User wants items that go WITH their outfit/item. Keywords: "what goes with", "complement", "match with", "pair with", "style with", "wear with"

        SEARCH - User wants to FIND similar or same items. Keywords: "find", "looking for", "where to buy", "similar to", "like this", "search for", "want this"

        AMBIGUOUS - Cannot determine intent clearly, or the request is too vague

        Rules:
        - If user has an image and NO text, classify as COMPLEMENT
        - Be decisive - choose the most likely intent based on language patterns
        - Only use AMBIGUOUS for genuinely unclear requests

        Output format (JSON):
        {{
          "intent": "COMPLEMENT|SEARCH|AMBIGUOUS",
          "reason": "Brief explanation of classification"
        }}
        '''
        
        try:
            response = await self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": system_prompt
                    }
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=150
            )
            
            result = json.loads(response.choices[0].message.content)
            print(f"Intent classification: {result.get('intent', 'UNKNOWN')}")
            return result
            
        except Exception as e:
            print(f"Error classifying user intent: {e}")
            # Default to COMPLEMENT if classification fails
            return {"intent": "COMPLEMENT", "reason": f"Error in classification: {str(e)}"}
    
    async def handle_ambiguous_intent(self, user_text: str, conversation_context: str = "") -> Dict[str, str]:
        """
        Handle ambiguous intent by generating a general fashion response.
        
        Args:
            user_text: The user's current message
            conversation_context: Previous conversation context
            
        Returns:
            Dict with 'content' for the response
        """
        system_prompt = f'''
        You are an expert fashion assistant helping a user with their fashion needs.
        
        Current user message: "{user_text}"
        
        Previous conversation context:
        {conversation_context or "No previous context available."}
        
        Based on the conversation history and current message, provide a helpful general fashion response that addresses their query. Use the context to make your response more personalized and relevant.
        
        Output format (JSON):
        {{
          "content": "A helpful general fashion response addressing their query based on context"
        }}
        '''
        
        try:
            response = await self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user", 
                        "content": system_prompt
                    }
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
                max_tokens=400
            )
            
            result = json.loads(response.choices[0].message.content)
            print("Generated general response for ambiguous intent")
            return result
            
        except Exception as e:
            print(f"Error handling ambiguous intent: {e}")
            return {
                "content": "I'd be happy to help with your fashion needs! Could you provide a bit more detail about what you're looking for?"
            }

# Initialize the intent classifier
intent_classifier = IntentClassifier()