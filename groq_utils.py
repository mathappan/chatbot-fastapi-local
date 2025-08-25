import json
import os
from groq import AsyncGroq
from prompts import (
    TOP_LEVEL_ROUTER_PROMPT,
    GENERAL_AGENT_PROMPT,
    SALES_ROUTER_PROMPT,
    SEARCH_QUERY_CREATION_PROMPT,
    FOLLOW_UP_MESSAGE_PROMPT
)
from master_search import master_search
from rag_search_and_pitch import rag_search_and_pitch
from rag_search_flow import search_and_pitch
from conversation_storage import conversation_storage
from logger_config import logger
from data_loader import data_loader

# Initialize Groq client
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

async def call_groq_json(message_for_groq: str, system_prompt: str, model: str = "llama-3.1-8b-instant"):
    """Generic function to call the Groq API and get a JSON response."""
    if not groq_client:
        raise ConnectionError("Groq client not initialized.")
        
    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_for_groq}
            ],
            model=model,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        response_content = chat_completion.choices[0].message.content
        return json.loads(response_content)
    except Exception as e:
        print(f"An error occurred while calling the Groq API: {e}")
        return {"error": str(e)}

async def call_groq_text(message_for_groq: str, system_prompt: str, model: str = "llama-3.1-8b-instant"):
    """Generic function to call the Groq API and get a text response."""
    if not groq_client:
        raise ConnectionError("Groq client not initialized.")
        
    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_for_groq}
            ],
            model=model,
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred while calling the Groq API: {e}")
        return "Sorry, I'm having a little trouble right now. Please try again in a moment."

async def top_level_router_agent(intent_analysis: str) -> str:
    """Classifies user intent based on the derived intent analysis."""
    print("🤖 Top-Level Router Agent analyzing...")
    response_json = await call_groq_json(intent_analysis, TOP_LEVEL_ROUTER_PROMPT)
    decision = response_json.get("decision", "general_query")  # Default to general
    print(f"    ↳ Decision: {decision}")
    return decision

async def general_agent(intent_analysis: str) -> str:
    """Handles general, non-sales questions."""
    print("🤖 General Agent generating answer...")
    
    response = await call_groq_text(intent_analysis, GENERAL_AGENT_PROMPT)
    print(f"    ↳ Response: {response}")
    return response

async def sales_router_agent(intent_analysis: str) -> dict:
    """Decides whether to ask questions or search for a product."""
    print("🤖 Sales Router Agent analyzing...")
    
    response_json = await call_groq_json(intent_analysis, SALES_ROUTER_PROMPT)
    print(f"    ↳ Decision: {response_json}")
    return response_json

async def generate_search_query(intent_analysis: str, gender: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Generate a concise Google-style search query from chat context.
    
    Args:
        intent_analysis (str): The derived intent analysis from the conversation.
        chat_history (list): The complete chat history.
        gender (str): Gender of the user (e.g., 'female', 'male', 'unisex').

    Returns:
        str: A concise search query.
    """
    # Convert chat history to text format
 
    
    context_message = f'''Intent Analysis: {intent_analysis}\nGender: {gender}. 
    Only use the intent analysis to generate the search query. Avoid adding or removing information'''
    
    return await call_groq_text(context_message, SEARCH_QUERY_CREATION_PROMPT, model)

async def generate_sizes_for_product_types(user_preferred_size: str = None, product_types: list = None, model: str = "llama-3.1-8b-instant") -> dict:
    """
    Generate recommended sizes based on user preferences and available sizes for product types.
    
    Args:
        user_preferred_size (str): User's preferred size from user preferences
        product_types (list): List of product types the user is searching for
        model (str): Groq model to use
        
    Returns:
        dict: JSON object with "sizes" key containing list of recommended sizes
    """
    print(f"🔍 Generating sizes for product types: {product_types}")
    print(f"👤 User preferred size: {user_preferred_size}")
    
    try:
        # Get available sizes for the product types
        available_sizes = set()
        for product_type in product_types:
            if product_type in data_loader.sizes_by_product_type:
                available_sizes.update(data_loader.sizes_by_product_type[product_type])
        
        available_sizes_list = sorted(list(available_sizes))
        print(f"📏 Available sizes for {product_types}: {available_sizes_list}")
        
        if not available_sizes_list:
            print("⚠️ No sizes found for product types, returning empty sizes object")
            return {"sizes": []}
        
        # Prepare integrated user message for Groq
        user_message = f"""
You are an expert fashion sizing consultant. Your task is to recommend the most appropriate sizes for a customer based on their preferences and the available sizes for specific product types.

User's preferred size: {user_preferred_size if user_preferred_size else "Not specified"}
Product types: {', '.join(product_types)}
Available sizes for these product types: {available_sizes_list}

Rules:
1. If the user has specified a size preference, prioritize that size and include similar/adjacent sizes
2. If no size preference is given, return a reasonable range of common sizes
3. Only return sizes that are actually available for the given product types
4. Consider size variations (e.g., if user wants "M", also include "m", "medium" if available)
5. Limit to maximum 3-5 most relevant sizes
6. Return sizes in the exact format they appear in the available sizes list

Return only a JSON object with "sizes" as the key containing an array of recommended sizes, no other text.

Example:
{{
  "sizes": ["s", "m", "l"]
}}
"""
        
        # Call Groq with integrated message (no separate system prompt)
        groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        response = await groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": user_message}
            ],
            model=model,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        response_json = json.loads(response.choices[0].message.content)
        
        # Extract sizes from the expected JSON format
        recommended_sizes = response_json.get("sizes", [])
        
        # Validate that recommended sizes are actually available
        valid_sizes = [size for size in recommended_sizes if size in available_sizes_list]
        
        print(f"✅ Generated sizes: {valid_sizes}")
        return {"sizes": valid_sizes}
        
    except Exception as e:
        print(f"❌ Error generating sizes: {e}")
        # Fallback: return common sizes if available
        common_sizes = ['s', 'm', 'l']
        fallback_sizes = []
        for product_type in product_types:
            if product_type in data_loader.sizes_by_product_type:
                available_for_type = data_loader.sizes_by_product_type[product_type]
                fallback_sizes.extend([size for size in common_sizes if size in available_for_type])
        
        return {"sizes": list(set(fallback_sizes))[:3]}

async def generate_follow_up_message(chat_history: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Generate a follow-up message after showing product recommendations.
    
    Args:
        chat_history (str): The complete chat conversation history
        model (str): Groq model to use
        
    Returns:
        str: A friendly, engaging follow-up message
    """
    print("🤖 Generating follow-up message...")
    
    try:
        # Format the prompt with chat history
        formatted_prompt = FOLLOW_UP_MESSAGE_PROMPT.format(chat_history=chat_history)
        
        # Call Groq to generate the follow-up message
        response = await call_groq_text("", formatted_prompt, model)
        
        print(f"✅ Generated follow-up message: {response[:100]}...")
        return response.strip()
        
    except Exception as e:
        print(f"❌ Error generating follow-up message: {e}")
        # Fallback message
        return "Hope you found something you love! Let me know if you'd like to see more options or have any questions. 😊"

# --- Main Chatbot Logic ---
from typing import AsyncIterator, Dict, Any

async def chatbot_flow(intent_analysis: str, conversation_id: str = None, conversation_redis_client=None, user_preferences: Dict[str, Any] = None) -> AsyncIterator[Dict[str, Any]]:
    """
    Yields zero-or-more dicts shaped like
        {"event": <str>, "data": <str>}
    in the order they should be streamed via SSE.

    Yields one intial search event as soon as an expensive product
    search is about to start, then (after the search completes) yields
    the normal conversational response.
    """
    intent_decision = await top_level_router_agent(intent_analysis)
    print("intent decision - ", intent_decision)

    # ────────────────────────────────────────────────────────────────────
    # GENERAL (non-sales) FLOW
    # ────────────────────────────────────────────────────────────────────
    if intent_decision == "general_query":
        reply = await general_agent(intent_analysis)
        yield {"event": "message", "data": reply}
        return

    # ────────────────────────────────────────────────────────────────────
    # SALES FLOW
    # ────────────────────────────────────────────────────────────────────
    if intent_decision == "sales_query":
        sales_decision = await sales_router_agent(intent_analysis)

        # ───── perform_product_search ─────
        if sales_decision.get("action") == "perform_product_search":
            # Use gender from user preferences, default to "female" for Libas
            gender = user_preferences.get("gender", "female") if user_preferences else "female"
            print(f"🔄 Using user preferences: {user_preferences}")
            print(f"🎯 Gender for search: {gender}")
            optimized_query = await generate_search_query(intent_analysis, gender)

            # Extract product categories once for both RAG and filter searches
            from unified_product_category import unified_product_category_extractor
            category_response = await unified_product_category_extractor.get_product_category(optimized_query, gender)
            allowed_product_types = category_response.get('garment_type', [])
            
            # Generate recommended sizes based on user preferences and product types
            user_preferred_sizes = user_preferences.get("size", []) if user_preferences else []
            
            # Always use AI to generate intelligent size recommendations
            # Pass user preferred sizes so LLM can make informed decisions based on available product sizes
            user_preferred_size_str = ", ".join(user_preferred_sizes) if user_preferred_sizes else None
            recommended_sizes_response = await generate_sizes_for_product_types(user_preferred_size_str, allowed_product_types)
            sizes = recommended_sizes_response.get("sizes", [])
            print(f"🎯 Using AI recommended sizes based on user preferences {user_preferred_sizes}: {sizes}")
            
            # Extract price preferences from user preferences
            user_min_budget = user_preferences.get("price_min") if user_preferences else None
            user_max_budget = user_preferences.get("price_max") if user_preferences else None
            
            # ➊  STREAM THE RAG RIGHT AWAY with filtering
            res = await search_and_pitch(
                optimized_query, 
                gender, 
                conversation_id, 
                conversation_redis_client, 
                allowed_product_types,
                sizes,
                user_min_budget,
                user_max_budget
            )
            
            # Filter RAG results to only include exact matches initially
            exact_match_pitches = {}
            non_exact_match_pitches = {}
            
            if res.get('pitches'):
                original_count = len(res['pitches'])
                
                for product_id, pitch in res['pitches'].items():
                    # Check both 'match' and 'exact_match' keys, convert to lowercase string
                    match_value = str(pitch.get('match', False)).lower()
                    exact_match_value = str(pitch.get('exact_match', False)).lower()
                    
                    if match_value == 'true' or exact_match_value == 'true':
                        exact_match_pitches[product_id] = pitch
                    else:
                        non_exact_match_pitches[product_id] = pitch
                
                print(f"RAG: Found {len(exact_match_pitches)} exact matches from {original_count} total results")
                
                # Check if we have 10 or more exact matches
                if len(exact_match_pitches) >= 10:
                    res['pitches'] = exact_match_pitches
                    print(f"RAG: Using {len(exact_match_pitches)} exact matches (>= 10 found)")
                else:
                    print(f"RAG: Only {len(exact_match_pitches)} exact matches found, need to run filter search")
            else:
                print("RAG: No pitches found, proceeding to filter search")
            
            # Always yield RAG results with exact matches (if any exist)
            if len(exact_match_pitches) > 0:
                res['pitches'] = exact_match_pitches
                yield {"event": "rag_search", "rag_results": "yes", "data": res}
                
                # Store RAG search results in Redis
                if conversation_id and res.get('pitches') and conversation_redis_client:
                    conversation_storage.set_redis_client(conversation_redis_client)
                    rag_product_ids = list(res['pitches'].keys())
                    await conversation_storage.store_search_results(
                        conversation_id=conversation_id,
                        search_type='rag',
                        product_ids=rag_product_ids,
                        metadata={
                            'query': optimized_query, 
                            'gender': gender,
                            'used_sizes': sizes,
                            'user_specified_sizes': user_preferred_sizes
                        }
                    )
            
            # Only skip filter search if we have 10+ exact matches
            if len(exact_match_pitches) >= 10:
                # ➌  GENERATE AND STREAM FOLLOW-UP MESSAGE (for RAG-only results)
                chat_history_text = f"User Query: {intent_analysis}\nSearch Results: {res} product recommendations"
                follow_up_message = await generate_follow_up_message(chat_history_text)
                yield {"event": "follow_up", "data": follow_up_message}
                return

            # Calculate how many more products we need
            remaining_slots = 10 - len(exact_match_pitches)

            # ➋  DO THE EXPENSIVE SEARCH (only if we have < 10 exact matches)
            search_results = await master_search.search_products(
                optimized_query, 
                gender, 
                conversation_id, 
                conversation_redis_client, 
                allowed_product_types,
                sizes,
                user_min_budget,
                user_max_budget,
                max_results=remaining_slots
            )

            if (
                search_results["status"] == "success"
                and search_results["products_found"] > 0
            ):
                final_reply = search_results["formatted_response"]
                print(f"Expensive search returned {len(final_reply.get('pitches', {}))} products to complement {len(exact_match_pitches)} RAG products")
                
                # Store filter search results in Redis
                if conversation_id and final_reply.get('pitches') and conversation_redis_client:
                    conversation_storage.set_redis_client(conversation_redis_client)
                    filter_product_ids = list(final_reply['pitches'].keys())
                    await conversation_storage.store_search_results(
                        conversation_id=conversation_id,
                        search_type='filter',
                        product_ids=filter_product_ids,
                        metadata={
                            'query': optimized_query, 
                            'gender': gender,
                            'used_sizes': sizes,
                            'user_specified_sizes': user_preferred_sizes
                        }
                    )
            else:
                final_reply = search_results.get(
                    "formatted_response",
                    "I'm sorry, I couldn't find any products that match your "
                    "search. Would you like to try different criteria?"
                )

            # ➌  STREAM THE FINAL RESPONSE
            yield {"event": "filter_search", "data": final_reply}
            
            # ➍  GENERATE AND STREAM FOLLOW-UP MESSAGE
            # Format chat history for the follow-up message
            chat_history_text = f"User Query: {intent_analysis}\nSearch Results: {res} {final_reply} product recommendations"
            follow_up_message = await generate_follow_up_message(chat_history_text)
            yield {"event": "follow_up", "data": follow_up_message}
            return

        # ───── ask_question ─────
        if sales_decision.get("action") == "ask_question":
            clarifier = sales_decision.get(
                "clarifying_question",
                "Could you please give me a few more details so I can help?"
            )
            yield {"event": "message", "data": clarifier}
            return

    # ────────────────────────────────────────────────────────────────────
    # FALL-BACK
    # ────────────────────────────────────────────────────────────────────
    fallback = (
        "I'm not sure how to help with that. "
        "Can you try asking another way?"
    )
    yield {"event": "message", "data": fallback}