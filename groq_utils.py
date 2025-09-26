import json
import os
from groq import AsyncGroq
from prompts import (
    TOP_LEVEL_ROUTER_PROMPT,
    GENERAL_AGENT_PROMPT,
    SALES_ROUTER_PROMPT,
    SEARCH_QUERY_CREATION_PROMPT,
    FOLLOW_UP_MESSAGE_PROMPT,
    ZERO_RESULTS_MESSAGE_PROMPT
)
from master_search import master_search
from rag_search_flow import search_and_pitch
from conversation_storage import conversation_storage
from logger_config import logger

# Use centralized Groq client from clients.py
from clients import groq_client

# Import for product metadata fetching
from redis_client_manager import get_redis_client
from redis.commands.search.query import Query

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
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"}
        )
        response_content = chat_completion.choices[0].message.content
        return json.loads(response_content)
    except Exception as e:
        from logger_config import log_detailed_error
        log_detailed_error(
            e,
            context="call_groq_json",
            local_vars={
                "message_for_groq": message_for_groq[:200],
                "system_prompt": system_prompt[:200],
                "model": model
            }
        )
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
        from logger_config import log_detailed_error
        log_detailed_error(
            e,
            context="call_groq_text",
            local_vars={
                "message_for_groq": message_for_groq[:200],
                "system_prompt": system_prompt[:200],
                "model": model
            }
        )
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

async def generate_search_query(intent_analysis: str, gender: str, chat_history: list = None, model: str = "llama-3.1-8b-instant") -> str:
    """
    Generate a concise Google-style search query from chat context.
    
    Args:
        intent_analysis (str): The derived intent analysis from the conversation.
        gender (str): Gender of the user (e.g., 'female', 'male', 'unisex').
        chat_history (list): The complete chat history.

    Returns:
        str: A concise search query.
    """
    # Convert chat history to text format
    chat_context = ""
    if chat_history:
        # Get last 3-5 messages for context
        recent_history = chat_history[-5:]
        chat_context = "\n".join([
            f"{'Customer' if msg.get('role') == 'customer' else msg.get('role', 'Unknown').capitalize()}: {msg.get('content', '')}"
            for msg in recent_history
        ])
    
    context_message = f'''Intent Analysis: {intent_analysis}
Gender: {gender}

Recent Chat History:
{chat_context if chat_context else "No previous conversation context."}

Generate a search query that considers both the current intent analysis and the conversation context.
Use the chat history to better understand what the user is looking for and make the search query more specific and relevant.'''
    
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
        # Get available sizes for the product types from Redis
        available_sizes = set()
        try:
            # Use centralized Redis client
            from redis_client_manager import get_redis_client
            from redis.commands.search.query import Query
            
            redis_client = get_redis_client()
            
            # Query Redis for sizes by product types - try different possible field names
            size_fields_to_try = ["sizes", "size_options", "size", "available_sizes"]
            
            for product_type in product_types:
                found_sizes = False
                for size_field in size_fields_to_try:
                    try:
                        query = Query(f"@product_type:{{{product_type}}}").return_fields(size_field).paging(0, 100)
                        results = redis_client.ft("idx:product_metadata").search(query)
                        
                        for doc in results.docs:
                            if hasattr(doc, size_field):
                                size_value = getattr(doc, size_field)
                                if size_value:
                                    # Parse sizes (could be JSON string or comma-separated)
                                    try:
                                        import json
                                        if isinstance(size_value, str):
                                            # Try JSON first
                                            try:
                                                parsed_sizes = json.loads(size_value)
                                                if isinstance(parsed_sizes, list):
                                                    available_sizes.update(parsed_sizes)
                                                    found_sizes = True
                                                    continue
                                            except:
                                                pass
                                            # Try comma-separated
                                            parsed_sizes = [s.strip() for s in size_value.split(',') if s.strip()]
                                            if parsed_sizes:
                                                available_sizes.update(parsed_sizes)
                                                found_sizes = True
                                        elif isinstance(size_value, list):
                                            available_sizes.update(size_value)
                                            found_sizes = True
                                    except Exception as parse_error:
                                        print(f"⚠️ Error parsing sizes for {product_type}: {parse_error}")
                        
                        if found_sizes:
                            print(f"✅ Found sizes for {product_type} using field '{size_field}'")
                            break
                            
                    except Exception as e:
                        print(f"⚠️ Error querying {size_field} for {product_type}: {e}")
                        continue
                
                if not found_sizes:
                    print(f"⚠️ No sizes found for product type: {product_type}")
                    
        except Exception as e:
            print(f"⚠️ Error connecting to Redis for sizes: {e}")
        
        # Ensure we always have some sizes available
        if not available_sizes:
            print("⚠️ No sizes found in Redis, using default sizes")
            available_sizes = {'s', 'm', 'l', 'xl', 'xs', 'xxl'}
            
        available_sizes_list = sorted(list(available_sizes))
        print(f"📏 Available sizes for {product_types}: {available_sizes_list}")
        
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
        from logger_config import log_detailed_error
        log_detailed_error(
            e,
            context="generate_sizes_for_product_types",
            local_vars={
                "user_preferred_size": user_preferred_size,
                "product_types": product_types,
                "available_sizes": locals().get("available_sizes_list", [])[:10],
                "model": model
            }
        )
        # Fallback: return common sizes
        print("❌ Error generating sizes: using fallback common sizes")
        return {"sizes": ['s', 'm', 'l']}

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
        from logger_config import log_detailed_error
        log_detailed_error(
            e,
            context="generate_follow_up_message",
            local_vars={
                "chat_history": chat_history[:200],
                "model": model,
                "formatted_prompt": locals().get("formatted_prompt", "")[:200]
            }
        )
        # Fallback message
        return "Hope you found something you love! Let me know if you'd like to see more options or have any questions. 😊"

async def generate_zero_results_message(user_query: str, shortlisted_products: list = None, model: str = "llama-3.1-8b-instant") -> str:
    """
    Generate a helpful message when zero products are found from both RAG and filter searches,
    using shortlisted products that were considered but not recommended.
    
    Args:
        user_query (str): The user's original search query and context
        shortlisted_products (list): List of products that were shortlisted but not recommended
        model (str): Groq model to use
        
    Returns:
        str: A helpful, encouraging message highlighting shortlisted product features
    """
    print("🔍 Generating zero results message...")
    print(f"📝 Shortlisted products available: {len(shortlisted_products) if shortlisted_products else 0}")
    
    max_retries = 3
    required_keys = {"acknowledgement", "feature_highlight", "search_refinement_offer"}
    
    try:
        # Prepare context with shortlisted product descriptions if available
        context_data = f"User Query: {user_query}"
        
        if shortlisted_products and len(shortlisted_products) > 0:
            # Extract product descriptions and key features for analysis
            product_descriptions = []
            for product in shortlisted_products[:10]:  # Limit to 10 to avoid token limits
                desc_parts = []
                
                # Get title
                title = product.get('title', '').strip()
                if title:
                    desc_parts.append(f"Title: {title}")
                
                # Get description (prioritize body_text which contains full product description)
                description = product.get('body_text', '').strip() or product.get('text_description', '').strip()
                if description:
                    # Limit description to first 200 characters to avoid token bloat
                    if len(description) > 200:
                        description = description[:200] + "..."
                    desc_parts.append(f"Description: {description}")
                
                # Add price if available
                price = product.get('max_price')
                if price and float(price) > 0:
                    desc_parts.append(f"Price: ₹{price}")
                
                if desc_parts:
                    product_descriptions.append(" | ".join(desc_parts))
            
            if product_descriptions:
                context_data += f"\n\nShortlisted Products Found (considered but didn't match exactly):\n"
                for i, desc in enumerate(product_descriptions, 1):
                    context_data += f"{i}. {desc}\n"
                
                context_data += f"\nTotal shortlisted products analyzed: {len(product_descriptions)}"
        
        # Format the prompt with enhanced context
        formatted_prompt = ZERO_RESULTS_MESSAGE_PROMPT.format(user_query=context_data)
        
        # Store conversation messages for retry attempts
        messages = [
            {
                "role": "user",
                "content": formatted_prompt
            }
        ]
        
        for attempt in range(max_retries):
            try:
                # Add assistant role for continuation
                current_messages = messages + [{"role": "assistant", "content": "```json"}]
                
                completion = await groq_client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    stop="```",
                    temperature=0.7
                )
                response_content = completion.choices[0].message.content
                response = json.loads(response_content)
                
                # Validate that all required keys are present and non-empty
                missing_keys = []
                empty_keys = []
                
                for key in required_keys:
                    if key not in response:
                        missing_keys.append(key)
                    elif not response[key] or not response[key].strip():
                        empty_keys.append(key)
                
                if not missing_keys and not empty_keys:
                    # All required keys are present and valid
                    acknowledgement = response.get("acknowledgement", "")
                    feature_highlight = response.get("feature_highlight", "")
                    search_refinement_offer = response.get("search_refinement_offer", "")
                    
                    # Combine all three parts with markdown line breaks and extra gaps
                    combined_message = f"{acknowledgement}  \n\n{feature_highlight}  \n\n{search_refinement_offer}".strip()
                    
                    print(f"✅ Generated zero results message: {combined_message[:100]}...")
                    return combined_message
                else:
                    # Some keys are missing or empty - add feedback for retry
                    issues = []
                    if missing_keys:
                        issues.append(f"Missing required keys: {missing_keys}")
                    if empty_keys:
                        issues.append(f"Empty values for keys: {empty_keys}")
                    
                    print(f"Attempt {attempt + 1}: JSON validation failed - {'; '.join(issues)}")
                    
                    if attempt < max_retries - 1:
                        # Add the assistant's response to conversation history
                        messages.append({
                            "role": "assistant", 
                            "content": f"```json\n{response_content}\n```"
                        })
                        
                        # Add feedback message for retry
                        feedback_message = f"""Your previous JSON response had issues: {'; '.join(issues)}

Required JSON format with all fields non-empty:
{{
  "acknowledgement": "Brief positive acknowledgment sentence formatted with markdown for readability",
  "feature_highlight": "Specific mention of materials, styles, colors found in shortlisted products formatted with markdown for readability", 
  "search_refinement_offer": "Offer to refine search or explore alternatives formatted with markdown for readability"
}}

Please provide a complete JSON response with all three required fields containing meaningful content."""
                        
                        messages.append({
                            "role": "user",
                            "content": feedback_message
                        })
                    else:
                        # Last attempt failed, use fallback with any valid parts
                        print(f"Final attempt: Using fallback with partial data")
                        acknowledgement = response.get("acknowledgement", "We couldn't find exact matches for your search, but I discovered some exciting alternatives.")
                        feature_highlight = response.get("feature_highlight", "I found lovely styles and colors that caught my attention.")
                        search_refinement_offer = response.get("search_refinement_offer", "Would you like to explore these styles or shall I help you refine your search?")
                        
                        combined_message = f"{acknowledgement}  \n\n{feature_highlight}  \n\n{search_refinement_offer}".strip()
                        return combined_message
                
            except json.JSONDecodeError as e:
                print(f"Attempt {attempt + 1} JSON parsing error: {str(e)}")
                if attempt < max_retries - 1:
                    # Add error feedback for retry
                    if len(messages) > 1:  # If there was a previous assistant response
                        messages.append({
                            "role": "user",
                            "content": f"""There was a JSON parsing error in your previous response: {str(e)}
Please provide a valid JSON response with the exact format:
{{
  "acknowledgement": "Brief positive acknowledgment sentence with markdown formatting",
  "feature_highlight": "Specific mention of materials, styles, colors with markdown formatting", 
  "search_refinement_offer": "Offer to refine search with markdown formatting"
}}

Make sure your JSON is properly formatted and contains all three required fields."""
                        })
                continue
            except Exception as e:
                print(f"Attempt {attempt + 1} general error: {str(e)}")
                if attempt < max_retries - 1:
                    continue
        
        # If all attempts failed, return fallback message
        print(f"❌ All {max_retries} attempts failed, using fallback message")
        return "We couldn't find exact matches for your search, but we have lots of amazing styles to explore! Try browsing our collections or searching with broader terms - I'm here to help you find something perfect! ✨"
        
    except Exception as e:
        from logger_config import log_detailed_error
        log_detailed_error(
            e,
            context="generate_zero_results_message",
            local_vars={
                "user_query": user_query[:200],
                "shortlisted_count": len(shortlisted_products) if shortlisted_products else 0,
                "model": model,
                "formatted_prompt": locals().get("formatted_prompt", "")[:200]
            }
        )
        # Fallback message for zero results
        return "We couldn't find exact matches for your search, but we have lots of amazing styles to explore! Try browsing our collections or searching with broader terms - I'm here to help you find something perfect! ✨"

async def fetch_shortlisted_product_details(product_ids: list) -> list:
    """
    Fetch full product details (including descriptions) for a list of shortlisted product IDs.
    
    Args:
        product_ids: List of product IDs to fetch details for
        
    Returns:
        List of product dictionaries with title, description, and metadata
    """
    if not product_ids:
        return []
    
    try:
        print(f"🔍 Fetching details for {len(product_ids)} shortlisted products")
        
        # Use centralized Redis client
        redis_client = get_redis_client()
        
        # Create OR query for all product IDs (limit to 20 to avoid token limits)
        limited_ids = product_ids[:20]
        query_string = " | ".join([f"@internal_id:{id_}" for id_ in limited_ids])
        metadata_query = (
            Query(query_string)
            .return_fields("internal_id", "image_url", "product_url", "title", "max_price", "body_text")
        )
        
        # Execute the query
        metadata_results = redis_client.ft("idx:product_metadata").search(metadata_query)
        
        # Build the product details list
        product_details = []
        for doc in metadata_results.docs:
            product_details.append({
                'product_id': doc.internal_id,
                'title': getattr(doc, 'title', f"Product {doc.internal_id}"),
                'image_url': getattr(doc, 'image_url', ''),
                'product_url': getattr(doc, 'product_url', f"/product/{doc.internal_id}"),
                'max_price': getattr(doc, 'max_price', 0),
                'body_text': getattr(doc, 'body_text', ''),  # Full product description
                'text_description': getattr(doc, 'body_text', '')  # Same as body_text for consistency
            })
        
        print(f"✅ Successfully fetched {len(product_details)} product details")
        return product_details
        
    except Exception as e:
        from logger_config import log_detailed_error
        log_detailed_error(
            e,
            context="fetch_shortlisted_product_details",
            local_vars={
                "product_ids_count": len(product_ids),
                "first_few_ids": product_ids[:5]
            }
        )
        print(f"❌ Error fetching product details: {e}")
        return []

# --- Main Chatbot Logic ---
from typing import AsyncIterator, Dict, Any

async def chatbot_flow(intent_analysis: str, conversation_id: str = None, conversation_redis_client=None, user_preferences: Dict[str, Any] = None, chat_history: list = None) -> AsyncIterator[Dict[str, Any]]:
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
        yield {"event": "general_message", "data": reply}
        return

    # ────────────────────────────────────────────────────────────────────
    # SALES FLOW
    # ────────────────────────────────────────────────────────────────────
    if intent_decision == "sales_query":
        sales_decision = await sales_router_agent(intent_analysis)

        # ───── perform_product_search ─────
        if sales_decision.get("action") == "perform_product_search":
            # Use all genders from user preferences (now a list), default to "female"
            genders = user_preferences.get("gender", ["female"]) if user_preferences else ["female"]
            genders = genders if genders and len(genders) > 0 else ["female"]
            print(f"🔄 Using user preferences: {user_preferences}")
            print(f"🎯 Genders for search: {genders}")
            # Use first gender for query generation (search query doesn't need to be gender-specific)
            optimized_query = await generate_search_query(intent_analysis, genders[0], chat_history)

            # Extract product categories once for both RAG and filter searches
            from unified_product_category import unified_product_category_extractor
            category_response = await unified_product_category_extractor.get_product_category(optimized_query, genders)
            allowed_product_types = category_response.get('garment_type', [])
            
            # No size filtering - sizes removed from user preferences
            sizes = []
            print(f"🎯 No size filtering applied")
            
            # Extract price preferences from user preferences
            user_min_budget = user_preferences.get("price_min") if user_preferences else None
            user_max_budget = user_preferences.get("price_max") if user_preferences else None
            
            # ➊  STREAM THE RAG RIGHT AWAY with filtering
            res = await search_and_pitch(
                optimized_query, 
                genders, 
                conversation_id, 
                conversation_redis_client, 
                allowed_product_types,
                sizes,
                user_min_budget,
                user_max_budget
            )
            
            print('rag_results - ', json.dumps(res, indent=4))
            # Get RAG search results (now in unified structure)
            rag_results = res.get('formatted_response', {}).get('search_results', [])
            rag_count = len(rag_results)
            
            print(f"RAG: Found {rag_count} search results")
            
            # Store RAG results BEFORE yielding to prevent race conditions
            if rag_count > 0:
                # Store RAG search results in Redis FIRST
                if conversation_id and rag_results and conversation_redis_client:
                    conversation_storage.set_redis_client(conversation_redis_client)
                    rag_product_ids = [result['product_id'] for result in rag_results]
                    await conversation_storage.store_search_results(
                        conversation_id=conversation_id,
                        search_type='rag',
                        product_ids=rag_product_ids,
                        metadata={
                            'query': optimized_query, 
                            'genders': genders,
                            'used_sizes': sizes
                        }
                    )
                
                # Then yield RAG results to frontend
                yield {"event": "rag_search", "rag_results": "yes", "data": res.get('formatted_response', {"search_results": []})}
            
            # Only skip filter search if we have 10+ results
            if rag_count >= 10:
                # ➌  GENERATE AND STREAM FOLLOW-UP MESSAGE (for RAG-only results)
                chat_history_text = f"User Query: {intent_analysis}\nSearch Results: {rag_count} product recommendations"
                follow_up_message = await generate_follow_up_message(chat_history_text)
                yield {"event": "follow_up", "data": follow_up_message}
                return

            # Calculate how many more products we need
            remaining_slots = 10 - rag_count

            # ➋  DO THE EXPENSIVE SEARCH (only if we have < 10 results)
            search_results = await master_search.search_products(
                optimized_query, 
                genders, 
                conversation_id, 
                conversation_redis_client, 
                allowed_product_types,
                sizes,
                user_min_budget,
                user_max_budget,
                max_results=remaining_slots
            )
            print("expensive search_results", json.dumps(search_results, indent=4))
            if (
                search_results["status"] == "success"
                and search_results["products_found"] > 0
            ):
                final_reply = search_results["formatted_response"]
                filter_results = final_reply.get('search_results', {})
                
                # Handle both dict and list formats for filter_results
                if isinstance(filter_results, list):
                    filter_count = len(filter_results)
                    filter_product_ids = [item.get('product_id') for item in filter_results if item.get('product_id')]
                else:
                    filter_count = len(filter_results)
                    filter_product_ids = list(filter_results.keys())
                
                print(f"Expensive search returned {filter_count} products to complement {rag_count} RAG products")
                
                # Store filter search results in Redis
                if conversation_id and filter_product_ids and conversation_redis_client:
                    conversation_storage.set_redis_client(conversation_redis_client)
                    await conversation_storage.store_search_results(
                        conversation_id=conversation_id,
                        search_type='filter',
                        product_ids=filter_product_ids,
                        metadata={
                            'query': optimized_query, 
                            'genders': genders,
                            'used_sizes': sizes
                        }
                    )
            else:
                final_reply = search_results.get(
                    "formatted_response",
                    "I'm sorry, I couldn't find any products that match your "
                    "search. Would you like to try different criteria?"
                )

            # ➍  GENERATE AND STREAM FOLLOW-UP MESSAGE
            # Format chat history for the follow-up message
            if isinstance(final_reply, dict):
                search_results_data = final_reply.get('search_results', {})
                if isinstance(search_results_data, list):
                    additional_count = len(search_results_data)
                else:
                    additional_count = len(search_results_data)
            else:
                additional_count = 0
            total_results = rag_count + additional_count
            
            # ➺  ZERO RESULTS DETECTION - Generate special message when no products found
            if total_results == 0:
                print(f"🚫 Zero results detected: RAG={rag_count}, Filter={additional_count}")
                
                # Collect reranked products from both search methods with their scores
                all_reranked_products = {}
                
                # Get reranked results from RAG search (from ranked_products)
                if res.get("ranked_products"):
                    rag_reranked = res["ranked_products"]
                    for product_id, score in rag_reranked.items():
                        all_reranked_products[product_id] = score
                    print(f"📋 Found {len(rag_reranked)} reranked products from RAG search")
                
                # Get reranked results from expensive search (from ranked_products)
                if search_results.get("ranked_products"):
                    expensive_reranked = search_results["ranked_products"]
                    for product_id, score in expensive_reranked.items():
                        # Use max score if product exists in both searches
                        if product_id in all_reranked_products:
                            all_reranked_products[product_id] = max(all_reranked_products[product_id], score)
                        else:
                            all_reranked_products[product_id] = score
                    print(f"📋 Found {len(expensive_reranked)} reranked products from expensive search")
                
                # Sort by reranked score and get top 3
                top_3_products = sorted(all_reranked_products.items(), key=lambda x: x[1], reverse=True)[:3]
                top_3_product_ids = [product_id for product_id, score in top_3_products]
                
                print(f"📊 Selected top 3 products by rerank score: {top_3_product_ids}")
                
                # Fetch product details for zero-results message generation first
                all_shortlisted_ids = list(all_reranked_products.keys())
                shortlisted_products_with_details = []
                if all_shortlisted_ids:
                    shortlisted_products_with_details = await fetch_shortlisted_product_details(all_shortlisted_ids)
                    print(f"✅ Retrieved details for {len(shortlisted_products_with_details)} shortlisted products for message")
                
                # Generate and yield zero-results message FIRST
                zero_results_message = await generate_zero_results_message(
                    intent_analysis, 
                    shortlisted_products_with_details
                )
                yield {"event": "follow_up", "data": zero_results_message}
                
                # Then yield top 3 products as search results
                if top_3_product_ids:
                    # Fetch full product details for top 3 products
                    top_3_products_with_details = await fetch_shortlisted_product_details(top_3_product_ids)
                    print(f"✅ Retrieved details for {len(top_3_products_with_details)} top products")
                    
                    # Format top 3 products as search results (same format as rag_search/filter_search)
                    if top_3_products_with_details:
                        formatted_top_3 = {
                            "search_results": []
                        }
                        
                        for product in top_3_products_with_details:
                            formatted_product = {
                                "product_id": product.get('product_id'),
                                "title": product.get('title', f"Product {product.get('product_id')}"),
                                "image_url": product.get('image_url', ''),
                                "product_url": product.get('product_url', f"/product/{product.get('product_id')}"),
                                "max_price": product.get('max_price', 0),
                                "relevance_score": all_reranked_products.get(product.get('product_id'), 0),
                                "text_description": product.get('body_text', ''),
                                "body_text": product.get('body_text', '')
                            }
                            formatted_top_3["search_results"].append(formatted_product)
                        
                        # Yield top 3 products as search results
                        yield {"event": "filter_search", "data": formatted_top_3}
                        
                        # Store top 3 products for future deduplication
                        if conversation_id and conversation_redis_client:
                            conversation_storage.set_redis_client(conversation_redis_client)
                            await conversation_storage.store_search_results(
                                conversation_id=conversation_id,
                                search_type='filter',
                                product_ids=top_3_product_ids,
                                metadata={
                                    'query': intent_analysis, 
                                    'type': 'zero_results_top_products',
                                    'rerank_scores': {pid: all_reranked_products.get(pid, 0) for pid in top_3_product_ids}
                                }
                            )
            else:
                # ➌  STREAM THE FINAL RESPONSE (only if we have results)
                yield {"event": "filter_search", "data": final_reply}
                
                # Normal follow-up for successful searches
                chat_history_text = f"User Query: {intent_analysis}\nSearch Results: {total_results} product recommendations"
                follow_up_message = await generate_follow_up_message(chat_history_text)
                yield {"event": "follow_up", "data": follow_up_message}
            return

        # ───── ask_question ─────
        if sales_decision.get("action") == "ask_question":
            clarifier = sales_decision.get(
                "clarifying_question",
                "Could you please give me a few more details so I can help?"
            )
            yield {"event": "ask_question", "data": clarifier}
            return

    # ────────────────────────────────────────────────────────────────────
    # FALL-BACK
    # ────────────────────────────────────────────────────────────────────
    fallback = (
        "I'm not sure how to help with that. "
        "Can you try asking another way?"
    )
    yield {"event": "message", "data": fallback}