import os
import json
import traceback
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from groq import AsyncGroq
from dotenv import load_dotenv, find_dotenv
import redis.asyncio as redis
from groq_utils import chatbot_flow
from logger_config import logger, log_error, error_handler
from image_utils import image_processor
from intent_classifier import intent_classifier
from pydantic import ValidationError
from prompts import USER_SUMMARY_SYSTEM_PROMPT, FASHION_SALESPERSON_SYSTEM_PROMPT, DETAIL_COLLECTION_PROMPT, INTENT_DERIVATION_SYSTEM_PROMPT
from response_models import (
    UnifiedResponse, ResponseType, ResponseStatus, SectionType, Section,
    ResponseData, ResponseMetadata, ProductSearchResult
)

# --- Environment and Configuration ---
load_dotenv()  # Load from .env file or environment variables

# --- Groq Client Configuration ---
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# --- Redis Client Configuration ---
from redis_client_manager import get_async_redis_client
redis_client = get_async_redis_client()

# --- Global metadata variables (loaded from Redis at startup) ---
group_descriptions = {}
reversed_attribute_mappings = {}
grouped_values = {}

# --- System Prompts imported from prompts.py ---

# --- Pydantic Models ---
class UserPreferences(BaseModel):
    preferences: Dict[str, Any] = {}

app = FastAPI()

# --- Startup/Shutdown Events ---
@app.on_event("startup")
async def load_metadata_from_redis():
    global group_descriptions, reversed_attribute_mappings, grouped_values
    
    # Load group_descriptions from Redis
    group_descriptions_json = await redis_client.get("group_descriptions")
    if group_descriptions_json:
        group_descriptions = json.loads(group_descriptions_json)
        print(f"Loaded group_descriptions for {len(group_descriptions)} product types from Redis")
        
        # Compute reversed_attribute_mappings from group_descriptions
        reversed_attribute_mappings = {}
        for product_type, groups in group_descriptions.items():
            reversed_attribute_mappings[product_type] = {}
            for group in groups.values():
                attr_name = group.get('group_name')
                attr_description = group.get('group_description')
                if attr_name and attr_description:
                    reversed_attribute_mappings[product_type][attr_description] = attr_name
        
        print(f"Computed reversed_attribute_mappings for {len(reversed_attribute_mappings)} product types")
    else:
        print("Warning: group_descriptions not found in Redis")
    
    # Load grouped_values (attributes_and_candidates) from Redis  
    attributes_and_candidates_json = await redis_client.get("attributes_and_candidates")
    if attributes_and_candidates_json:
        grouped_values = json.loads(attributes_and_candidates_json)
        print(f"Loaded grouped_values for {len(grouped_values)} product types from Redis")
    else:
        print("Warning: attributes_and_candidates not found in Redis")

@app.on_event("shutdown")
async def cleanup_redis_connections():
    """Clean up Redis connections on shutdown"""
    from redis_client_manager import RedisClientManager
    await RedisClientManager.close_connections()
    print("🔒 Redis connections closed on shutdown")

# --- CORS Middleware ---
origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom exception handler for Pydantic's validation errors.
    Returns a detailed error response to help with debugging.
    """
    print("--- VALIDATION ERROR (422) ---")
    print("Validation Errors:", exc.errors())
    print("Request URL:", request.url)
    print("Request Method:", request.method)
    print("------------------------------")
    
    # Return validation errors without trying to read consumed body
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "request_url": str(request.url)},
    )

# --- Helper Functions ---

async def get_user_summary_from_redis(redis_key: str) -> Dict[str, Any]:
    """Retrieves user summary from Redis."""
    summary_json = await redis_client.get(redis_key)
    if summary_json:
        return json.loads(summary_json)
    return {"preferences": {}}

async def store_user_summary_in_redis(redis_key: str, summary: Dict[str, Any]) -> None:
    """Stores user summary in Redis with TTL."""
    await redis_client.set(redis_key, json.dumps(summary), ex=3600)  # 1 hour TTL

async def get_available_gender_options() -> List[str]:
    """Get available gender options from Redis index."""
    try:
        from redis.commands.search.aggregation import AggregateRequest
        
        aggregate_request = AggregateRequest("*").group_by(["@gender"])
        agg_results = await redis_client.ft("idx:product_text_description_embedding").aggregate(aggregate_request)
        
        # Extract distinct gender values
        distinct_genders = [result[1] for result in agg_results.rows]
        
        print(f"📋 Available gender options: {distinct_genders}")
        return distinct_genders
        
    except Exception as e:
        print(f"⚠️ Error getting gender options from Redis: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        # Fallback to common gender options
        return ['female', 'male', 'unisex']

async def generate_user_summary_with_groq(chat_history: List[Dict[str, str]], existing_user_summary: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generates user summary from chat history using Groq with JSON validation and retry logic."""
    print("--- Generating User Summary with Groq ---")
    
    # Convert last 5 messages and handle 'customer' role
    last_messages = chat_history[-5:]
    chat_content = "\n".join([
        f"{'Customer' if msg['role'] == 'customer' else msg['role'].capitalize()}: {msg['content']}" 
        for msg in last_messages
    ])
    chat_content += '''\n Return only a JSON object with this structure:

                                    {
                                    "preferences": {}
                                    }
                                    '''
    print("chat_content - ", chat_content)
    # Build system prompt with existing summary if available
    system_content = USER_SUMMARY_SYSTEM_PROMPT
    if existing_user_summary:
        existing_summary_text = json.dumps(existing_user_summary, indent=2)
        system_content += f"\n\nEXISTING USER SUMMARY:\n{existing_summary_text}\n\nUpdate this summary with any new information from the chat history."
    
    # Retry logic with JSON validation using UserPreferences model
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"--- Attempt {attempt + 1} of {max_retries} ---")
            
            response = await groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": chat_content}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            summary_json = response.choices[0].message.content
            print("--- Raw User Summary Response ---", summary_json)
            
            # Parse JSON
            parsed_summary = json.loads(summary_json)
            
            # Gender validation removed - gender comes from frontend now
            
            # Validate structure using UserPreferences Pydantic model
            validated_summary = UserPreferences(**parsed_summary)
            
            print("--- User Summary Generated and Validated ---", json.dumps(validated_summary.model_dump(), indent=4))
            return validated_summary.model_dump()
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error on attempt {attempt + 1}: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            if attempt == max_retries - 1:
                print("Max retries reached. Returning default summary.")
                return {"preferences": {}}
            continue
            
        except ValidationError as e:
            print(f"Pydantic validation error on attempt {attempt + 1}: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            if attempt == max_retries - 1:
                print("Max retries reached. Returning default summary.")
                return {"preferences": {}}
            continue
            
        except Exception as e:
            from logger_config import log_detailed_error
            log_detailed_error(
                e,
                context=f"generate_user_summary_with_groq.attempt_{attempt + 1}",
                local_vars={
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "chat_content": chat_content[:200] if 'chat_content' in locals() else None,
                    "existing_user_summary": existing_user_summary
                }
            )
            if attempt == max_retries - 1:
                print("Max retries reached. Returning default summary.")
                return {"preferences": {}}
            continue
    
    # Fallback (should not reach here)
    return {"preferences": {}}


async def validate_audio_file(file: UploadFile) -> None:
    """Validates audio file format and size according to Groq requirements."""
    # Supported formats from Groq documentation
    supported_formats = {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"}
    
    # Get file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in supported_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported audio format. Supported formats: {', '.join(supported_formats)}"
        )
    
    # Check file size (25MB limit for free tier)
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    # Reset file pointer for later reading
    await file.seek(0)
    
    max_size = 25 * 1024 * 1024  # 25MB in bytes
    if file_size > max_size:
        raise HTTPException(
            status_code=400, 
            detail=f"File size ({file_size / (1024*1024):.2f}MB) exceeds maximum allowed size (25MB)"
        )
    
    print(f"Audio file validated: {file.filename}, size: {file_size / (1024*1024):.2f}MB")

async def transcribe_audio_with_groq(file: UploadFile) -> str:
    """Transcribes audio file to text using Groq's Speech-to-Text API."""
    print("--- Transcribing Audio with Groq Speech-to-Text ---")
    try:
        # Read file content
        audio_content = await file.read()
        
        # Create transcription request
        transcription = await groq_client.audio.transcriptions.create(
            file=(file.filename, audio_content, file.content_type),
            model="whisper-large-v3-turbo",  # Cost-efficient model
            temperature=0.0,  # For consistent results
            response_format="text"
        )
        
        transcribed_text = transcription
        print(f"--- Audio Transcribed Successfully --- Text: {transcribed_text[:100]}...")
        return transcribed_text
        
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        from logger_config import log_detailed_error
        log_detailed_error(
            e,
            context="transcribe_audio_with_groq",
            local_vars={
                "filename": file.filename,
                "file_size": locals().get("file_size"),
                "content_type": getattr(file, 'content_type', 'unknown')
            }
        )
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {e}")

# --- FastAPI Endpoints ---





@app.post("/chat/unified")
@error_handler("Unified Chat Endpoint")
async def unified_chat_handler(
    request: Request,
    chat_uuid: str = Form(...),
    image_file: Optional[Union[UploadFile, str]] = File(None),
    audio_file: Optional[Union[UploadFile, str]] = File(None), 
    text_message: Optional[str] = Form(None),
    chat_history: Optional[str] = Form(None),
    gender_preferences: Optional[str] = Form(None),
    budget_preference: Optional[str] = Form(None),
    ):
    """
    Unified endpoint that handles voice, image, and text input.
    Streams Server-Side Events with product recommendations using existing chatbot_flow.
    """
    
    # Read file contents BEFORE creating the async generator
    image_content = None
    if image_file and not (isinstance(image_file, str) and not image_file):
        try:
            raw_content = await image_file.read()
            if raw_content:
                image_content = await image_processor.validate_image_content(raw_content, image_file.filename)
                logger.info(f"Image uploaded: {image_file.filename}, size: {len(raw_content) / (1024*1024):.2f}MB")
            else:
                logger.error("Empty image file content")
                return JSONResponse(
                    status_code=400,
                    content={"error": "Empty image file"}
                )
        except Exception as e:
            logger.error(f"Error reading image file: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return JSONResponse(
                status_code=400,
                content={"error": "Failed to process image file"}
            )
    
    # Read audio file content BEFORE creating the async generator
    voice_transcription = ""
    if audio_file and not (isinstance(audio_file, str) and not audio_file):
        try:
            await validate_audio_file(audio_file)
            voice_transcription = await transcribe_audio_with_groq(audio_file)
            logger.info(f"Audio transcribed: {voice_transcription[:100]}")
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return JSONResponse(
                status_code=400,
                content={"error": "Failed to process audio file"}
            )
    
    async def generate_sse_stream():
        import time
        start_time = time.time()
        
        try:
            # Validate that at least one input is provided
            if not image_content and not voice_transcription and not text_message:
                yield f"data: {json.dumps({'error': 'At least one input (image, audio, or text) must be provided'})}\n\n"
                return
            
            # Parse chat history
            chat_history_list = []
            if chat_history:
                try:
                    chat_history_list = json.loads(chat_history)
                except json.JSONDecodeError:
                    chat_history_list = []
            
            # Create structured user query with clear input labels
            user_query_parts = []
            if text_message:
                user_query_parts.append(f"Text input: {text_message}")
            if voice_transcription:
                user_query_parts.append(f"Voice input: {voice_transcription}")
            
            user_query = " | ".join(user_query_parts)
            
            if not user_query:
                yield f"data: {json.dumps({'error': 'No valid input to process'})}\n\n"
                return
            
            # Store original user query in chat history FIRST
            chat_history_list.append({
                "role": "customer", 
                "content": user_query
            })
            
            # Parse frontend preferences
            try:
                frontend_genders = json.loads(gender_preferences) if gender_preferences else ["female"]
            except (json.JSONDecodeError, TypeError):
                frontend_genders = ["female"]
            
            try:
                frontend_budget = json.loads(budget_preference) if budget_preference else {}
            except (json.JSONDecodeError, TypeError):
                frontend_budget = {}
                
            # Generate user summary using existing function (with latest message included)
            redis_key = f"chat_summary:{chat_uuid}"
            existing_summary = await get_user_summary_from_redis(redis_key)
            chat_preferences = await generate_user_summary_with_groq(chat_history_list, existing_summary)
            await store_user_summary_in_redis(redis_key, chat_preferences)
            
            # Combine chat preferences with frontend preferences
            user_preferences = {
                "preferences": chat_preferences.get("preferences", {}),
                "gender": frontend_genders,
                "price_min": frontend_budget.get("min_price"),
                "price_max": frontend_budget.get("max_price")
            }
            
            # Determine intent from text/voice only (not image)
            intent_result = await intent_classifier.classify_user_intent(user_query, has_image=bool(image_content), chat_history=chat_history_list)
            intent = intent_result.get("intent", "COMPLEMENT")
            
            print(f"Determined intent: {intent}")
            
            # Store updated chat history
            messages_key = f"chat_messages:{chat_uuid}"
            await redis_client.set(messages_key, json.dumps(chat_history_list), ex=3600)
            
            # Send initial status event
            yield f"data: {json.dumps({'event': 'status', 'status': 'processing', 'intent': intent})}\n\n"
            
            # Process based on intent with streaming
            if intent == "SEARCH":
                # SEARCH FLOW: Generate single search query from text + optional image
                if image_content:
                    search_query_response = await image_processor.generate_search_query_for_intent(user_query, image_content, "SEARCH", chat_history_list)
                    
                    # Parse JSON response from image search
                    try:
                        search_query_json = json.loads(search_query_response)
                        search_query = search_query_json.get("search_query", user_query)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Failed to parse image search response: {search_query_response}")
                        print(f"Full traceback: {traceback.format_exc()}")
                        search_query = user_query  # Fallback to user query
                else:
                    # Text-only search - let LLM optimize the query
                    search_query = await image_processor.generate_search_query_for_intent(user_query, None, "SEARCH", chat_history_list)
                
                print(f"SEARCH query: {search_query}")
                
                # Send query processed event
                yield f"data: {json.dumps({'event': 'search_query', 'query': search_query})}\n\n"
                
                # Stream results from chatbot_flow
                async for result in chatbot_flow(
                    intent_analysis=search_query,
                    conversation_id=chat_uuid,
                    conversation_redis_client=redis_client,
                    user_preferences=user_preferences,
                    chat_history=chat_history_list
                ):
                    event_type = result.get("event", "")
                    if event_type in ["rag_search", "filter_search"]:
                        # Process and stream search results
                        data = result.get("data", {})
                        if isinstance(data, dict) and "search_results" in data:
                            search_results = data["search_results"]
                            products = []
                            
                            for search_result in search_results:
                                try:
                                    product = {
                                        "title": search_result.get("title", f"Product {search_result['product_id']}"),
                                        "image_url": search_result.get("image_url", ""),
                                        "product_url": search_result.get("product_url", f"/product/{search_result['product_id']}"),
                                        "max_price": float(search_result.get("max_price", 0)),
                                        "relevance_score": search_result.get("relevance_score"),
                                        "llm_filter_reason": search_result.get("text_description", ""),
                                        "body_text": search_result.get("body_text", "")
                                    }
                                    products.append(product)
                                except Exception as e:
                                    print(f"Error processing search result {search_result.get('product_id', 'unknown')}: {e}")
                                    print(f"Full traceback: {traceback.format_exc()}")
                            
                            # Stream the results
                            stream_data = {
                                "event": event_type,
                                "data": {
                                    "section_type": "SEARCH_RESULTS",
                                    "title": "Search Results",
                                    "description": f"Found {len(products)} items matching your search",
                                    "products": products,
                                    "total_results": len(products)
                                }
                            }
                            yield f"data: {json.dumps(stream_data)}\n\n"
                    
                    elif event_type == "follow_up":
                        # Stream follow-up message
                        follow_up_data = {
                            "event": "follow_up",
                            "data": result.get("data", "")
                        }
                        yield f"data: {json.dumps(follow_up_data)}\n\n"
                
                # Send completion event
                processing_time = time.time() - start_time
                completion_data = {
                    "event": "complete",
                    "processing_time": round(processing_time, 2),
                    "original_query": user_query,
                    "search_query": search_query
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            elif intent == "COMPLEMENT":
                # COMPLEMENT FLOW: Generate multiple search queries for complementary items
                if image_content:
                    complement_queries_json = await image_processor.generate_search_query_for_intent(user_query, image_content, "COMPLEMENT", chat_history_list)
                else:
                    # Text-only complement - let LLM generate complementary items
                    complement_queries_json = await image_processor.generate_search_query_for_intent(user_query, None, "COMPLEMENT", chat_history_list)
                
                try:
                    complement_data = json.loads(complement_queries_json)
                    complement_queries = complement_data.get("complementary_searches", ["complementary items"])
                except Exception as e:
                    print(f"Error parsing complement queries: {e}")
                    print(f"Full traceback: {traceback.format_exc()}")
                    complement_queries = ["complementary items"]
                
                print(f"COMPLEMENT queries: {complement_queries}")
                
                # Send complement queries event
                yield f"data: {json.dumps({'event': 'complement_queries', 'queries': complement_queries})}\n\n"
                
                # Search for each complementary item and stream results
                for i, query in enumerate(complement_queries[:3]):  # Limit to 3 queries
                    try:
                        async for result in chatbot_flow(
                            intent_analysis=query,
                            conversation_id=chat_uuid,
                            conversation_redis_client=redis_client,
                            user_preferences=user_preferences,
                            chat_history=chat_history_list
                        ):
                            event_type = result.get("event", "")
                            if event_type in ["rag_search", "filter_search"]:
                                data = result.get("data", {})
                                if isinstance(data, dict) and "search_results" in data:
                                    search_results = data["search_results"]
                                    products = []
                                    
                                    for search_result in search_results:
                                        try:
                                            product = {
                                                "title": search_result.get("title", f"Product {search_result['product_id']}"),
                                                "image_url": search_result.get("image_url", ""),
                                                "product_url": search_result.get("product_url", f"/product/{search_result['product_id']}"),
                                                "max_price": float(search_result.get("max_price", 0)),
                                                "relevance_score": search_result.get("relevance_score"),
                                                "llm_filter_reason": search_result.get("text_description", ""),
                                                "body_text": search_result.get("body_text", "")
                                            }
                                            products.append(product)
                                        except Exception as e:
                                            print(f"Error processing complement search result {search_result.get('product_id', 'unknown')}: {e}")
                                            print(f"Full traceback: {traceback.format_exc()}")
                                    
                                    # Stream complement section
                                    if products:
                                        stream_data = {
                                            "event": "complement_section",
                                            "data": {
                                                "section_type": "RECOMMENDATION",
                                                "title": query.title(),
                                                "description": f"Perfect {query.lower()} to complement your style",
                                                "products": products[:5],  # Limit to 5 products per section
                                                "total_results": len(products),
                                                "show_more_available": len(products) > 5
                                            }
                                        }
                                        yield f"data: {json.dumps(stream_data)}\n\n"
                        
                    except Exception as e:
                        print(f"Error processing complement query '{query}': {e}")
                        print(f"Full traceback: {traceback.format_exc()}")
                        continue
                
                # Send completion event
                processing_time = time.time() - start_time
                completion_data = {
                    "event": "complete",
                    "processing_time": round(processing_time, 2),
                    "original_query": user_query,
                    "detected_items": complement_queries
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            elif intent == "GENERAL":
                # Import general_agent from groq_utils
                from groq_utils import general_agent
                
                llm_response = await general_agent(user_query)
                processing_time = time.time() - start_time
                
                general_data = {
                    "event": "general_message",
                    "data": llm_response
                }
                yield f"data: {json.dumps(general_data)}\n\n"
                
                # Send completion event
                completion_data = {
                    "event": "complete",
                    "processing_time": round(processing_time, 2),
                    "original_query": user_query
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            elif intent == "AMBIGUOUS":
                llm_response = await intent_classifier.handle_ambiguous_intent(user_query, "")
                processing_time = time.time() - start_time
                
                ambiguous_data = {
                    "event": "ambiguous_message",
                    "data": llm_response.get("content", "I'd be happy to help with your fashion needs!")
                }
                yield f"data: {json.dumps(ambiguous_data)}\n\n"
                
                # Send completion
                yield f"data: {json.dumps({'event': 'complete'})}\n\n"
                
        except Exception as e:
            print(f"Error in unified_chat_handler: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            from logger_config import log_detailed_error
            processing_time = time.time() - start_time
            log_detailed_error(
                e,
                context="unified_chat_handler.generate_sse_stream",
                local_vars={
                    "chat_uuid": locals().get("chat_uuid"),
                    "user_query": locals().get("user_query"),
                    "intent": locals().get("intent"),
                    "processing_time": processing_time
                }
            )
            
            error_data = {
                "event": "error",
                "error": f"Internal server error: {str(e)}",
                "processing_time": round(processing_time, 2)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_sse_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

async def get_available_budget_options() -> List[Dict]:
    """Get available budget options from Redis."""
    try:
        budget_data = await redis_client.get("budget_options")
        budget_options = json.loads(budget_data) if budget_data else []
        print(f"📋 Available budget options: {budget_options}")
        return budget_options
        
    except Exception as e:
        print(f"⚠️ Error getting budget options from Redis: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        # Fallback to default budget options
        return [
            {"label": "Under ₹1000", "max_price": 1000, "min_price": 0},
            {"label": "Under ₹2500", "max_price": 2500, "min_price": 0},
            {"label": "Under ₹4000", "max_price": 4000, "min_price": 0},
            {"label": "Under ₹6000", "max_price": 6000, "min_price": 0},
            {"label": "No limit", "max_price": None, "min_price": 0},
        ]

@app.get("/api/filters/options")
async def get_filter_options():
    """Get available gender and budget options for frontend."""
    try:
        # Get gender options from Redis
        gender_options = await get_available_gender_options()
        
        # Get budget options from Redis
        budget_options = await get_available_budget_options()
        
        return {
            "gender_options": gender_options,
            "budget_options": budget_options
        }
        
    except Exception as e:
        print(f"❌ Error getting available options: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return {
            "gender_options": ["female", "male", "unisex"],
            "budget_options": [
                {"label": "Under ₹1000", "max_price": 1000, "min_price": 0},
                {"label": "Under ₹2500", "max_price": 2500, "min_price": 0},
                {"label": "Under ₹4000", "max_price": 4000, "min_price": 0},
                {"label": "Under ₹6000", "max_price": 6000, "min_price": 0},
                {"label": "No limit", "max_price": None, "min_price": 0},
            ]
        }

@app.get("/")
def hello():
    return "hello"

def read_root():
    return {"message": "Welcome to the E-commerce Chatbot Intent API"}

# To run this app:
# 1. Make sure you have a .env file with your GROQ_API_KEY and Redis details.
# 2. Run in your terminal: uvicorn main:app --reload 