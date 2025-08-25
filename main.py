import asyncio
import os
import json
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field, ValidationError
from groq import AsyncGroq
from dotenv import load_dotenv
import redis.asyncio as redis
import redis as sync_redis
from contextlib import asynccontextmanager
from dotenv import load_dotenv, find_dotenv
from prompts import (
    INTENT_DERIVATION_SYSTEM_PROMPT,
    FASHION_SALESPERSON_SYSTEM_PROMPT,
    USER_SUMMARY_SYSTEM_PROMPT,
    GENERAL_AGENT_PROMPT,
    DETAIL_COLLECTION_PROMPT,
    SEARCH_QUERY_CREATION_PROMPT
)
from groq_utils import chatbot_flow
from rag_search_and_pitch import rag_search_and_pitch
from logger_config import logger, log_error, error_handler, log_user_interaction
from image_utils import image_processor
from intent_classifier import intent_classifier
from response_models import (
    UnifiedResponse, ResponseType, ResponseStatus, SectionType, Section,
    ResponseData, ResponseMetadata, ProductSearchResult
)

# --- Environment and Configuration ---
load_dotenv(find_dotenv('.env.txt'))

# --- Redis Client Configuration ---
redis_client = redis.Redis(
    host='redis-19985.c212.ap-south-1-1.ec2.redns.redis-cloud.com',
    port=19985,
    decode_responses=True,
    username="default",
    password="9sctWbao6E8VydXA2KlBTp1CTcOjAlko",
    socket_connect_timeout=5,
    socket_timeout=5,
    socket_keepalive=True,
    socket_keepalive_options={},
    max_connections=10,
)

# --- System Prompts imported from prompts.py ---

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    chat_uuid: str = Field(..., description="Unique identifier for the chat session.", example="610f22f2-b015-459d-b912-471c94b6edf9")
    messages: List[ChatMessage] = Field(..., description="The complete chat history.")
    transcribed_text: Optional[str] = Field(None, description="Transcribed text from audio input (if any)")

class UserPreferences(BaseModel):
    preferences: Dict[str, Any] = {}
    gender: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    size: Optional[List[str]] = None


class AnalysisResponse(BaseModel):
    analysis: str
    user_summary: Dict[str, Any]
    detail_collection_response: str = None

class FashionResponse(BaseModel):
    response: str

app = FastAPI()

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
    Logs the request body and returns a detailed error response to help with debugging.
    """
    body = await request.body()
    print("--- VALIDATION ERROR (422) ---")
    print("Request Body Received:", body.decode())
    print("Validation Errors:", exc.errors())
    print("------------------------------")
    
    # Return the received body in the response for easier client-side debugging
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "received_body": body.decode()},
    )

# --- Helper Functions ---

def get_latest_user_message(chat_history: List[Dict[str, str]]) -> str:
    for msg in reversed(chat_history):
        if msg.get("role") == "customer":
            return msg.get("content", "")
    return ""


async def get_user_summary_from_redis(redis_key: str) -> Dict[str, Any]:
    """Retrieves user summary from Redis."""
    summary_json = await redis_client.get(redis_key)
    if summary_json:
        return json.loads(summary_json)
    return {"preferences": {}, "gender": None, "price_min": None, "price_max": None, "size": []}

async def store_user_summary_in_redis(redis_key: str, summary: Dict[str, Any]) -> None:
    """Stores user summary in Redis with TTL."""
    await redis_client.set(redis_key, json.dumps(summary), ex=3600)  # 1 hour TTL

def merge_user_summaries(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merges new user summary data with existing summary."""
    merged = existing.copy()
    
    # Merge preferences
    if new.get("preferences"):
        merged["preferences"].update(new["preferences"])
    
    # Update gender, price_min, price_max, size if new values are provided
    for key in ["gender", "price_min", "price_max", "size"]:
        if new.get(key) is not None:
            merged[key] = new[key]
    
    return merged

async def reset_user_summary(chat_uuid: str) -> None:
    """Resets user summary in Redis."""
    redis_key = f"user_summary:{chat_uuid}"
    await redis_client.delete(redis_key)

def get_missing_user_details(user_summary: Dict[str, Any]) -> List[str]:
    """Checks user summary and returns list of missing critical details."""
    missing_details = []
    
    if not user_summary.get("size"):
        missing_details.append("size")
    
    if not user_summary.get("price_min") and not user_summary.get("price_max"):
        missing_details.append("budget")
    
    if not user_summary.get("gender"):
        missing_details.append("gender")
    
    return missing_details

async def derive_intent_with_groq(chat_history: List[Dict[str, str]], user_summary: Dict[str, Any] = None) -> str:
    print("--- Deriving Intent with Groq LLM ---")
    try:
        groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Build system prompt with user summary context if available
        system_content = INTENT_DERIVATION_SYSTEM_PROMPT
        if user_summary and any(user_summary.values()):
            user_summary_text = json.dumps(user_summary, indent=2)
            system_content += f'''\n\nEXISTING USER SUMMARY:\n{user_summary_text}\n\nUse this existing user summary as context when analyzing the conversation.
            Do not suggest products to user.'''
        
        # Get last 5 messages and format as requested
        last_messages = chat_history[-5:]
        
        # Format chat history as: customer: ..., bot: ..., customer: ..., bot: ...
        formatted_history = []
        latest_customer_message = ""
        
        for msg in last_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "customer":
                formatted_history.append(f"customer: {content}")
                latest_customer_message = content
            elif role == "assistant":
                formatted_history.append(f"salesperson: {content}")
        
        # Join the formatted history and add the latest customer message
        chat_history_text = ", ".join(formatted_history)
        if latest_customer_message:
            chat_history_text += f", latest_customer_message: {latest_customer_message}"
            chat_history_text += f"whats the intent behind the latest message from the customer"
        print("chat_history - ", json.dumps(chat_history, indent=4))
        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": chat_history_text}
            ],
            temperature=0,
        )
        
        analysis = response.choices[0].message.content
        print("--- LLM Response Received for Intent---", analysis )
        return analysis

    except Exception as e:
        print(f"Error calling Groq API: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {e}")

async def generate_user_summary_with_groq(chat_history: List[Dict[str, str]], existing_user_summary: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generates user summary from chat history using Groq with JSON validation and retry logic."""
    print("--- Generating User Summary with Groq ---")
    
    groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Convert last 5 messages and handle 'customer' role
    last_messages = chat_history[-5:]
    chat_content = "\n".join([
        f"{'Customer' if msg['role'] == 'customer' else msg['role'].capitalize()}: {msg['content']}" 
        for msg in last_messages
    ])
    chat_content += '''Return only a JSON object with this structure:

                                    {
                                    "preferences": {},
                                    "gender": null,
                                    "price_min": null,
                                    "price_max": null,
                                    "size": null
                                    }
                                    '''
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
            
            # Validate structure using UserPreferences Pydantic model
            validated_summary = UserPreferences(**parsed_summary)
            
            print("--- User Summary Generated and Validated ---", json.dumps(validated_summary.model_dump(), indent=4))
            return validated_summary.model_dump()
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Returning default summary.")
                return {"preferences": {}, "gender": None, "price_min": None, "price_max": None, "size": []}
            continue
            
        except ValidationError as e:
            print(f"Pydantic validation error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Returning default summary.")
                return {"preferences": {}, "gender": None, "price_min": None, "price_max": None, "size": []}
            continue
            
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Returning default summary.")
                return {"preferences": {}, "gender": None, "price_min": None, "price_max": None, "size": []}
            continue
    
    # Fallback (should not reach here)
    return {"preferences": {}, "gender": None, "price_min": None, "price_max": None, "size": []}

async def get_fashion_response_with_groq(chat_history: List[Dict[str, str]]) -> str:
    """Generates a conversational response from the fashion salesperson LLM."""
    print("--- Getting Fashion Response with Groq LLM ---")
    try:
        groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

        # Get last 5 messages and format as JSON string
        last_messages = chat_history[-5:]
        messages_json = json.dumps(last_messages, indent=2)
        
        response = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": FASHION_SALESPERSON_SYSTEM_PROMPT},
                {"role": "user", "content": messages_json}
            ],
            temperature=0.7,  # Higher temperature for more creative/natural conversation
        )
        
        assistant_response = response.choices[0].message.content
        print("--- LLM Fashion Response Received ---")
        return assistant_response

    except Exception as e:
        print(f"Error calling Groq API for fashion response: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {e}")

async def get_detail_collection_response_with_groq(missing_detail: str, chat_history: List[Dict[str, str]]) -> str:
    """Generates a gentle prompt to collect missing user details."""
    print(f"--- Getting Detail Collection Response for {missing_detail} ---")
    try:
        groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

        # Add specific missing detail context to the system prompt
        system_content = DETAIL_COLLECTION_PROMPT + f"\n\nMISSING DETAIL: {missing_detail}\n\nAsk for this specific detail in a natural, helpful way."
        print("MISSING_DETAIL FOR GROQ - ", missing_detail)
        
        # Get last 5 messages and format as JSON string
        last_messages = chat_history[-5:]
        messages_json = json.dumps(last_messages, indent=2)
        
        response = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": messages_json}
            ],
            temperature=0.5,  # Moderate temperature for natural but consistent responses
        )
        
        detail_response = response.choices[0].message.content
        print('detail_response - ', detail_response)
        print("--- Detail Collection Response Received ---")
        return detail_response

    except Exception as e:
        print(f"Error calling Groq API for detail collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {e}")

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
        groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        
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
        print(f"Error transcribing audio with Groq: {e}")
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {e}")

# --- FastAPI Endpoints ---

# POST endpoint: Voice input with streaming response
@app.post("/chat/voice/stream")
@error_handler("Voice Chat Stream Endpoint")
async def voice_chat_stream(
    chat_uuid: str = Form(...),
    chat_history: str = Form(...),  # JSON string of chat history
    audio_file: UploadFile = File(...)
):
    """
    Voice endpoint that:
    1. Transcribes audio to text using Groq Speech-to-Text
    2. Adds transcribed message to chat history
    3. Uses shared chat processing logic for streaming response
    """
    try:
        # Parse chat history
        try:
            chat_history_list = json.loads(chat_history)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid chat_history JSON format")
        
        # Validate and transcribe audio file
        await validate_audio_file(audio_file)
        transcribed_text = await transcribe_audio_with_groq(audio_file)
        logger.info(f"Audio transcribed - Session: {chat_uuid}, Text: {transcribed_text[:100]}")
        
        # Add the transcribed message to chat history
        chat_history_list.append({
            "role": "customer",
            "content": transcribed_text
        })
        
        # Store the updated chat history in Redis
        messages_key = f"chat_messages:{chat_uuid}"
        await redis_client.set(messages_key, json.dumps(chat_history_list), ex=3600)
        
        # Stream response using shared chat logic
        async def event_generator():
            try:
                # First, send the transcribed text to client
                yield ServerSentEvent(
                    data=json.dumps({"transcribed_text": transcribed_text}),
                    event="transcription"
                )
                
                # Then stream the chat response
                async for result in process_chat_logic(chat_uuid):
                    yield ServerSentEvent(
                        data=result["data"],
                        event=result["event"]
                    )
            except Exception as exc:
                log_error(
                    exc,
                    f"Voice chat stream error - Session: {chat_uuid}",
                    {"session_id": chat_uuid, "transcribed_text": transcribed_text[:200]}
                )
                yield ServerSentEvent(
                    data=json.dumps({"error": "An error occurred while processing your voice message."}),
                    event="error"
                )

        return EventSourceResponse(event_generator())
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in voice chat stream: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing voice message: {str(e)}")

# POST endpoint: Voice input (non-streaming, for compatibility)
@app.post("/chat/voice")
@error_handler("Voice Chat POST Endpoint")
async def voice_chat_handler(
    chat_uuid: str = Form(...),
    chat_history: str = Form(...),  # JSON string of chat history
    audio_file: Optional[UploadFile] = File(None),
    text_message: Optional[str] = Form(None),
    ):
    """
    Legacy voice endpoint for compatibility.
    Handles both text and audio input, but only stores the message without processing.
    Use /chat/voice/stream for full voice processing with streaming response.
    """
    try:
        # Parse chat history
        try:
            chat_history_list = json.loads(chat_history)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid chat_history JSON format")
        
        # Determine the user message content
        user_message_content = ""
        
        if audio_file:
            # Validate and transcribe audio file
            await validate_audio_file(audio_file)
            user_message_content = await transcribe_audio_with_groq(audio_file)
            logger.info(f"Audio transcribed - Session: {chat_uuid}, Text: {user_message_content[:100]}")
        elif text_message:
            user_message_content = text_message
        else:
            raise HTTPException(status_code=400, detail="Either audio_file or text_message must be provided")
        
        # Add the new user message to chat history
        chat_history_list.append({
            "role": "customer",
            "content": user_message_content
        })
        
        # Store the updated chat history in Redis
        messages_key = f"chat_messages:{chat_uuid}"
        await redis_client.set(messages_key, json.dumps(chat_history_list), ex=3600)
        
        logger.info(f"Voice/Text message processed - Session: {chat_uuid}, Message: {user_message_content[:100]}")
        
        return {
            "message": "Message processed successfully",
            "session_id": chat_uuid,
            "transcribed_text": user_message_content if audio_file else None,
            "message_count": len(chat_history_list)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in voice chat handler: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

# POST endpoint: Store messages and validate input
@app.post("/chat/initiate")
@error_handler("Chat POST Endpoint")
async def store_chat_messages(payload: ChatPayload):
    """
    POST endpoint for storing chat messages.
    Accepts message history and stores it in Redis with session ID.
    Does NOT process or respond - that's handled by the GET SSE endpoint.
    """
    session_id = payload.chat_uuid
    chat_history = [msg.model_dump() for msg in payload.messages]
    
    # Get latest user message for logging
    latest_user_message = ""
    for msg in reversed(chat_history):
        if msg.get("role") == "customer":
            latest_user_message = msg.get("content", "")
            break
    
    logger.info(f"Chat messages stored - Session: {session_id}, Message: {latest_user_message[:100]}")
    
    # Store messages in Redis with session ID
    messages_key = f"chat_messages:{session_id}"
    await redis_client.set(messages_key, json.dumps(chat_history), ex=3600)  # 1 hour TTL
    
    return {
        "message": "Messages stored successfully",
        "session_id": session_id,
        "message_count": len(chat_history)
    }

async def process_chat_logic(session_id: str):
    """
    Shared chat processing logic that handles:
    - User detail collection
    - Intent derivation
    - Chatbot flow execution
    Returns an async generator for streaming responses.
    """
    # ─── 1) Retrieve stored history ─────────────────────────────────
    messages_key = f"chat_messages:{session_id}"
    stored_history = await redis_client.get(messages_key)
    if not stored_history:
        yield {"event": "error", "data": json.dumps({"error": "No messages found for this session"})}
        return

    chat_history = json.loads(stored_history)
    redis_key = f"chat_summary:{session_id}"

    latest_user_message = next(
        (m["content"] for m in reversed(chat_history) if m["role"] == "customer"),
        ""
    )
    logger.info(
        f"Chat processing started - Session: {session_id}, "
        f"Message: {latest_user_message[:100]}"
    )

    # ─── 2) Missing-detail Q&A short-circuit ───────────────────────
    existing_summary = await get_user_summary_from_redis(redis_key)
    missing_details = get_missing_user_details(existing_summary)

    if missing_details:
        updated_summary = await generate_user_summary_with_groq(
            chat_history, existing_summary
        )
        missing_details = get_missing_user_details(updated_summary)

        if missing_details:  # still missing → ask
            prompt_for = missing_details[0]
            detail_ask = await get_detail_collection_response_with_groq(
                prompt_for, chat_history
            )
            await store_user_summary_in_redis(redis_key, updated_summary)

            yield {"event": "message", "data": json.dumps({"response": detail_ask})}
            return  # wait for next user turn

    # ─── 3) Kick off summary update (parallel) ─────────────────────
    summary_task = asyncio.create_task(
        generate_user_summary_with_groq(chat_history, existing_summary)
    )

    # ─── 4) Derive intent, then **STREAM** chatbot_flow ────────────
    groq_intent = await derive_intent_with_groq(chat_history, existing_summary)

    # Get updated user preferences to pass to chatbot_flow
    updated_summary = await summary_task

    async for evt in chatbot_flow(groq_intent, session_id, redis_client, updated_summary):
        # evt = {"event": <str>, "data": <str>}
        yield {"event": evt.get("event", "message"), "data": json.dumps({"response": evt["data"]})}

    # ─── 5) Persist summary once all done ──────────────────────────
    await store_user_summary_in_redis(redis_key, updated_summary)

    # ─── 6) Log − done ─────────────────────────────────────────────
    log_user_interaction(
        session_id, latest_user_message,
        "completed"  # no need to truncate; we just label the flow
    )
    logger.info(f"Chat response sent - Session: {session_id}")

# GET SSE endpoint: Stream real-time responses
@app.get("/chat/stream")
@error_handler("Chat SSE Endpoint")
async def stream_chat_response(
    session_id: str,
    ):
    """
    Streams real-time chat responses using the shared chat processing logic.
    """
    async def event_generator():
        try:
            async for result in process_chat_logic(session_id):
                yield ServerSentEvent(
                    data=result["data"],
                    event=result["event"]
                )
        except Exception as exc:
            log_error(
                exc,
                f"Chat SSE endpoint error - Session: {session_id}",
                {"session_id": session_id}
            )
            yield ServerSentEvent(
                data=json.dumps({"error": "An error occurred while processing your request."}),
                event="error"
            )

    return EventSourceResponse(event_generator())

@app.post("/chat/fashion", response_model=FashionResponse)
async def fashion_chat_handler(payload: ChatPayload):
    """
    Handles a conversational chat with the fashion salesperson AI.
    - Manages conversation history in Redis.
    - Generates a natural, conversational response from the AI.
    - Returns the AI's response.
    """
    session_id = payload.chat_uuid
    # Use a different key prefix to keep histories separate
    redis_key = f"fashion_chat_history:{session_id}"
    
    try:
        # Retrieve history
        stored_history = await redis_client.get(redis_key)
        chat_history = json.loads(stored_history) if stored_history else []

        # Append new messages from the payload
        for msg in payload.messages:
            # Avoid adding duplicate messages if client sends full history
            if msg.model_dump() not in chat_history:
                 chat_history.append(msg.model_dump())

        # Get the new response from the assistant
        assistant_response = await get_fashion_response_with_groq(chat_history)
        
        # Append the new assistant response to the history
        chat_history.append({"role": "assistant", "content": assistant_response})
        
        # Store the updated history
        await redis_client.set(redis_key, json.dumps(chat_history))

        return FashionResponse(response=assistant_response)

    except Exception as e:
        print(f"An unexpected error occurred in fashion chat handler: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/reset")
async def reset_chat_summary(payload: dict):
    """Reset user summary for a chat session."""
    chat_uuid = payload.get("chat_uuid")
    if not chat_uuid:
        raise HTTPException(status_code=400, detail="chat_uuid is required")
    
    await reset_user_summary(chat_uuid)
    return {"message": "User summary reset successfully"}

@app.post("/chat/unified", response_model=UnifiedResponse)
@error_handler("Unified Chat Endpoint")
async def unified_chat_handler(
    request: Request,
    chat_uuid: str = Form(...),
    image_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None), 
    text_message: Optional[str] = Form(None),
    chat_history: Optional[str] = Form(None),
    ):
    """
    Unified endpoint that handles voice, image, and text input.
    Returns UnifiedResponse with product recommendations using existing chatbot_flow.
    """
    import time
    start_time = time.time()
    
    try:
        # Validate that at least one input is provided
        if not image_file and not audio_file and not text_message:
            raise HTTPException(status_code=400, detail="At least one input (image, audio, or text) must be provided")
        
        # Parse chat history
        chat_history_list = []
        if chat_history:
            try:
                chat_history_list = json.loads(chat_history)
            except json.JSONDecodeError:
                chat_history_list = []
        
        # Process voice input if provided
        voice_transcription = ""
        if audio_file:
            await validate_audio_file(audio_file)
            voice_transcription = await transcribe_audio_with_groq(audio_file)
            logger.info(f"Audio transcribed: {voice_transcription[:100]}")
        
        # Create structured user query with clear input labels
        user_query_parts = []
        if text_message:
            user_query_parts.append(f"Text input: {text_message}")
        if voice_transcription:
            user_query_parts.append(f"Voice input: {voice_transcription}")
        
        user_query = " | ".join(user_query_parts)
        
        if not user_query:
            raise HTTPException(status_code=400, detail="No valid input to process")
        
        # We'll add to chat history after processing image (if any)
        # This ensures we store the complete user intent
        
        # Generate user summary using existing function
        redis_key = f"chat_summary:{chat_uuid}"
        existing_summary = await get_user_summary_from_redis(redis_key)
        user_preferences = await generate_user_summary_with_groq(chat_history_list, existing_summary)
        await store_user_summary_in_redis(redis_key, user_preferences)
        
        # Determine intent from text/voice only (not image)
        intent_result = await intent_classifier.classify_user_intent(user_query, has_image=bool(image_file))
        intent = intent_result.get("intent", "COMPLEMENT")
        
        print(f"Determined intent: {intent}")
        
        # Store original user query in chat history
        chat_history_list.append({
            "role": "customer", 
            "content": user_query
        })
        
        # Store updated chat history
        messages_key = f"chat_messages:{chat_uuid}"
        await redis_client.set(messages_key, json.dumps(chat_history_list), ex=3600)
        
        # Process based on intent - two separate flows
        if intent == "SEARCH":
            # SEARCH FLOW: Generate single search query from text + optional image
            if image_file:
                image_content = await image_processor.validate_image_file(image_file)
                search_query = await image_processor.generate_search_query_for_intent(user_query, image_content, "SEARCH")
            else:
                # Text-only search - let LLM optimize the query
                search_query = await image_processor.generate_search_query_for_intent(user_query, None, "SEARCH")
            
            print(f"SEARCH query: {search_query}")
            
            # Pass single search query to chatbot_flow
            collected_results = []
            final_products = []
            
            async for result in chatbot_flow(
                intent_analysis=search_query,
                conversation_id=chat_uuid,
                conversation_redis_client=redis_client,
                user_preferences=user_preferences
            ):
                collected_results.append(result)
            
            # Extract products from results
            for result in collected_results:
                event_type = result.get("event", "")
                if event_type in ["rag_search", "filter_search"]:
                    data = result.get("data", {})
                    if isinstance(data, dict) and "pitches" in data:
                        pitches = data["pitches"]
                        for product_id, pitch_data in pitches.items():
                            try:
                                product = ProductSearchResult(
                                    title=pitch_data.get("title", f"Product {product_id}"),
                                    image_url=pitch_data.get("first_image_url", ""),
                                    product_url=pitch_data.get("product_url", f"/product/{product_id}"),
                                    max_price=float(pitch_data.get("max_price", 0)),
                                    store_name=pitch_data.get("store_name", "Libas"),
                                    relevance_score=pitch_data.get("relevance_score"),
                                    llm_filter_reason=pitch_data.get("explanation")
                                )
                                final_products.append(product)
                            except Exception as e:
                                print(f"Error processing product {product_id}: {e}")
            
            processing_time = time.time() - start_time
            
            # Create search results section
            sections = []
            if final_products:
                sections.append(Section(
                    section_type=SectionType.SEARCH_RESULTS,
                    title="Search Results",
                    description=f"Found {len(final_products)} items matching your search",
                    products=final_products,
                    total_results=len(final_products),
                    show_more_available=False
                ))
            
            return UnifiedResponse(
                response_type=ResponseType.SEARCH,
                status=ResponseStatus.SUCCESS if final_products else ResponseStatus.ERROR,
                data=ResponseData(sections=sections, clarification_options=[]),
                metadata=ResponseMetadata(
                    processing_time_seconds=round(processing_time, 2),
                    original_query=user_query,
                    search_query=search_query,
                    detected_items=[]
                ),
                error=None if final_products else "No products found matching your search."
            )
        
        elif intent == "COMPLEMENT":
            # COMPLEMENT FLOW: Generate multiple search queries for complementary items
            if image_file:
                image_content = await image_processor.validate_image_file(image_file)
                complement_queries_json = await image_processor.generate_search_query_for_intent(user_query, image_content, "COMPLEMENT")
            else:
                # Text-only complement - let LLM generate complementary items
                complement_queries_json = await image_processor.generate_search_query_for_intent(user_query, None, "COMPLEMENT")
            
            try:
                complement_data = json.loads(complement_queries_json)
                complement_queries = complement_data.get("complementary_searches", ["complementary items"])
            except:
                complement_queries = ["complementary items"]
            
            print(f"COMPLEMENT queries: {complement_queries}")
            
            # Search for each complementary item using chatbot_flow
            sections = []
            all_detected_items = []
            
            for i, query in enumerate(complement_queries[:3]):  # Limit to 3 queries
                try:
                    item_products = []
                    async for result in chatbot_flow(
                        intent_analysis=query,
                        conversation_id=f"{chat_uuid}_complement_{i}",
                        conversation_redis_client=redis_client,
                        user_preferences=user_preferences
                    ):
                        event_type = result.get("event", "")
                        if event_type in ["rag_search", "filter_search"]:
                            data = result.get("data", {})
                            if isinstance(data, dict) and "pitches" in data:
                                pitches = data["pitches"]
                                for product_id, pitch_data in pitches.items():
                                    try:
                                        product = ProductSearchResult(
                                            title=pitch_data.get("title", f"Product {product_id}"),
                                            image_url=pitch_data.get("first_image_url", ""),
                                            product_url=pitch_data.get("product_url", f"/product/{product_id}"),
                                            max_price=float(pitch_data.get("max_price", 0)),
                                            store_name=pitch_data.get("store_name", "Libas"),
                                            relevance_score=pitch_data.get("relevance_score"),
                                            llm_filter_reason=pitch_data.get("explanation")
                                        )
                                        item_products.append(product)
                                    except Exception as e:
                                        print(f"Error processing complement product {product_id}: {e}")
                    
                    # Create section for this complementary search
                    if item_products:
                        section = Section(
                            section_type=SectionType.RECOMMENDATION,
                            title=query.title(),
                            description=f"Perfect {query.lower()} to complement your style",
                            products=item_products[:5],  # Limit to 5 products per section
                            total_results=len(item_products),
                            show_more_available=len(item_products) > 5
                        )
                        sections.append(section)
                        all_detected_items.append(query)
                        
                except Exception as e:
                    print(f"Error processing complement query '{query}': {e}")
                    continue
            
            processing_time = time.time() - start_time
            
            return UnifiedResponse(
                response_type=ResponseType.COMPLEMENT,
                status=ResponseStatus.SUCCESS if sections else ResponseStatus.ERROR,
                data=ResponseData(sections=sections, clarification_options=[]),
                metadata=ResponseMetadata(
                    processing_time_seconds=round(processing_time, 2),
                    original_query=user_query,
                    detected_items=all_detected_items
                ),
                error=None if sections else "No complementary items found."
            )
        
        elif intent == "AMBIGUOUS":
            llm_response = await intent_classifier.handle_ambiguous_intent(user_query, "")
            processing_time = time.time() - start_time
            
            return UnifiedResponse(
                response_type=ResponseType.GENERAL,
                status=ResponseStatus.SUCCESS,
                data=ResponseData(
                    sections=[],
                    clarification_options=[],
                    message=llm_response.get("content", "I'd be happy to help with your fashion needs!")
                ),
                metadata=ResponseMetadata(
                    processing_time_seconds=round(processing_time, 2),
                    original_query=user_query,
                    detected_items=[]
                )
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in unified chat handler: {e}")
        processing_time = time.time() - start_time
        
        return UnifiedResponse(
            response_type=ResponseType.GENERAL,
            status=ResponseStatus.ERROR,
            data=ResponseData(sections=[], clarification_options=[]),
            metadata=ResponseMetadata(
                processing_time_seconds=round(processing_time, 2),
                original_query=text_message or voice_transcription or "unknown",
                detected_items=[]
            ),
            error=f"Internal server error: {str(e)}"
        )


@app.get("/")
def hello():
    return "hello"

def read_root():
    return {"message": "Welcome to the E-commerce Chatbot Intent API"}

# To run this app:
# 1. Make sure you have a .env file with your GROQ_API_KEY and Redis details.
# 2. Run in your terminal: uvicorn main:app --reload 