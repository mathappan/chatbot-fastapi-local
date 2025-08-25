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

# --- Environment and Configuration ---
load_dotenv(find_dotenv('.env.txt'))

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

# --- Redis Connection ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_user = os.getenv("REDIS_USERNAME")
    redis_pass = os.getenv("REDIS_PASSWORD")

    if redis_user and redis_pass:
        redis_url = f"redis://{redis_user}:{redis_pass}@{redis_host}:{redis_port}"
    else:
        redis_url = f"redis://{redis_host}:{redis_port}"

    app.state.conversation_redis = redis.from_url(redis_url, decode_responses=True)
    yield
    await app.state.conversation_redis.close()

app = FastAPI(lifespan=lifespan)

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


async def get_user_summary_from_redis(conversation_redis_client: redis.Redis, redis_key: str) -> Dict[str, Any]:
    """Retrieves user summary from Redis."""
    summary_json = await conversation_redis_client.get(redis_key)
    if summary_json:
        return json.loads(summary_json)
    return {"preferences": {}, "gender": None, "price_min": None, "price_max": None, "size": []}

async def store_user_summary_in_redis(conversation_redis_client: redis.Redis, redis_key: str, summary: Dict[str, Any]) -> None:
    """Stores user summary in Redis with TTL."""
    await conversation_redis_client.set(redis_key, json.dumps(summary), ex=3600)  # 1 hour TTL

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

async def reset_user_summary(conversation_redis_client: redis.Redis, chat_uuid: str) -> None:
    """Resets user summary in Redis."""
    redis_key = f"user_summary:{chat_uuid}"
    await conversation_redis_client.delete(redis_key)

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

# POST endpoint: Handle both text and audio input
@app.post("/chat/voice")
@error_handler("Voice Chat POST Endpoint")
async def voice_chat_handler(
    chat_uuid: str = Form(...),
    chat_history: str = Form(...),  # JSON string of chat history
    audio_file: Optional[UploadFile] = File(None),
    text_message: Optional[str] = Form(None),
    conversation_redis_client: redis.Redis = Depends(lambda: app.state.conversation_redis)
):
    """
    Handles both text and audio input for chat.
    - If audio_file is provided, transcribes it to text using Groq Speech-to-Text
    - If text_message is provided, uses it directly
    - Processes the message through the existing chatbot pipeline
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
        await conversation_redis_client.set(messages_key, json.dumps(chat_history_list), ex=3600)
        
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
async def store_chat_messages(payload: ChatPayload, conversation_redis_client: redis.Redis = Depends(lambda: app.state.conversation_redis)):
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
    await conversation_redis_client.set(messages_key, json.dumps(chat_history), ex=3600)  # 1 hour TTL
    
    return {
        "message": "Messages stored successfully",
        "session_id": session_id,
        "message_count": len(chat_history)
    }

# GET SSE endpoint: Stream real-time responses
@app.get("/chat/stream")
@error_handler("Chat SSE Endpoint")
async def stream_chat_response(
    session_id: str,
    conversation_redis_client: redis.Redis = Depends(lambda: app.state.conversation_redis)
):
    """
    Streams real-time chat responses.  
    • If user details are missing, asks for them and returns.  
    • Otherwise runs chatbot_flow **as an async generator** and forwards
      each chunk to the client the instant it is produced.
    """

    async def event_generator():
        try:
            # ─── 1) Retrieve stored history ─────────────────────────────────
            messages_key = f"chat_messages:{session_id}"
            stored_history = await conversation_redis_client.get(messages_key)
            if not stored_history:
                yield ServerSentEvent(
                    data=json.dumps({"error": "No messages found for this session"}),
                    event="error",
                )
                return

            chat_history = json.loads(stored_history)
            redis_key    = f"chat_summary:{session_id}"

            latest_user_message = next(
                (m["content"] for m in reversed(chat_history) if m["role"] == "customer"),
                ""
            )
            logger.info(
                f"Chat processing started - Session: {session_id}, "
                f"Message: {latest_user_message[:100]}"
            )

            # ─── 2) Missing-detail Q&A short-circuit ───────────────────────
            existing_summary = await get_user_summary_from_redis(conversation_redis_client, redis_key)
            missing_details  = get_missing_user_details(existing_summary)

            if missing_details:
                updated_summary = await generate_user_summary_with_groq(
                    chat_history, existing_summary
                )
                missing_details = get_missing_user_details(updated_summary)

                if missing_details:        # still missing → ask
                    prompt_for = missing_details[0]
                    detail_ask = await get_detail_collection_response_with_groq(
                        prompt_for, chat_history
                    )
                    await store_user_summary_in_redis(
                        conversation_redis_client, redis_key, updated_summary
                    )

                    yield ServerSentEvent(
                        data=json.dumps({"response": detail_ask}),
                        event="message"
                    )
                    return  # wait for next user turn

            # ─── 3) Kick off summary update (parallel) ─────────────────────
            summary_task = asyncio.create_task(
                generate_user_summary_with_groq(chat_history, existing_summary)
            )

            # ─── 4) Derive intent, then **STREAM** chatbot_flow ────────────
            groq_intent = await derive_intent_with_groq(chat_history, existing_summary)

            # Get updated user preferences to pass to chatbot_flow
            updated_summary = await summary_task

            async for evt in chatbot_flow(groq_intent, session_id, conversation_redis_client, updated_summary):
                # evt = {"event": <str>, "data": <str>}
                yield ServerSentEvent(
                    data=json.dumps({"response": evt["data"]}),
                    event=evt.get("event", "message")
                )

            # ─── 5) Persist summary once all done ──────────────────────────
            await store_user_summary_in_redis(conversation_redis_client, redis_key, updated_summary)

            # ─── 6) Log − done ─────────────────────────────────────────────
            log_user_interaction(
                session_id, latest_user_message,
                "completed"  # no need to truncate; we just label the flow
            )
            logger.info(f"Chat response sent - Session: {session_id}")

        except Exception as exc:
            log_error(
                exc,
                f"Chat SSE endpoint error - Session: {session_id}",
                {"session_id": session_id, "user_message": latest_user_message[:200]}
            )
            yield ServerSentEvent(
                data=json.dumps({"error": "An error occurred while processing your request."}),
                event="error"
            )

    return EventSourceResponse(event_generator())


@app.post("/chat/fashion", response_model=FashionResponse)
async def fashion_chat_handler(payload: ChatPayload, conversation_redis_client: redis.Redis = Depends(lambda: app.state.conversation_redis)):
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
        stored_history = await conversation_redis_client.get(redis_key)
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
        await conversation_redis_client.set(redis_key, json.dumps(chat_history))

        return FashionResponse(response=assistant_response)

    except Exception as e:
        print(f"An unexpected error occurred in fashion chat handler: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/reset")
async def reset_chat_summary(payload: dict, conversation_redis_client: redis.Redis = Depends(lambda: app.state.conversation_redis)):
    """Reset user summary for a chat session."""
    chat_uuid = payload.get("chat_uuid")
    if not chat_uuid:
        raise HTTPException(status_code=400, detail="chat_uuid is required")
    
    await reset_user_summary(conversation_redis_client, chat_uuid)
    return {"message": "User summary reset successfully"}

@app.get("/")
def hello():
    return "hello"

def read_root():
    return {"message": "Welcome to the E-commerce Chatbot Intent API"}

# To run this app:
# 1. Make sure you have a .env file with your GROQ_API_KEY and Redis details.
# 2. Run in your terminal: uvicorn main:app --reload 