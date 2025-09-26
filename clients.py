import redis.asyncio as redis
import voyageai
from groq import AsyncGroq
from config import GROQ_API_KEY, VOYAGE_API_KEY, REDIS_URL

# Initialize clients
try:
    groq_client = AsyncGroq(api_key=GROQ_API_KEY)
except Exception as e:
    print("Error initializing Groq client. Make sure the GROQ_API_KEY environment variable is set.")
    print(f"Details: {e}")
    groq_client = None

# Initialize Voyage AI
voyageai.api_key = VOYAGE_API_KEY
voyageai_client = voyageai.AsyncClient()

# Initialize Redis client
from redis_client_manager import get_async_redis_client
redis_client = get_async_redis_client()