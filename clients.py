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