"""
Centralized Redis Client Manager

Provides a single source of truth for Redis connections with proper connection pooling,
async/sync support, and centralized configuration.
"""

import redis
import redis.asyncio as async_redis
from typing import Optional


class RedisClientManager:
    """Singleton Redis client manager with proper connection pooling"""
    
    _sync_client: Optional[redis.Redis] = None
    _async_client: Optional[async_redis.Redis] = None
    _connection_config = {
        'host': 'redis-19985.c212.ap-south-1-1.ec2.redns.redis-cloud.com',
        'port': 19985,
        'decode_responses': True,
        'username': "default",
        'password': "9sctWbao6E8VydXA2KlBTp1CTcOjAlko",
    }
    
    @classmethod
    def get_sync_client(cls) -> redis.Redis:
        """
        Get synchronous Redis client with connection pooling
        
        Returns:
            redis.Redis: Configured sync Redis client
        """
        if cls._sync_client is None:
            cls._sync_client = redis.Redis(**cls._connection_config)
            print("✅ Created sync Redis client with connection pooling")
        return cls._sync_client
    
    @classmethod
    def get_async_client(cls) -> async_redis.Redis:
        """
        Get asynchronous Redis client with connection pooling
        
        Returns:
            async_redis.Redis: Configured async Redis client
        """
        if cls._async_client is None:
            cls._async_client = async_redis.Redis(**cls._connection_config)
            print("✅ Created async Redis client with connection pooling")
        return cls._async_client
    
    @classmethod
    async def close_connections(cls):
        """Close all Redis connections properly"""
        if cls._async_client:
            await cls._async_client.close()
            cls._async_client = None
            print("🔒 Closed async Redis client")
            
        if cls._sync_client:
            cls._sync_client.close()
            cls._sync_client = None  
            print("🔒 Closed sync Redis client")
    
    @classmethod
    def get_connection_info(cls) -> dict:
        """Get Redis connection configuration (without password)"""
        config = cls._connection_config.copy()
        config['password'] = '***'  # Hide password in logs
        return config


# Convenience functions for common usage patterns
def get_redis_client() -> redis.Redis:
    """Get sync Redis client - most common usage"""
    return RedisClientManager.get_sync_client()


def get_async_redis_client() -> async_redis.Redis:
    """Get async Redis client for async operations"""
    return RedisClientManager.get_async_client()


# For vector embeddings (commonly used pattern)
def get_vector_redis_client() -> redis.Redis:
    """Get Redis client for vector operations (sync)"""
    return RedisClientManager.get_sync_client()