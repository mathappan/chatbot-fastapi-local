"""
Conversation Storage Module for Redis

This module handles storing conversation data including:
- Product IDs from RAG-based search
- Product IDs from filter-based search  
- Search metadata and timestamps
"""

import json
import time
from typing import Dict, List, Any
class ConversationStorage:
    """Handles storage of conversation data in Redis"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.conversation_ttl = 86400  # 24 hours
    
    def set_redis_client(self, redis_client):
        """Set the Redis client to use for conversation storage"""
        self.redis_client = redis_client
    
    async def store_search_results(self, conversation_id: str, search_type: str, product_ids: List[str], metadata: Dict[str, Any] = None):
        """
        Store search results for a conversation using Redis SET for product IDs
        
        Args:
            conversation_id: Unique identifier for the conversation
            search_type: 'rag' or 'filter' 
            product_ids: List of product IDs from the search
            metadata: Additional metadata (query, timestamp, etc.)
        """
        # Different keys for different data types
        product_ids_key = f"conversation:{conversation_id}:search:{search_type}:product_ids"
        metadata_key = f"conversation:{conversation_id}:search:{search_type}:metadata"
        
        print(f"Product IDs key: {product_ids_key}")
        print(f"Metadata key: {metadata_key}")
        
        # Store product IDs as Redis SET (auto-deduplication and appending)
        if product_ids:
            await self.redis_client.sadd(product_ids_key, *product_ids)
            await self.redis_client.expire(product_ids_key, self.conversation_ttl)
        
        # Store metadata as JSON
        metadata_data = {
            "search_type": search_type,
            "timestamp": int(time.time()),
            "metadata": metadata or {}
        }
        await self.redis_client.setex(metadata_key, self.conversation_ttl, json.dumps(metadata_data))
        
        # Also store in a conversation index
        conversation_key = f"conversation:{conversation_id}:searches"
        await self.redis_client.sadd(conversation_key, search_type)
        await self.redis_client.expire(conversation_key, self.conversation_ttl)
        
        print(f"Added {len(product_ids)} product IDs for {search_type} search in conversation {conversation_id}")
        
        # Retrieve and display the stored data for verification
        stored_product_ids = await self.redis_client.smembers(product_ids_key)
        stored_metadata = await self.redis_client.get(metadata_key)
        
        if stored_product_ids is not None:
            print(f"Retrieved product IDs from SET '{product_ids_key}': {list(stored_product_ids)}")
            print(f"Total unique product IDs: {len(stored_product_ids)}")
        
        if stored_metadata:
            metadata_data = json.loads(stored_metadata)
            print(f"Retrieved metadata from '{metadata_key}':")
            print(f"Search Type: {metadata_data.get('search_type', 'N/A')}")
            print(f"Timestamp: {metadata_data.get('timestamp', 'N/A')}")
            print(f"Metadata: {metadata_data.get('metadata', {})}")
        else:
            print(f"Failed to retrieve metadata from key '{metadata_key}'")
    
    async def get_search_results(self, conversation_id: str, search_type: str = None) -> Dict[str, Any]:
        """
        Retrieve search results for a conversation
        
        Args:
            conversation_id: Unique identifier for the conversation
            search_type: Specific search type to retrieve, or None for all
            
        Returns:
            Dict with search results
        """
        if search_type:
            product_ids_key = f"conversation:{conversation_id}:search:{search_type}:product_ids"
            metadata_key = f"conversation:{conversation_id}:search:{search_type}:metadata"
            
            # Get product IDs from SET
            product_ids = await self.redis_client.smembers(product_ids_key)
            product_ids_list = list(product_ids) if product_ids else []
            
            # Get metadata from JSON
            metadata_data = await self.redis_client.get(metadata_key)
            metadata_dict = json.loads(metadata_data) if metadata_data else {}
            
            if product_ids_list or metadata_dict:
                return {
                    search_type: {
                        "product_ids": product_ids_list,
                        **metadata_dict
                    }
                }
            return {}
        else:
            # Get all search types for this conversation
            conversation_key = f"conversation:{conversation_id}:searches"
            search_types = await self.redis_client.smembers(conversation_key)
            
            results = {}
            for stype in search_types:
                product_ids_key = f"conversation:{conversation_id}:search:{stype}:product_ids"
                metadata_key = f"conversation:{conversation_id}:search:{stype}:metadata"
                
                # Get product IDs from SET
                product_ids = await self.redis_client.smembers(product_ids_key)
                product_ids_list = list(product_ids) if product_ids else []
                
                # Get metadata from JSON
                metadata_data = await self.redis_client.get(metadata_key)
                metadata_dict = json.loads(metadata_data) if metadata_data else {}
                
                if product_ids_list or metadata_dict:
                    results[stype] = {
                        "product_ids": product_ids_list,
                        **metadata_dict
                    }
            
            return results
    
    async def get_all_product_ids(self, conversation_id: str) -> Dict[str, List[str]]:
        """
        Get all product IDs from both search types for a conversation
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Dict with 'rag' and 'filter' keys containing product ID lists
        """
        results = await self.get_search_results(conversation_id)
        
        product_ids = {
            'rag': [],
            'filter': []
        }
        
        for search_type, data in results.items():
            if search_type in product_ids:
                product_ids[search_type] = data.get('product_ids', [])
        
        return product_ids
    
    async def clear_conversation(self, conversation_id: str):
        """Clear all data for a conversation"""
        conversation_key = f"conversation:{conversation_id}:searches"
        search_types = await self.redis_client.smembers(conversation_key)
        
        # Delete individual search results (both product IDs and metadata)
        for search_type in search_types:
            product_ids_key = f"conversation:{conversation_id}:search:{search_type}:product_ids"
            metadata_key = f"conversation:{conversation_id}:search:{search_type}:metadata"
            await self.redis_client.delete(product_ids_key)
            await self.redis_client.delete(metadata_key)
        
        # Delete conversation index
        await self.redis_client.delete(conversation_key)
        
        print(f"Cleared all data for conversation {conversation_id}")

# Global instance - Redis client will be set dynamically
conversation_storage = ConversationStorage()