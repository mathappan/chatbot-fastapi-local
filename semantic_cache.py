import json
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import voyageai
from groq import AsyncGroq
import os

# Initialize clients
voyageai_client = voyageai.AsyncClient(api_key=os.getenv("VOYAGER_API_KEY"))
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

class DummyCache:
    """Simple in-memory cache for demonstration."""
    
    def __init__(self):
        self.queries: Dict[str, List[str]] = {}
        
    def add_query(self, cache_type: str, query: str):
        """Add a query to the cache."""
        if cache_type not in self.queries:
            self.queries[cache_type] = []
        
        if query not in self.queries[cache_type]:
            self.queries[cache_type].append(query)
            print(f"📦 Added to cache: '{query}' in {cache_type}")
    
    def get_queries(self, cache_type: str) -> List[str]:
        """Get all cached queries for a specific cache type."""
        return self.queries.get(cache_type, [])
    
    def get_all_cache_types(self) -> List[str]:
        """Get all available cache types."""
        return list(self.queries.keys())

# Global dummy cache instance
dummy_cache = DummyCache()

# LLM system prompt for cache result finalization
CACHE_SELECTION_PROMPT = """
You are an intelligent query matcher. Your job is to analyze a user's query and determine which cached query (if any) is the best semantic match from a list of reranked candidates.

You will be given:
1. The user's current query
2. A list of up to 10 reranked cached queries with their similarity scores

Your task:
1. Analyze semantic similarity between the user query and cached queries
2. Determine if any cached query is semantically equivalent (threshold: 75% relevance)
3. Select the BEST matching cached query, or return null if none are good enough

Be conservative - only return a match if you're confident the cached query represents the same search intent.

Return your response in JSON format:
{
  "best_match": "selected cached query" or null,
  "confidence_score": 0.0-1.0,
  "reasoning": "brief explanation of why this query matches or why no match was selected"
}
"""

async def semantic_cache_lookup(
    query: str, 
    cache_type: str, 
    rerank_top_k: int = 10,
    similarity_threshold: float = 0.7,
    llm_confidence_threshold: float = 0.75
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Find the best matching cached query using VoyageAI reranker + LLM finalization.
    
    Args:
        query: The user's search query
        cache_type: Type of cache to search in
        rerank_top_k: Number of top results to get from reranker
        similarity_threshold: Minimum similarity score from reranker
        llm_confidence_threshold: Minimum confidence score from LLM
        
    Returns:
        Tuple of (matched_query, timing_info) where timing_info contains detailed timings
    """
    
    # Initialize timing tracking
    start_time = time.time()
    timing_info = {
        "total_time": 0.0,
        "cache_retrieval_time": 0.0,
        "reranker_time": 0.0,
        "llm_time": 0.0,
        "preprocessing_time": 0.0
    }
    
    # Step 1: Get all cached queries for this cache type
    cache_start = time.time()
    cached_queries = dummy_cache.get_queries(cache_type)
    timing_info["cache_retrieval_time"] = time.time() - cache_start
    
    if len(cached_queries) < 1:
        timing_info["total_time"] = time.time() - start_time
        print(f"❌ No cached queries found for cache type: {cache_type}")
        print(f"⏱️  Total time: {timing_info['total_time']*1000:.1f}ms")
        return None, timing_info
    
    print(f"🔍 Searching through {len(cached_queries)} cached queries for '{query}'")
    
    # Step 2: Use VoyageAI reranker to find top similar queries
    try:
        rerank_start = time.time()
        reranking = await voyageai_client.rerank(
            query=query,
            documents=cached_queries,
            model="rerank-2-lite",
            top_k=min(rerank_top_k, len(cached_queries))
        )
        timing_info["reranker_time"] = time.time() - rerank_start
        
        # Filter by similarity threshold
        preprocess_start = time.time()
        candidates = [
            {"query": result.document, "score": result.relevance_score}
            for result in reranking.results 
            if result.relevance_score >= similarity_threshold
        ]
        timing_info["preprocessing_time"] = time.time() - preprocess_start
        
        if not candidates:
            timing_info["total_time"] = time.time() - start_time
            print(f"❌ No candidates above similarity threshold ({similarity_threshold})")
            print(f"⏱️  Reranker time: {timing_info['reranker_time']*1000:.1f}ms")
            print(f"⏱️  Total time: {timing_info['total_time']*1000:.1f}ms")
            return None, timing_info
            
        print(f"✅ Found {len(candidates)} candidates above threshold")
        for candidate in candidates:
            print(f"   - '{candidate['query']}' (score: {candidate['score']:.3f})")
        
    except Exception as e:
        timing_info["total_time"] = time.time() - start_time
        print(f"❌ Error in reranking: {e}")
        print(f"⏱️  Total time: {timing_info['total_time']*1000:.1f}ms")
        return None, timing_info
    
    # Step 3: Use LLM to make final selection
    llm_input = {
        "user_query": query,
        "cache_type": cache_type,
        "candidates": candidates
    }
    
    try:
        llm_start = time.time()
        response = await groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": CACHE_SELECTION_PROMPT},
                {"role": "user", "content": json.dumps(llm_input, indent=2)}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        timing_info["llm_time"] = time.time() - llm_start
        
        decision = json.loads(response.choices[0].message.content)
        
        print(f"🤖 LLM Decision: {decision}")
        
        # Calculate total time
        timing_info["total_time"] = time.time() - start_time
        
        # Step 4: Return result based on LLM decision
        if (decision.get("best_match") and 
            decision.get("confidence_score", 0) >= llm_confidence_threshold):
            
            best_match = decision.get("best_match")
            print(f"✅ Cache MATCH: '{query}' → '{best_match}' (confidence: {decision.get('confidence_score')})")
            print(f"⏱️  Timing breakdown:")
            print(f"   Cache retrieval: {timing_info['cache_retrieval_time']*1000:.1f}ms")
            print(f"   Reranker: {timing_info['reranker_time']*1000:.1f}ms")
            print(f"   LLM: {timing_info['llm_time']*1000:.1f}ms")
            print(f"   Total: {timing_info['total_time']*1000:.1f}ms")
            return best_match, timing_info
        
        print(f"❌ Cache MISS: LLM rejected candidates (confidence: {decision.get('confidence_score', 0)})")
        print(f"⏱️  Timing breakdown:")
        print(f"   Cache retrieval: {timing_info['cache_retrieval_time']*1000:.1f}ms")
        print(f"   Reranker: {timing_info['reranker_time']*1000:.1f}ms")
        print(f"   LLM: {timing_info['llm_time']*1000:.1f}ms")
        print(f"   Total: {timing_info['total_time']*1000:.1f}ms")
        return None, timing_info
        
    except Exception as e:
        timing_info["total_time"] = time.time() - start_time
        print(f"❌ Error in LLM finalization: {e}")
        print(f"⏱️  Total time: {timing_info['total_time']*1000:.1f}ms")
        return None, timing_info

async def add_to_cache(cache_type: str, query: str):
    """Add a new query to the cache."""
    dummy_cache.add_query(cache_type, query)

async def demo_semantic_cache():
    """Demonstrate the semantic cache functionality."""
    
    print("🚀 Starting Semantic Cache Demo")
    print("=" * 50)
    
    # Step 1: Populate cache with some queries
    print("\n📦 Populating cache with sample queries...")
    
    sample_queries = [
        "blue saree for party",
        "red dress for wedding", 
        "casual kurta for work",
        "silk saree traditional",
        "cotton dress summer",
        "black jeans casual",
        "white shirt formal",
        "ethnic wear festive"
    ]
    
    for query in sample_queries:
        await add_to_cache("product_search", query)
    
    # Step 2: Test semantic lookups
    print("\n🔍 Testing semantic cache lookups...")
    print("-" * 30)
    
    test_queries = [
        "blue saree party wear",          # Should match "blue saree for party"
        "red wedding dress",              # Should match "red dress for wedding"  
        "office kurta casual",            # Should match "casual kurta for work"
        "black leather jacket",           # Should not match anything
        "traditional silk saree",         # Should match "silk saree traditional"
        "summer cotton dress",            # Should match "cotton dress summer"
    ]
    
    for test_query in test_queries:
        print(f"\n🔍 Testing query: '{test_query}'")
        
        matched_query, timing_info = await semantic_cache_lookup(test_query, "product_search")
        
        if matched_query:
            print(f"✅ MATCHED: '{test_query}' → '{matched_query}'")
        else:
            print(f"❌ NO MATCH: '{test_query}' - would need fresh computation")
    
    print("\n" + "=" * 50)
    print("🎯 Demo completed!")
    
    # Show final cache state
    print(f"\n📊 Final cache state:")
    for cache_type in dummy_cache.get_all_cache_types():
        queries = dummy_cache.get_queries(cache_type)
        print(f"   {cache_type}: {len(queries)} queries")

# Simple function to test the cache lookup
async def test_cache_lookup(query: str, cache_type: str = "product_search") -> Tuple[Optional[str], Dict[str, float]]:
    """
    Simple function to test if a query matches anything in cache.
    
    Args:
        query: Query to search for
        cache_type: Type of cache to search in
        
    Returns:
        Tuple of (matched_query, timing_info)
    """
    return await semantic_cache_lookup(query, cache_type)

# Function to test multiple queries and show performance summary
async def benchmark_cache_performance(test_queries: List[str], cache_type: str = "product_search"):
    """
    Benchmark cache performance with multiple queries.
    
    Args:
        test_queries: List of queries to test
        cache_type: Type of cache to search in
    """
    print("🚀 Starting Cache Performance Benchmark")
    print("=" * 60)
    
    all_timings = []
    hit_count = 0
    miss_count = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📊 Test {i}/{len(test_queries)}: '{query}'")
        
        matched_query, timing_info = await semantic_cache_lookup(query, cache_type)
        all_timings.append(timing_info)
        
        if matched_query:
            hit_count += 1
            print(f"✅ HIT → '{matched_query}'")
        else:
            miss_count += 1
            print(f"❌ MISS")
    
    # Calculate summary statistics
    total_times = [t["total_time"] * 1000 for t in all_timings]  # Convert to ms
    reranker_times = [t["reranker_time"] * 1000 for t in all_timings]
    llm_times = [t["llm_time"] * 1000 for t in all_timings]
    
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total queries tested: {len(test_queries)}")
    print(f"Cache hits: {hit_count}")
    print(f"Cache misses: {miss_count}")
    print(f"Hit rate: {(hit_count / len(test_queries) * 100):.1f}%")
    print()
    print("⏱️  TIMING ANALYSIS:")
    print(f"Average total time: {sum(total_times) / len(total_times):.1f}ms")
    print(f"Min total time: {min(total_times):.1f}ms")
    print(f"Max total time: {max(total_times):.1f}ms")
    print()
    print(f"Average reranker time: {sum(reranker_times) / len(reranker_times):.1f}ms")
    print(f"Average LLM time: {sum(llm_times) / len(llm_times):.1f}ms")
    print()
    
    # Show time breakdown
    avg_cache_time = sum(t["cache_retrieval_time"] * 1000 for t in all_timings) / len(all_timings)
    avg_reranker_time = sum(reranker_times) / len(reranker_times)
    avg_llm_time = sum(llm_times) / len(llm_times)
    avg_preprocessing_time = sum(t["preprocessing_time"] * 1000 for t in all_timings) / len(all_timings)
    
    print("📊 AVERAGE TIME BREAKDOWN:")
    print(f"   Cache retrieval: {avg_cache_time:.1f}ms ({avg_cache_time / (sum(total_times) / len(total_times)) * 100:.1f}%)")
    print(f"   Reranker: {avg_reranker_time:.1f}ms ({avg_reranker_time / (sum(total_times) / len(total_times)) * 100:.1f}%)")
    print(f"   LLM: {avg_llm_time:.1f}ms ({avg_llm_time / (sum(total_times) / len(total_times)) * 100:.1f}%)")
    print(f"   Preprocessing: {avg_preprocessing_time:.1f}ms ({avg_preprocessing_time / (sum(total_times) / len(total_times)) * 100:.1f}%)")
    
    return {
        "hit_rate": hit_count / len(test_queries),
        "avg_total_time_ms": sum(total_times) / len(total_times),
        "avg_reranker_time_ms": avg_reranker_time,
        "avg_llm_time_ms": avg_llm_time,
        "all_timings": all_timings
    }

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_semantic_cache())