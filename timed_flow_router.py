import json
import time
import asyncio
import sys
from groq_utils import top_level_router_agent, sales_router_agent
from logger_config import logger

async def timed_chatbot_flow_to_search(intent_analysis: str):
    """
    Performs the flow routing up to the product search decision point and times the execution.
    
    Args:
        intent_analysis (str): The derived intent from the conversation
    
    Returns:
        dict: Contains timing info and decision result
    """
    start_time = time.time()
    
    # 1. Top-level routing based on intent
    intent_start = time.time()
    intent_decision = await top_level_router_agent(intent_analysis)
    intent_duration = time.time() - intent_start
    
    print(f"Intent decision: {intent_decision} (took {intent_duration:.3f}s)")
    
    result = {
        "intent_decision": intent_decision,
        "intent_routing_time": intent_duration,
        "sales_decision": None,
        "sales_routing_time": None,
        "total_time": None,
        "ready_for_product_search": False
    }

    if intent_decision == "general_query":
        result["total_time"] = time.time() - start_time
        result["ready_for_product_search"] = False
        
    elif intent_decision == "sales_query":
        # 2. For sales queries, use sales router agent to decide next action
        sales_start = time.time()
        sales_decision = await sales_router_agent(intent_analysis)
        sales_duration = time.time() - sales_start
        
        result["sales_decision"] = sales_decision
        result["sales_routing_time"] = sales_duration
        
        print(f"Sales decision: {sales_decision} (took {sales_duration:.3f}s)")
        
        if sales_decision.get("action") == "perform_product_search":
            result["ready_for_product_search"] = True
            print("✅ Ready for product search!")
        else:
            result["ready_for_product_search"] = False
            print("❌ Not ready for product search - needs more info")

    result["total_time"] = time.time() - start_time
    
    print(f"Total flow routing time: {result['total_time']:.3f}s")
    logger.info(f"TIMED FLOW ROUTING: Total: {result['total_time']:.3f}s, Intent: {intent_duration:.3f}s, Sales: {sales_duration if result['sales_routing_time'] else 'N/A'}")
    
    return result

async def main():
    """Main function to run the timed flow router with user input."""
    intent_analysis = input("Enter your intent analysis: ")
    print(f"Running timed flow router with intent: '{intent_analysis}'")
    print("-" * 60)
    
    result = await timed_chatbot_flow_to_search(intent_analysis)
    
    print("-" * 60)
    print("Final Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())