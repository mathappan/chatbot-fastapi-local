import json
import asyncio
from typing import Dict, List, Any, Tuple
from collections import defaultdict, OrderedDict

from clients import groq_client, voyageai_client

from global_metadata import get_global_metadata

from prompts import (
    HIGH_LEVEL_USER_QUERY_EXTRACTION_PROMPT,
    SEARCH_ATTRIBUTE_SYSTEM_PROMPT,
    FILTER_RERANKED_ATTRIBUTES_SYSTEM_PROMPT,
    VALUE_SYSTEM_PROMPT
)

class SearchEngine:
    """Main search engine for product discovery."""
    
    def __init__(self):
        # Product categories will be loaded dynamically from Redis or global variables
        self.product_categories = []
    
    
    async def get_attributes_for_search(self, user_query: str, garment: str, gender: str) -> Dict:
        """Get search attributes for a specific garment type with retry logic."""
        user_query_message = f'''Gender = {gender}. User query - {user_query}. For this user query, you are going to search for {garment}. Suggest attributes and values specifically for {garment} that can be used as a filter in an ecommerce catalog along with the weights.
            Only include attributes that are clearly implied in the user query. Do not include attributes that the user hasn't mentioned.
            Give output in specified json format'''
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                completion = await groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": SEARCH_ATTRIBUTE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_query_message},
                        {"role": "assistant", "content": "```json"}
                    ],
                    response_format={"type": "json_object"}
                )
                
                json_result = json.loads(completion.choices[0].message.content)
                return {garment: json_result}
                
            except Exception as e:
                from logger_config import log_detailed_error
                last_error = e
                log_detailed_error(
                    e,
                    context=f"get_attributes_for_search.attempt_{attempt + 1}",
                    local_vars={
                        "garment": garment,
                        "gender": gender,
                        "user_query": user_query,
                        "attempt": attempt + 1,
                        "max_retries": max_retries
                    }
                )
                if attempt < max_retries - 1:
                    continue
        
        # If all retries failed, return a fallback response
        print(f"All {max_retries} attempts failed for garment '{garment}'. Using fallback response.")
        return {garment: {"attributes": []}}
    
    async def get_search_attributes(self, user_query: str, genders, allowed_product_types) -> Dict:
        """Get search attributes for all identified garment types."""
        garments = allowed_product_types
        print(f"Using garments: {garments}")
        
        # Use first gender for attribute extraction (attributes are generally not gender-specific)
        primary_gender = genders[0] if isinstance(genders, list) and genders else genders if isinstance(genders, str) else "female"
        tasks = [self.get_attributes_for_search(user_query, garment, primary_gender) for garment in garments]
        results = await asyncio.gather(*tasks)
        
        # Merge individual dicts into one
        merged_results = {}
        for result in results:
            merged_results.update(result)
        
        return merged_results
    
    async def rerank_description(self, attr_desc: str, documents: List[str], description_to_group: Dict) -> Tuple:
        """Rerank descriptions using Voyage AI."""
        reranking = await voyageai_client.rerank(attr_desc, documents, model="rerank-2-lite", top_k=3)
        return attr_desc, [
            {
                "group_name": description_to_group[r.document],
                "description": r.document,
                "relevance_score": r.relevance_score
            }
            for r in reranking.results
        ]
    
    async def rerank_all_for_product_type(self, product_type: str, attributes_for_search: Dict) -> Tuple:
        """Rerank all attributes for a product type."""
        group_descriptions, reversed_attribute_mappings, grouped_values = get_global_metadata()
        
        documents = [v["group_description"] for v in group_descriptions.get(product_type, {}).values()]
        description_to_group = reversed_attribute_mappings.get(product_type, {})
        attribute_descriptions = [attr['attribute_description'] for attr in attributes_for_search['attributes']]

        tasks = [
            self.rerank_description(desc, documents, description_to_group)
            for desc in attribute_descriptions
        ]
        results = await asyncio.gather(*tasks)
        return product_type, {desc: matches for desc, matches in results}
    
    async def map_attributes_to_existing(self, query_desc: str, reranked: List) -> Tuple:
        """Map attributes to existing catalog attributes."""
        gpt_input = {
            "query_attribute_description": query_desc,
            "reranked_attributes": [
                {
                    "attribute_name": match["group_name"],
                    "attribute_description": match["description"],
                    "sample_values_of_attribute": match["sample_values"]
                }
                for match in reranked
            ]
        }

        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": FILTER_RERANKED_ATTRIBUTES_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(gpt_input)}
            ],
            temperature=0
        )

        try:
            return query_desc, json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Failed to parse for '{query_desc}': {e}")
            return query_desc, []
    
    async def map_all_attributes_for_product_type(self, product_type: str, results_by_attribute: Dict) -> Tuple:
        """Map all attributes for a product type."""
        tasks = [
            self.map_attributes_to_existing(query_desc, reranked)
            for query_desc, reranked in results_by_attribute.items()
        ]
        output = await asyncio.gather(*tasks)
        return product_type, {query: matched for query, matched in output}
    
    async def run_attribute_search_for_all_product_types(self, product_attributes_for_search: Dict) -> Dict:
        """Run attribute search for all product types."""
        # Step 1: Rerank all product types
        rerank_tasks = [
            self.rerank_all_for_product_type(pt, attributes_for_search) 
            for pt, attributes_for_search in product_attributes_for_search.items()
        ]
        rerank_results = await asyncio.gather(*rerank_tasks)
        reranked_by_product_type = {pt: result for pt, result in rerank_results}

        # Add attribute values for better context
        group_descriptions, reversed_attribute_mappings, grouped_values = get_global_metadata()
        for product_type, attributes in reranked_by_product_type.items():
            grouped_attr_values = grouped_values.get(product_type, {})
            for attribute_name, reranked_list in attributes.items():
                for attr_entry in reranked_list:
                    attr_name = attr_entry["group_name"]
                    sample_values = grouped_attr_values.get(attr_name, [])[:5]
                    attr_entry["sample_values"] = sample_values
        
        # Step 2: Map all attributes via GPT
        map_tasks = [
            self.map_all_attributes_for_product_type(pt, results_by_attribute)
            for pt, results_by_attribute in reranked_by_product_type.items()
        ]
        mapped_results = await asyncio.gather(*map_tasks)
        return {pt: mapped for pt, mapped in mapped_results}
    
    async def process_single_value(
        self,
        query_value: str,
        product_type: str,
        candidate_attributes: List[str],
        product_specific_attributes: Dict,
    ) -> Tuple[str, List[str]]:
        """Match a single query value against canonical values with validation and retry."""
        try:
            # Gather candidate values from product type's attributes
            candidate_options = set()
            for attr_name in candidate_attributes:
                values = product_specific_attributes.get(attr_name, [])
                candidate_options.update(values)

            if not candidate_options:
                return query_value, {"values": []}
            
            candidate_options = list(candidate_options)

            # Rerank candidates
            reranked = await voyageai_client.rerank(
                query=query_value, documents=candidate_options, model="rerank-2-lite", top_k=5
            )

            top_matches = [
                r.document for r in reranked.results if r.relevance_score > 0.1
            ]
            
            if not top_matches:
                return query_value, {"values": []}

            # Use LLM for precise filtering with validation and retry
            gpt_input = {
                "query_value": query_value,
                "reranked_canonicals": [{"canonical_value": match} for match in top_matches]
            }

            max_retries = 3
            last_response = None
            
            # Store conversation history for retry attempts
            messages = [
                {"role": "system", "content": VALUE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(gpt_input)}
            ]

            for attempt in range(max_retries):
                try:
                    gpt_response = await groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    
                    content = gpt_response.choices[0].message.content
                    matched_values = json.loads(content)
                    last_response = matched_values
                    
                    # Validate the response format
                    if self._validate_value_mapping_response(matched_values):
                        # Return the full matched_values dict which contains {"values": [...]}
                        values_list = matched_values.get("values", [])
                        print(f"✅ Valid value mapping for '{query_value}': {values_list}")
                        return query_value, matched_values
                    else:
                        print(f"❌ Attempt {attempt + 1}: Invalid value mapping format for '{query_value}': {matched_values}")
                        
                        if attempt < max_retries - 1:
                            # Add feedback for retry
                            messages.append({
                                "role": "assistant",
                                "content": content
                            })
                            messages.append({
                                "role": "user",
                                "content": f"""Your response format is invalid. You must return a JSON object with a "values" key containing a list of strings.

Example correct format:
{{"values": ["string1", "string2"]}}

Your response contained: {matched_values}

Please provide a valid JSON response with only the "values" key containing a list of matching canonical values."""
                            })
                                
                except json.JSONDecodeError as e:
                    print(f"Attempt {attempt + 1}: JSON parsing error for value '{query_value}': {e}")
                    if attempt < max_retries - 1:
                        continue
                except Exception as e:
                    from logger_config import log_detailed_error
                    log_detailed_error(
                        e,
                        context=f"process_single_value.attempt_{attempt + 1}",
                        local_vars={
                            "query_value": query_value,
                            "product_type": product_type,
                            "candidate_attributes": candidate_attributes,
                            "attempt": attempt + 1,
                            "max_retries": max_retries
                        }
                    )
                    if attempt < max_retries - 1:
                        continue
            
            
            print(f"❌ All {max_retries} attempts failed for value '{query_value}', returning empty list")
            return query_value, {"values": []}

        except Exception as e:
            from logger_config import log_detailed_error
            log_detailed_error(
                e,
                context="process_single_value.outer_exception",
                local_vars={
                    "query_value": query_value,
                    "product_type": product_type,
                    "candidate_attributes": candidate_attributes,
                    "product_specific_attributes": str(product_specific_attributes)[:200]
                }
            )
            return query_value, {"values": []}
    
    def _validate_value_mapping_response(self, response: dict) -> bool:
        """Validate that the value mapping response has correct format."""
        if not isinstance(response, dict):
            return False
        
        if "values" not in response:
            return False
            
        values = response["values"]
        if not isinstance(values, list):
            return False
            
        # Check that all values in the list are strings
        return all(isinstance(v, str) for v in values)
    
    
    async def map_search_values_to_catalog(
        self,
        product_attributes_for_search: Dict,
        attribute_mapping: Dict,
    ) -> Dict:
        """Map search values to catalog values."""
        tasks = []
        context = []

        # Get global metadata
        group_descriptions, reversed_attribute_mappings, grouped_values = get_global_metadata()

        # Create tasks for each attribute lookup
        for product_type, config in product_attributes_for_search.items():
            attributes = config.get("attributes", [])
            for attr_spec in attributes:
                description = attr_spec["attribute_description"]
                possible_values = attr_spec.get("possible_values", [])
                
                mapped_catalog_attributes = attribute_mapping.get(product_type, {}).get(description)

                if not mapped_catalog_attributes or not possible_values:
                    continue

                for value in possible_values:
                    for mapped_attribute in mapped_catalog_attributes:
                        tasks.append(
                            self.process_single_value(
                                query_value=value,
                                product_type=product_type,
                                candidate_attributes=[mapped_attribute],
                                product_specific_attributes=grouped_values.get(product_type, {}),
                            )
                        )
                        context.append((product_type, description, value, mapped_attribute))

        # Execute all tasks
        results = await asyncio.gather(*tasks)

        # Reconstruct final dictionary
        final_value_mapping = defaultdict(
            lambda: defaultdict(lambda: {"value_mapping": defaultdict(dict)})
        )

        for i, (matched_query_value, canonicals) in enumerate(results):
            product_type, description, original_query_value, mapped_attribute = context[i]

            if original_query_value == matched_query_value and canonicals:
                all_mapped_attrs = attribute_mapping.get(product_type, {}).get(description, [])
                final_value_mapping[product_type][description]['mapped_attributes'] = all_mapped_attrs

                final_value_mapping[product_type][description]["value_mapping"][
                    original_query_value
                ][mapped_attribute] = canonicals

        return json.loads(json.dumps(final_value_mapping))

# Global search engine instance
search_engine = SearchEngine()