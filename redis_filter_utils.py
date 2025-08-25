"""
Redis filter utilities for product search
Provides functions to create size and price filters for Redis queries
"""

def create_size_filter(sizes: list) -> str:
    """
    Create Redis filter for size options
    
    Args:
        sizes: List of sizes to filter by (e.g., ['s', 'm', 'l'])
        
    Returns:
        str: Redis filter string for sizes
    """
    if not sizes:
        return ""
    
    size_query_parts = [f'@size_options:{{{size}}}' for size in sizes]
    return f"({' | '.join(size_query_parts)})"


def create_dynamic_price_filter(user_min_budget: float, user_max_budget: float, max_price_count: int = 6) -> str:
    """
    Create OR filter across all dynamic price fields
    
    Args:
        user_min_budget: Minimum budget range
        user_max_budget: Maximum budget range
        max_price_count: Number of price fields to check (default 6 for price1-price6)
        
    Returns:
        str: Redis filter string for price range
    """
    if not user_min_budget and not user_max_budget:
        return ""
    
    # Use reasonable defaults if one bound is missing
    min_price = user_min_budget if user_min_budget else 0
    max_price = user_max_budget if user_max_budget else 50000  # High default max
    
    price_conditions = []
    
    for i in range(1, max_price_count + 1):
        price_conditions.append(f"@price{i}:[{min_price} {max_price}]")
    
    if price_conditions:
        return f"({' | '.join(price_conditions)})"
    return ""


def create_product_type_filter(allowed_product_types: list) -> str:
    """
    Create Redis filter for product types
    
    Args:
        allowed_product_types: List of product types to filter by
        
    Returns:
        str: Redis filter string for product types
    """
    if not allowed_product_types:
        return "*"  # No filter, match all
    
    if len(allowed_product_types) == 1:
        # For TAG fields, use curly braces for exact match
        return f"@product_type:{{{allowed_product_types[0]}}}"
    else:
        # For multiple types with TAG field
        filters = [f"@product_type:{{{pt}}}" for pt in allowed_product_types]
        return "(" + " | ".join(filters) + ")"


def create_exclude_filter(excluded_ids: list) -> str:
    """
    Create Redis filter to exclude specific product IDs
    
    Args:
        excluded_ids: List of product IDs to exclude
        
    Returns:
        str: Redis filter string to exclude products
    """
    if not excluded_ids:
        return ""
    
    return " ".join([f"-@product_id:{pid}" for pid in excluded_ids])


def build_combined_filter(
    allowed_product_types: list = None,
    sizes: list = None,
    user_min_budget: float = None,
    user_max_budget: float = None,
    excluded_ids: list = None,
    max_price_count: int = 6
) -> str:
    """
    Build a combined Redis filter string with all specified filters
    
    Args:
        allowed_product_types: List of product types to include
        sizes: List of sizes to filter by
        user_min_budget: Minimum price range
        user_max_budget: Maximum price range
        excluded_ids: List of product IDs to exclude
        max_price_count: Number of dynamic price fields to check
        
    Returns:
        str: Combined Redis filter string
    """
    filters = []
    
    # Add product type filter
    product_type_filter = create_product_type_filter(allowed_product_types)
    if product_type_filter and product_type_filter != "*":
        filters.append(product_type_filter)
    
    # Add size filter
    size_filter = create_size_filter(sizes)
    if size_filter:
        filters.append(size_filter)
    
    # Add price filter
    price_filter = create_dynamic_price_filter(user_min_budget, user_max_budget, max_price_count)
    if price_filter:
        filters.append(price_filter)
    
    # Add exclusion filter
    exclude_filter = create_exclude_filter(excluded_ids)
    if exclude_filter:
        filters.append(exclude_filter)
    
    # Combine all filters
    if not filters:
        return "*"  # No filters, match all
    
    # Join with AND logic (space separator)
    return " ".join(filters)