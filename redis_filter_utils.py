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


def create_price_filter(user_min_budget: float, user_max_budget: float) -> str:
    """
    Create price filter using max_price field
    
    Args:
        user_min_budget: Minimum budget range
        user_max_budget: Maximum budget range
        
    Returns:
        str: Redis filter string for price range
    """
    if not user_min_budget and not user_max_budget:
        return ""
    
    # Use reasonable defaults if one bound is missing
    min_price = user_min_budget if user_min_budget else 0
    max_price = user_max_budget if user_max_budget else 50000  # High default max
    
    return f"@max_price:[{min_price} {max_price}]"


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


def create_gender_filter(genders) -> str:
    """
    Create Redis filter for multiple genders including unisex products
    
    Args:
        genders: List of gender preferences or single gender string
        
    Returns:
        str: Redis filter string for gender
    """
    if not genders:
        return ""
    
    # Handle both list and string input for backward compatibility
    if isinstance(genders, str):
        genders = [genders]
    
    if not isinstance(genders, list) or len(genders) == 0:
        return ""
    
    # Create filter parts for each gender
    gender_parts = []
    for gender in genders:
        if gender and isinstance(gender, str):
            gender_parts.append(f"@gender:{{{gender}}}")
    
    # Always include unisex unless the user specifically only wants unisex
    if not (len(genders) == 1 and genders[0] == "unisex"):
        gender_parts.append("@gender:{unisex}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_parts = []
    for part in gender_parts:
        if part not in seen:
            seen.add(part)
            unique_parts.append(part)
    
    if len(unique_parts) == 1:
        return unique_parts[0]
    else:
        return f"({' | '.join(unique_parts)})"


def build_combined_filter(
    allowed_product_types: list = None,
    sizes: list = None,
    user_min_budget: float = None,
    user_max_budget: float = None,
    excluded_ids: list = None,
    gender = None
) -> str:
    """
    Build a combined Redis filter string with all specified filters
    
    Args:
        allowed_product_types: List of product types to include
        sizes: List of sizes to filter by
        user_min_budget: Minimum price range
        user_max_budget: Maximum price range
        excluded_ids: List of product IDs to exclude
        gender: Gender preference(s) - can be string or list (will include unisex automatically)
        
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
    
    # Add gender filter
    gender_filter = create_gender_filter(gender)
    if gender_filter:
        filters.append(gender_filter)
    
    # Add price filter
    price_filter = create_price_filter(user_min_budget, user_max_budget)
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