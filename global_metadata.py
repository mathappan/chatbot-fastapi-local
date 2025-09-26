"""
Global metadata access module - breaks circular import between main.py and search_engine.py

This module provides a way for search_engine.py to access the same metadata 
that main.py loads, without creating a circular import dependency.
"""

def get_global_metadata():
    """Get global metadata variables from main module without circular import."""
    try:
        # Import only when needed to avoid circular dependency at module level
        import main
        return main.group_descriptions, main.reversed_attribute_mappings, main.grouped_values
    except (ImportError, AttributeError):
        # Fallback to empty dicts if not available
        return {}, {}, {}