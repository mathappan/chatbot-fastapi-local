import json
import os
from collections import defaultdict
from config import STORE_NAME
from typing import Dict, List, Any
from logger_config import logger, log_error, error_handler

class DataLoader:
    """Handles loading of all product and attribute data files."""
    
    def __init__(self):
        # Initialize attributes that will be set dynamically via setattr()
        self.group_descriptions = {}
        self.grouped_values = {}
        self.text_descriptions = {}
        self.standardised_jsons = {}
        self.cleaned_results_with_type = []
        self.product_list = []
        self.store_name = STORE_NAME
        # Get the directory of this file
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        logger.info(f"DataLoader initialized. Data directory: {self.data_dir}")
        self._load_all_data()
    
    @error_handler("Data Loading")
    def _load_all_data(self):
        """Load all required data files."""
        try:
            logger.info("Starting data loading process...")
            
            # Define file mappings
            files_to_load = {
                'group_descriptions': f'{self.store_name}_group_descriptions_deepseek.json',
                'grouped_values': f'{self.store_name}_attributes_and_candidates_deepseek.json',
                'text_descriptions': f'{self.store_name}_rich_text_descriptions.json',
                'standardised_jsons': f'{self.store_name}_json_descriptons_with_price_size_deepseek.json',
                'cleaned_results_with_type': f'{self.store_name}_cleaned_products_with_product_category.json',
                'product_list': f'{self.store_name}_products_list.json'
            }
            
            # Fallback to original file if price_size version doesn't exist
            standardised_jsons_file_path = os.path.join(self.data_dir, f'{self.store_name}_json_descriptons_with_price_size_deepseek.json')
            if not os.path.exists(standardised_jsons_file_path):
                logger.warning(f"Price/size enhanced file not found: {standardised_jsons_file_path}")
                logger.info("Falling back to original json_descriptons file")
                files_to_load['standardised_jsons'] = f'{self.store_name}_json_descriptons_deepseek.json'
            
            # Load each file
            for attr_name, filename in files_to_load.items():
                file_path = os.path.join(self.data_dir, filename)
                logger.info(f"Loading {filename} from {file_path}")
                
                if not os.path.exists(file_path):
                    error_msg = f"File not found: {file_path}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        setattr(self, attr_name, data)
                        logger.info(f"Successfully loaded {filename} - {len(data) if isinstance(data, (list, dict)) else 'N/A'} items")
                except json.JSONDecodeError as e:
                    log_error(e, f"JSON decode error in {filename}", {"file_path": file_path})
                    raise
                except Exception as e:
                    log_error(e, f"Error loading {filename}", {"file_path": file_path})
                    raise
            
            # Create product titles mapping
            self.product_titles = {str(product["id"]): product["title"] for product in self.product_list}
            
            # Create product image mapping
            self.product_image_map = {str(p['id']): p['first_image_url'] for p in self.product_list}
            
            # Create size options mapping by product type
            self.sizes_by_product_type = self._extract_sizes_by_product_type()
            
            logger.info(f"Data loading completed successfully. Created {len(self.product_titles)} product title mappings, {len(self.product_image_map)} image mappings, and {len(self.sizes_by_product_type)} product type size mappings.")
            
        except Exception as e:
            log_error(e, "Critical error during data loading")
            raise
        
        # Transform attribute structure
        self.attribute_descriptions = self._transform_attribute_structure(self.group_descriptions)
        self.reversed_attribute_mappings = self._reverse_all_product_type_mappings(self.attribute_descriptions)
        
        # Create product categories list
        self.product_categories = list(self.reversed_attribute_mappings.keys())
    
    def _transform_attribute_structure(self, original_data: Dict) -> Dict:
        """Transform the original attribute structure."""
        transformed = {}
        for product_type, groups in original_data.items():
            transformed[product_type] = {}
            for group in groups.values():
                attr_name = group.get('group_name')
                attr_description = group.get('group_description')
                if attr_name and attr_description:
                    transformed[product_type][attr_name] = attr_description
        return transformed
    
    def _reverse_all_product_type_mappings(self, transformed_data: Dict) -> Dict:
        """Reverse the attribute mappings."""
        reversed_nested = {}
        for product_type, attr_map in transformed_data.items():
            reversed_nested[product_type] = {
                desc: attr for attr, desc in attr_map.items()
            }
        return reversed_nested
    
    def _extract_sizes_by_product_type(self) -> Dict[str, List[str]]:
        """
        Extract unique size options for each product type.
        
        Returns:
            Dict mapping product_type to sorted list of unique sizes
        """
        # Initialize dictionary to collect sizes by product_type
        sizes_by_type = defaultdict(set)
        
        # Loop through each product
        for product in self.cleaned_results_with_type:
            product_id = str(product['id'])
            product_type = product.get('product_type')
            
            # Skip if product_type is missing
            if not product_type:
                continue
            
            # Get size options from standardised_jsons
            size_options = self.standardised_jsons.get(product_id, {}).get('size_options')
            if size_options:
                # Convert sizes to lowercase and update the set for the product_type
                sizes_by_type[product_type].update(size.lower() for size in size_options if size)
        
        # Convert sets to sorted lists
        unique_sizes_by_type = {ptype: sorted(list(sizes)) for ptype, sizes in sizes_by_type.items()}
        print("sizes in data_loader - ", unique_sizes_by_type)
        logger.info(f"Extracted sizes for {len(unique_sizes_by_type)} product types")
        for ptype, sizes in unique_sizes_by_type.items():
            logger.info(f"  {ptype}: {len(sizes)} unique sizes - {sizes[:5]}{'...' if len(sizes) > 5 else ''}")
        
        return unique_sizes_by_type

# Global data loader instance
data_loader = DataLoader()