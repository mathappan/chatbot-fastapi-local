from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from enum import Enum


class ResponseType(str, Enum):
    """Response type enumeration."""
    COMPLEMENT = "COMPLEMENT"
    SEARCH = "SEARCH"
    AMBIGUOUS = "AMBIGUOUS"
    GENERAL = "GENERAL"


class ResponseStatus(str, Enum):
    """Response status enumeration."""
    SUCCESS = "success"
    CLARIFICATION_NEEDED = "clarification_needed"
    ERROR = "error"


class SectionType(str, Enum):
    """Section type enumeration."""
    RECOMMENDATION = "RECOMMENDATION"
    SEARCH_RESULTS = "SEARCH_RESULTS"
    CLARIFICATION = "CLARIFICATION"


class ProductSearchResult(BaseModel):
    """Product search result model."""
    title: str
    image_url: str
    product_url: str
    max_price: float
    store_name: str
    relevance_score: Optional[float] = None
    llm_filter_reason: Optional[str] = None


class ClarificationOption(BaseModel):
    """Clarification option model."""
    option_id: str
    display_text: str
    description: str


class Section(BaseModel):
    """Response section model."""
    section_type: SectionType
    title: str
    description: str
    products: List[ProductSearchResult] = []
    total_results: int = 0
    show_more_available: bool = False


class ResponseData(BaseModel):
    """Unified response data structure."""
    sections: List[Section] = []
    clarification_options: List[ClarificationOption] = []
    message: Optional[str] = None  # For general responses


class ResponseMetadata(BaseModel):
    """Response metadata."""
    processing_time_seconds: float
    original_query: Optional[str] = None
    detected_items: List[str] = []
    search_query: Optional[str] = None


class UnifiedResponse(BaseModel):
    """Unified response structure for all endpoint responses."""
    response_type: ResponseType
    status: ResponseStatus
    data: ResponseData
    metadata: ResponseMetadata
    error: Optional[str] = None


# Legacy models kept for backward compatibility
class ApparelRecommendation(BaseModel):
    """Response model for apparel recommendations."""
    complementary_items: List[str]


class RecommendationWithProducts(BaseModel):
    """Recommendation with associated product search results."""
    recommendation: str
    products: List[ProductSearchResult]
    total_results: int
    error: Optional[str] = None


class CompleteApparelResponse(BaseModel):
    """Complete response with recommendations and product searches."""
    complementary_items: List[str]
    product_searches: List[RecommendationWithProducts]
    processing_time_seconds: Optional[float] = None


class DirectSearchResponse(BaseModel):
    """Response model for direct product search (SEARCH intent)."""
    search_query: str
    original_query: str
    products: List[ProductSearchResult]
    total_results: int
    processing_time_seconds: Optional[float] = None
    error: Optional[str] = None


class AmbiguousResponse(BaseModel):
    """Response model for ambiguous intent requiring clarification."""
    message: str
    clarification_options: List[str]
    processing_time_seconds: Optional[float] = None


class GeneralFashionResponse(BaseModel):
    """Response model for general fashion assistance."""
    response: str
    processing_time_seconds: Optional[float] = None


class CombinedInput(BaseModel):
    """Input model for combined image and text."""
    text: str
    chat_uuid: str
    budget: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: str