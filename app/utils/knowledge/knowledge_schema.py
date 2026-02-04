from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# PRODUCT SCHEMA

class ProductKnowledge(BaseModel):
    """
    Product entity schema for vector database.
    
    Matches the product index schema with all required fields.
    """
    product_id: str = Field(..., description="Unique product identifier")
    name: str = Field(..., description="Product name")
    description: str = Field(..., description="Product description")
    price: float = Field(..., description="Product price")
    category: str = Field(..., description="Main category (one of 7 categories)")
    subcategory: Optional[str] = Field(None, description="Product subcategory")
    type: Optional[str] = Field(None, description="Product type")
    stock_quantity: int = Field(0, description="Available stock quantity")
    discount: float = Field(0.0, description="Discount percentage (0-100)")
    tags: List[str] = Field(default_factory=list, description="Product tags for search")
    about: Optional[str] = Field(None, description="Additional information about the product")



# SERVICE SCHEMA

class ServiceKnowledge(BaseModel):
    """
    Service entity schema for vector database.
    
    Represents services offered by the platform.
    """
    service_id: str = Field(..., description="Unique service identifier")
    name: str = Field(..., description="Service name")
    description: str = Field(..., description="Service description")
    price: float = Field(..., description="Service price")
    category: str = Field(..., description="Service category")
    duration: int = Field(..., description="Service duration in minutes")
    total_slot: int = Field(..., description="Total available slots")
    location: Optional[str] = Field(None, description="Service location")
    tags: List[str] = Field(default_factory=list, description="Service tags")
    start_time: Optional[datetime] = Field(None, description="Service start time")
    end_time: Optional[datetime] = Field(None, description="Service end time")
    service_type: Optional[str] = Field(None, description="Type of service")
    


# CONSULTATION SCHEMA

class ConsultationKnowledge(BaseModel):
    """
    Consultation entity schema for vector database.
    
    Represents consultation sessions available on the platform.
    """
    consultation_id: str = Field(..., description="Unique consultation identifier")
    name: str = Field(..., description="Consultant/professional name")
    description: str = Field(..., description="Consultation description")
    price: float = Field(..., description="Consultation price")
    category: str = Field(..., description="Consultation category")
    duration: int = Field(..., description="Consultation duration in minutes")
    title: Optional[str] = Field(None, description="Professional title")
    consultation_time: Optional[datetime] = Field(None, description="Available consultation time")
    experience: Optional[str] = Field(None, description="Years of experience or description")
    type: Optional[str] = Field(None, description="Type of consultation")

    

# SPECIALIST SCHEMA

class SpecialistKnowledge(BaseModel):
    """
    Specialist entity schema for vector database.
    
    Represents specialist profiles and their expertise.
    """
    specialist_id: str = Field(..., description="Unique specialist identifier")
    name: str = Field(..., description="Specialist name")
    description: str = Field(..., description="Specialist bio/description")
    price: float = Field(..., description="Specialist hourly rate or session price")
    category: str = Field(..., description="Specialist category/field")
    duration: int = Field(..., description="Typical session duration in minutes")
    title: Optional[str] = Field(None, description="Professional title/designation")
    specialist_time: Optional[datetime] = Field(None, description="Available times")
    experience: Optional[str] = Field(None, description="Years of experience")
    type: Optional[str] = Field(None, description="Type of specialist")
    rating: float = Field(0.0, description="Average rating (0-5)")


# RESPONSE SCHEMAS

class SearchResult(BaseModel):
    """Generic search result schema"""
    id: str
    data: dict
    relevance_score: float = Field(..., description="Similarity score (0-1)")


class OperationResponse(BaseModel):
    """Generic operation response"""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    data: Optional[dict] = None