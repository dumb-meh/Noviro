"""
Application Configuration

Centralized configuration following 12-factor app principles.
All environment variables and settings belong here.
"""

import os
from typing import Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


# ======================== SETTINGS ========================

class Settings:
    """Application-wide settings from environment variables"""
    
    def __init__(self):
        # OpenAI Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        # Abacus AI Configuration
        self.ABACUS_API_KEY = os.getenv("ABACUS_API_KEY")
        self.ABACUS_DEPLOYMENT_ID = os.getenv("ABACUS_DEPLOYMENT_ID")
        self.ABACUS_CONVERSATION_TTL_DAYS = int(os.getenv("ABACUS_CONVERSATION_TTL_DAYS", "7"))
        
        # Redis Configuration
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.REDIS_DB = int(os.getenv("REDIS_DB", "0"))
        self.CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))
        
        # ChromaDB Configuration
        self.CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.CHROMA_EMBEDDING_MODEL = os.getenv("CHROMA_EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Application Configuration
        self.MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "15"))
        self.FOLLOWUP_DETECTION_WINDOW = int(os.getenv("FOLLOWUP_DETECTION_WINDOW", "5"))


# ======================== CHATBOT CONFIG ========================

class ModelConfig(BaseModel):
    """OpenAI model configuration for chatbot"""
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000


class GuardrailConfig(BaseModel):
    """Guardrail system configuration for e-commerce filtering"""
    enabled: bool = True
    temperature: float = 0.3
    
    guardrail_system_prompt: str = """You are a content classifier for an e-commerce platform. 
Your task is to determine if a user's query is related to e-commerce topics.

E-commerce topics include:
- Product searches, recommendations, or information
- Product specifications, features, or comparisons
- Pricing, discounts, promotions
- Order status, shipping, delivery
- Returns, refunds, exchanges
- Customer reviews and ratings
- Account management, wishlist
- Payment methods
- Store policies
- Customer support for shopping-related issues

Respond with ONLY a JSON object in this exact format:
{"is_ecommerce": true/false, "reason": "brief explanation"}"""

    rejection_message: str = """I'm an e-commerce assistant designed to help with shopping-related questions. 
I can assist you with:
- Finding and recommending products
- Product specifications and details
- Pricing and promotions
- Shipping and delivery information
- Order tracking
- Returns and refunds
- Customer reviews

How can I help you with your shopping today?"""


class ResponseConfig(BaseModel):
    """Response generation configuration for chatbot"""
    system_prompt: str = """You are a helpful e-commerce assistant. 
Your role is to provide accurate, friendly, and helpful responses to customer queries.

Guidelines:
- Use the provided context from our databases to answer questions accurately
- Be concise but informative
- If you don't have enough information, acknowledge it politely
- Maintain a professional yet friendly tone
- Always prioritize customer satisfaction
- For product recommendations, explain why you're suggesting them
- For pricing questions, be specific and mention any current promotions

Context from our databases:
{context}

User Query: {query}

Provide a helpful response:"""

    temperature: float = 0.7
    max_tokens: int = 800


class ChatbotConfig:
    """Chatbot configuration aggregator"""
    
    def __init__(self):
        self.model_config = ModelConfig()
        self.guardrail_config = GuardrailConfig()
        self.response_config = ResponseConfig()


# ======================== GLOBAL INSTANCES ========================

settings = Settings()
chatbot_config = ChatbotConfig()
