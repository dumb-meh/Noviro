"""
Configuration settings for the LangGraph E-commerce Chatbot.
This file contains all configurable parameters for the chatbot agent.
"""

import os
from typing import Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class ModelConfig(BaseModel):
    """OpenAI model configuration"""
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000


class GuardrailConfig(BaseModel):
    """Guardrail system configuration"""
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


class VectorDBConfig(BaseModel):
    """Vector Database configuration placeholders"""
    # These will be populated when connecting actual vector databases
    products_db: Dict[str, Any] = {
        "enabled": True,
        "index_name": "products_catalog",
        "description": "Main product catalog with names, descriptions, categories"
    }
    
    product_specs_db: Dict[str, Any] = {
        "enabled": True,
        "index_name": "product_specifications",
        "description": "Detailed technical specifications and features"
    }
    
    reviews_db: Dict[str, Any] = {
        "enabled": True,
        "index_name": "customer_reviews",
        "description": "Customer reviews, ratings, and feedback"
    }
    
    inventory_db: Dict[str, Any] = {
        "enabled": True,
        "index_name": "inventory_stock",
        "description": "Real-time stock availability and warehouse data"
    }
    
    pricing_db: Dict[str, Any] = {
        "enabled": True,
        "index_name": "pricing_promotions",
        "description": "Current prices, discounts, and promotional offers"
    }
    
    shipping_db: Dict[str, Any] = {
        "enabled": True,
        "index_name": "shipping_info",
        "description": "Shipping options, delivery times, and tracking"
    }
    
    support_db: Dict[str, Any] = {
        "enabled": True,
        "index_name": "support_docs",
        "description": "FAQs, policies, and customer support documentation"
    }


class ResponseConfig(BaseModel):
    """Response generation configuration"""
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
    """Main configuration class for the chatbot"""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_config = ModelConfig()
        self.guardrail_config = GuardrailConfig()
        self.vector_db_config = VectorDBConfig()
        self.response_config = ResponseConfig()
    
    def get_enabled_vector_dbs(self) -> list:
        """Get list of enabled vector databases"""
        dbs = []
        for attr_name in dir(self.vector_db_config):
            if attr_name.endswith('_db') and not attr_name.startswith('_'):
                db_config = getattr(self.vector_db_config, attr_name)
                if isinstance(db_config, dict) and db_config.get('enabled', False):
                    dbs.append({
                        'name': attr_name,
                        'index': db_config.get('index_name'),
                        'description': db_config.get('description')
                    })
        return dbs


# Global config instance
chatbot_config = ChatbotConfig()
