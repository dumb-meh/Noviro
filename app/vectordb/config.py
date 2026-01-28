"""
ChromaDB Vector Database Configuration

Sets up four separate ChromaDB collections for:
- Products
- Services
- Consultations
- Specialists

Each collection uses OpenAI embeddings for semantic search.
"""

import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()


class VectorDBManager:
    """Manages ChromaDB collections for the e-commerce platform"""
    
    def __init__(self):
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize OpenAI embedding function
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-small"  # Cost-effective, 1536 dimensions
        )
        
        # Initialize collections
        self._init_collections()
    
    def _init_collections(self):
        """Initialize all four collections"""
        # Products Collection
        self.products_collection = self.client.get_or_create_collection(
            name="products_index",
            embedding_function=self.embedding_function,
            metadata={"description": "Product catalog with names, descriptions, categories"}
        )
        
        # Services Collection
        self.services_collection = self.client.get_or_create_collection(
            name="services_index",
            embedding_function=self.embedding_function,
            metadata={"description": "Services offered by the platform"}
        )
        
        # Consultations Collection
        self.consultations_collection = self.client.get_or_create_collection(
            name="consultations_index",
            embedding_function=self.embedding_function,
            metadata={"description": "Available consultations"}
        )
        
        # Specialists Collection
        self.specialists_collection = self.client.get_or_create_collection(
            name="specialists_index",
            embedding_function=self.embedding_function,
            metadata={"description": "Specialist profiles and expertise"}
        )
    
    def get_products_collection(self):
        """Get the products collection"""
        return self.products_collection
    
    def get_services_collection(self):
        """Get the services collection"""
        return self.services_collection
    
    def get_consultations_collection(self):
        """Get the consultations collection"""
        return self.consultations_collection
    
    def get_specialists_collection(self):
        """Get the specialists collection"""
        return self.specialists_collection
    
    def reset_all_collections(self):
        """Reset all collections (use with caution!)"""
        self.client.delete_collection("products_index")
        self.client.delete_collection("services_index")
        self.client.delete_collection("consultations_index")
        self.client.delete_collection("specialists_index")
        self._init_collections()


# Global instance
vector_db_manager = VectorDBManager()
