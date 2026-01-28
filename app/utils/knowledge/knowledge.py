from typing import List, Dict, Optional, Any
from enum import Enum
import uuid
import json
from app.vectordb.config import vector_db
from .knowledge_schema import ProductKnowledge

class KnowledgeManager:
    def __init__(self):
        self.collection = vector_db.get_collection()
        
    
    def flatten_metadata(self,metadata: dict) -> dict:
        flattened = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                flattened[key] = ", ".join(str(v) for v in value)
            else:
                flattened[key] = value
        return flattened
    
    def add_product(self, product: ProductKnowledge) -> Dict[str, Any]:
        """Add a new product to the vector database using the provided productId"""
        try:
            product_id = product.productId
            
            searchable_text = self._create_searchable_text(product)
            
            product_dict = product.dict(exclude_none=True)
            flattened_metadata = self.flatten_metadata(product_dict) 
            self.collection.add(
                documents=[searchable_text],
                metadatas=[flattened_metadata],
                ids=[product_id]  
            )
            
            return {
                "success": True,
                "product_id": product_id,
                "message": "Product added successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_products(self, query: str, n_results: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search for products using vector similarity"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters if filters else None
            )
            
            products = []
            if results['metadatas'] and len(results['metadatas']) > 0:
                for i, metadata in enumerate(results['metadatas'][0]):
                    product = {
                        "id": results['ids'][0][i] if results['ids'] else None,
                        "data": metadata,
                        "relevance_score": 1 - results['distances'][0][i] if results['distances'] else 0
                    }
                    products.append(product)
            
            return products
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def update_product(self, product_id: str, product: ProductKnowledge) -> Dict[str, Any]:
        """Update an existing product"""
        try:
            searchable_text = self._create_searchable_text(product)
            product_dict = product.dict(exclude_none=True)
            flattened_metadata = self.flatten_metadata(product_dict) 
            
            self.collection.update(
                ids=[product_id],
                documents=[searchable_text],
                metadatas=[flattened_metadata]
            )
            
            return {
                "success": True,
                "message": "Product updated successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_product(self, product_id: str) -> Dict[str, Any]:
        """Delete a product from the database"""
        try:
            self.collection.delete(ids=[product_id])
            return {
                "success": True,
                "message": "Product deleted successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_all_products(self, limit: int = 100) -> List[Dict]:
        """Get all products from the database"""
        try:
            results = self.collection.get(limit=limit)
            products = []
            
            if results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    product = {
                        "id": results['ids'][i] if results['ids'] else None,
                        "data": metadata
                    }
                    products.append(product)
            
            return products
        except Exception as e:
            print(f"Error getting products: {e}")
            return []
    
    def _create_searchable_text(self, product: ProductKnowledge) -> str:
        """Create a searchable text representation of the product"""
        parts = []
        
        if product.productName:
            parts.append(f"Product: {product.productName}")
        if product.brand:
            parts.append(f"Brand: {product.brand}")
        if product.model:
            parts.append(f"Model: {product.model}")
        if product.type:
            parts.append(f"Type: {product.type}")
        if product.color:
            parts.append(f"Color: {product.color}")
        if product.description:
            parts.append(f"Description: {product.description}")
        if product.price:
            parts.append(f"Price: ${product.price}")
        if product.priceWithInstallation:
            parts.append(f"Price with installation: ${product.priceWithInstallation}")
        if product.condition:
            parts.append(f"Condition: {product.condition}")
        if product.warrantyType:
            parts.append(f"Warranty: {product.warrantyType}")
        if product.status:
            parts.append(f"Status: {product.status}")
        
        return " | ".join(parts)

knowledge_manager = KnowledgeManager()
