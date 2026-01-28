"""
Product Knowledge Manager

Handles CRUD operations for products in the ChromaDB vector database.
Includes semantic search, add, update, and delete operations.
"""

from typing import List, Dict, Optional, Any
from app.vectordb.manager import vector_db_manager
from .knowledge_schema import ProductKnowledge, SearchResult, OperationResponse


class ProductKnowledgeManager:
    """Manages product entities in the vector database"""
    
    def __init__(self):
        self.collection = vector_db_manager.get_products_collection()
    
    def flatten_metadata(self, metadata: dict) -> dict:
        """
        Flatten nested metadata for ChromaDB compatibility.
        Converts lists to comma-separated strings.
        """
        flattened = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                flattened[key] = ", ".join(str(v) for v in value)
            elif value is None:
                flattened[key] = ""
            else:
                flattened[key] = value
        return flattened
    
    def _create_searchable_text(self, product: ProductKnowledge) -> str:
        """
        Create searchable text representation for embeddings.
        Combines key product fields into a coherent text.
        """
        parts = []
        
        parts.append(f"Product: {product.name}")
        parts.append(f"Description: {product.description}")
        parts.append(f"Category: {product.category}")
        
        if product.subcategory:
            parts.append(f"Subcategory: {product.subcategory}")
        if product.type:
            parts.append(f"Type: {product.type}")
        if product.tags:
            parts.append(f"Tags: {', '.join(product.tags)}")
        if product.about:
            parts.append(f"About: {product.about}")
        
        parts.append(f"Price: ${product.price}")
        if product.discount > 0:
            parts.append(f"Discount: {product.discount}%")
        parts.append(f"Stock: {product.stock_quantity} units")
        parts.append(f"Available: {'Yes' if product.availability else 'No'}")
        
        return " | ".join(parts)
    
    def add_product(self, product: ProductKnowledge) -> OperationResponse:
        """
        Add a new product to the vector database.
        
        Args:
            product: ProductKnowledge object with all product details
            
        Returns:
            OperationResponse with success status and product_id
        """
        try:
            # Create searchable text for embedding
            searchable_text = self._create_searchable_text(product)
            
            # Prepare metadata
            product_dict = product.dict(exclude_none=True)
            flattened_metadata = self.flatten_metadata(product_dict)
            
            # Add to collection
            self.collection.add(
                documents=[searchable_text],
                metadatas=[flattened_metadata],
                ids=[product.product_id]
            )
            
            return OperationResponse(
                success=True,
                message="Product added successfully",
                data={"product_id": product.product_id}
            )
        except Exception as e:
            return OperationResponse(
                success=False,
                error=str(e)
            )
    
    def update_product(self, product_id: str, product: ProductKnowledge) -> OperationResponse:
        """
        Update an existing product in the vector database.
        
        Args:
            product_id: ID of the product to update
            product: ProductKnowledge object with updated details
            
        Returns:
            OperationResponse with success status
        """
        try:
            # Create searchable text for embedding
            searchable_text = self._create_searchable_text(product)
            
            # Prepare metadata
            product_dict = product.dict(exclude_none=True)
            flattened_metadata = self.flatten_metadata(product_dict)
            
            # Update in collection
            self.collection.update(
                ids=[product_id],
                documents=[searchable_text],
                metadatas=[flattened_metadata]
            )
            
            return OperationResponse(
                success=True,
                message="Product updated successfully",
                data={"product_id": product_id}
            )
        except Exception as e:
            return OperationResponse(
                success=False,
                error=str(e)
            )
    
    def delete_product(self, product_id: str) -> OperationResponse:
        """
        Delete a product from the vector database.
        
        Args:
            product_id: ID of the product to delete
            
        Returns:
            OperationResponse with success status
        """
        try:
            self.collection.delete(ids=[product_id])
            return OperationResponse(
                success=True,
                message="Product deleted successfully",
                data={"product_id": product_id}
            )
        except Exception as e:
            return OperationResponse(
                success=False,
                error=str(e)
            )
    
    def search_products(
        self, 
        query: str, 
        n_results: int = 5, 
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Search for products using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Number of results to return (default: 5)
            filters: Optional metadata filters (e.g., {"category": "Electronics"})
            
        Returns:
            List of SearchResult objects with product data and relevance scores
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters if filters else None
            )
            
            products = []
            if results['metadatas'] and len(results['metadatas']) > 0:
                for i, metadata in enumerate(results['metadatas'][0]):
                    product = SearchResult(
                        id=results['ids'][0][i] if results['ids'] else "",
                        data=metadata,
                        relevance_score=1 - results['distances'][0][i] if results['distances'] else 0.0
                    )
                    products.append(product)
            
            return products
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """
        Get a specific product by ID.
        
        Args:
            product_id: Product ID to retrieve
            
        Returns:
            Product data dictionary or None if not found
        """
        try:
            results = self.collection.get(ids=[product_id])
            if results['metadatas'] and len(results['metadatas']) > 0:
                return results['metadatas'][0]
            return None
        except Exception as e:
            print(f"Error getting product: {e}")
            return None
    
    def get_all_products(self, limit: int = 100) -> List[Dict]:
        """
        Get all products from the database.
        
        Args:
            limit: Maximum number of products to return (default: 100)
            
        Returns:
            List of product dictionaries
        """
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


# Global instance
product_knowledge_manager = ProductKnowledgeManager()
