"""
Service Knowledge Manager

Handles CRUD operations for services in the ChromaDB vector database.
Includes semantic search, add, update, and delete operations.
"""

from typing import List, Dict, Optional, Any
from app.vectordb.config import vector_db_manager
from .knowledge_schema import ServiceKnowledge, SearchResult, OperationResponse


class ServiceKnowledgeManager:
    """Manages service entities in the vector database"""
    
    def __init__(self):
        self.collection = vector_db_manager.get_services_collection()
    
    def flatten_metadata(self, metadata: dict) -> dict:
        """
        Flatten nested metadata for ChromaDB compatibility.
        Converts lists to comma-separated strings, datetime to ISO format.
        """
        flattened = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                flattened[key] = ", ".join(str(v) for v in value)
            elif value is None:
                flattened[key] = ""
            elif hasattr(value, 'isoformat'):  # datetime objects
                flattened[key] = value.isoformat()
            else:
                flattened[key] = value
        return flattened
    
    def _create_searchable_text(self, service: ServiceKnowledge) -> str:
        """
        Create searchable text representation for embeddings.
        Combines key service fields into a coherent text.
        """
        parts = []
        
        parts.append(f"Service: {service.name}")
        parts.append(f"Description: {service.description}")
        parts.append(f"Category: {service.category}")
        
        if service.service_type:
            parts.append(f"Type: {service.service_type}")
        if service.location:
            parts.append(f"Location: {service.location}")
        if service.tags:
            parts.append(f"Tags: {', '.join(service.tags)}")
        
        parts.append(f"Price: ${service.price}")
        parts.append(f"Duration: {service.duration} minutes")
        parts.append(f"Available slots: {service.total_slot}")
        parts.append(f"Available: {'Yes' if service.availability else 'No'}")
        
        return " | ".join(parts)
    
    def add_service(self, service: ServiceKnowledge) -> OperationResponse:
        """
        Add a new service to the vector database.
        
        Args:
            service: ServiceKnowledge object with all service details
            
        Returns:
            OperationResponse with success status and service_id
        """
        try:
            searchable_text = self._create_searchable_text(service)
            service_dict = service.dict(exclude_none=True)
            flattened_metadata = self.flatten_metadata(service_dict)
            
            self.collection.add(
                documents=[searchable_text],
                metadatas=[flattened_metadata],
                ids=[service.service_id]
            )
            
            return OperationResponse(
                success=True,
                message="Service added successfully",
                data={"service_id": service.service_id}
            )
        except Exception as e:
            return OperationResponse(
                success=False,
                error=str(e)
            )
    
    def update_service(self, service_id: str, service: ServiceKnowledge) -> OperationResponse:
        """
        Update an existing service in the vector database.
        
        Args:
            service_id: ID of the service to update
            service: ServiceKnowledge object with updated details
            
        Returns:
            OperationResponse with success status
        """
        try:
            searchable_text = self._create_searchable_text(service)
            service_dict = service.dict(exclude_none=True)
            flattened_metadata = self.flatten_metadata(service_dict)
            
            self.collection.update(
                ids=[service_id],
                documents=[searchable_text],
                metadatas=[flattened_metadata]
            )
            
            return OperationResponse(
                success=True,
                message="Service updated successfully",
                data={"service_id": service_id}
            )
        except Exception as e:
            return OperationResponse(
                success=False,
                error=str(e)
            )
    
    def delete_service(self, service_id: str) -> OperationResponse:
        """
        Delete a service from the vector database.
        
        Args:
            service_id: ID of the service to delete
            
        Returns:
            OperationResponse with success status
        """
        try:
            self.collection.delete(ids=[service_id])
            return OperationResponse(
                success=True,
                message="Service deleted successfully",
                data={"service_id": service_id}
            )
        except Exception as e:
            return OperationResponse(
                success=False,
                error=str(e)
            )
    
    def search_services(
        self, 
        query: str, 
        n_results: int = 5, 
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Search for services using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Number of results to return (default: 5)
            filters: Optional metadata filters (e.g., {"category": "Home Services"})
            
        Returns:
            List of SearchResult objects with service data and relevance scores
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters if filters else None
            )
            
            services = []
            if results['metadatas'] and len(results['metadatas']) > 0:
                for i, metadata in enumerate(results['metadatas'][0]):
                    service = SearchResult(
                        id=results['ids'][0][i] if results['ids'] else "",
                        data=metadata,
                        relevance_score=1 - results['distances'][0][i] if results['distances'] else 0.0
                    )
                    services.append(service)
            
            return services
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_service_by_id(self, service_id: str) -> Optional[Dict]:
        """
        Get a specific service by ID.
        
        Args:
            service_id: Service ID to retrieve
            
        Returns:
            Service data dictionary or None if not found
        """
        try:
            results = self.collection.get(ids=[service_id])
            if results['metadatas'] and len(results['metadatas']) > 0:
                return results['metadatas'][0]
            return None
        except Exception as e:
            print(f"Error getting service: {e}")
            return None
    
    def get_all_services(self, limit: int = 100) -> List[Dict]:
        """
        Get all services from the database.
        
        Args:
            limit: Maximum number of services to return (default: 100)
            
        Returns:
            List of service dictionaries
        """
        try:
            results = self.collection.get(limit=limit)
            services = []
            
            if results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    service = {
                        "id": results['ids'][i] if results['ids'] else None,
                        "data": metadata
                    }
                    services.append(service)
            
            return services
        except Exception as e:
            print(f"Error getting services: {e}")
            return []


# Global instance
service_knowledge_manager = ServiceKnowledgeManager()
