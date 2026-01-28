"""
Specialist Knowledge Manager

Handles CRUD operations for specialists in the ChromaDB vector database.
Includes semantic search, add, update, and delete operations.
"""

from typing import List, Dict, Optional, Any
from app.vectordb.manager import vector_db_manager
from .knowledge_schema import SpecialistKnowledge, SearchResult, OperationResponse


class SpecialistKnowledgeManager:
    """Manages specialist entities in the vector database"""
    
    def __init__(self):
        self.collection = vector_db_manager.get_specialists_collection()
    
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
    
    def _create_searchable_text(self, specialist: SpecialistKnowledge) -> str:
        """
        Create searchable text representation for embeddings.
        Combines key specialist fields into a coherent text.
        """
        parts = []
        
        parts.append(f"Specialist: {specialist.name}")
        if specialist.title:
            parts.append(f"Title: {specialist.title}")
        parts.append(f"Description: {specialist.description}")
        parts.append(f"Category: {specialist.category}")
        
        if specialist.type:
            parts.append(f"Type: {specialist.type}")
        if specialist.experience:
            parts.append(f"Experience: {specialist.experience}")
        if specialist.rating > 0:
            parts.append(f"Rating: {specialist.rating}/5.0")
        
        parts.append(f"Price: ${specialist.price}")
        parts.append(f"Duration: {specialist.duration} minutes")
        parts.append(f"Available: {'Yes' if specialist.availability else 'No'}")
        
        return " | ".join(parts)
    
    def add_specialist(self, specialist: SpecialistKnowledge) -> OperationResponse:
        """
        Add a new specialist to the vector database.
        
        Args:
            specialist: SpecialistKnowledge object with all specialist details
            
        Returns:
            OperationResponse with success status and specialist_id
        """
        try:
            searchable_text = self._create_searchable_text(specialist)
            specialist_dict = specialist.dict(exclude_none=True)
            flattened_metadata = self.flatten_metadata(specialist_dict)
            
            self.collection.add(
                documents=[searchable_text],
                metadatas=[flattened_metadata],
                ids=[specialist.specialist_id]
            )
            
            return OperationResponse(
                success=True,
                message="Specialist added successfully",
                data={"specialist_id": specialist.specialist_id}
            )
        except Exception as e:
            return OperationResponse(
                success=False,
                error=str(e)
            )
    
    def update_specialist(self, specialist_id: str, specialist: SpecialistKnowledge) -> OperationResponse:
        """
        Update an existing specialist in the vector database.
        
        Args:
            specialist_id: ID of the specialist to update
            specialist: SpecialistKnowledge object with updated details
            
        Returns:
            OperationResponse with success status
        """
        try:
            searchable_text = self._create_searchable_text(specialist)
            specialist_dict = specialist.dict(exclude_none=True)
            flattened_metadata = self.flatten_metadata(specialist_dict)
            
            self.collection.update(
                ids=[specialist_id],
                documents=[searchable_text],
                metadatas=[flattened_metadata]
            )
            
            return OperationResponse(
                success=True,
                message="Specialist updated successfully",
                data={"specialist_id": specialist_id}
            )
        except Exception as e:
            return OperationResponse(
                success=False,
                error=str(e)
            )
    
    def delete_specialist(self, specialist_id: str) -> OperationResponse:
        """
        Delete a specialist from the vector database.
        
        Args:
            specialist_id: ID of the specialist to delete
            
        Returns:
            OperationResponse with success status
        """
        try:
            self.collection.delete(ids=[specialist_id])
            return OperationResponse(
                success=True,
                message="Specialist deleted successfully",
                data={"specialist_id": specialist_id}
            )
        except Exception as e:
            return OperationResponse(
                success=False,
                error=str(e)
            )
    
    def search_specialists(
        self, 
        query: str, 
        n_results: int = 5, 
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Search for specialists using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Number of results to return (default: 5)
            filters: Optional metadata filters (e.g., {"category": "Finance"})
            
        Returns:
            List of SearchResult objects with specialist data and relevance scores
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters if filters else None
            )
            
            specialists = []
            if results['metadatas'] and len(results['metadatas']) > 0:
                for i, metadata in enumerate(results['metadatas'][0]):
                    specialist = SearchResult(
                        id=results['ids'][0][i] if results['ids'] else "",
                        data=metadata,
                        relevance_score=1 - results['distances'][0][i] if results['distances'] else 0.0
                    )
                    specialists.append(specialist)
            
            return specialists
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_specialist_by_id(self, specialist_id: str) -> Optional[Dict]:
        """
        Get a specific specialist by ID.
        
        Args:
            specialist_id: Specialist ID to retrieve
            
        Returns:
            Specialist data dictionary or None if not found
        """
        try:
            results = self.collection.get(ids=[specialist_id])
            if results['metadatas'] and len(results['metadatas']) > 0:
                return results['metadatas'][0]
            return None
        except Exception as e:
            print(f"Error getting specialist: {e}")
            return None
    
    def get_all_specialists(self, limit: int = 100) -> List[Dict]:
        """
        Get all specialists from the database.
        
        Args:
            limit: Maximum number of specialists to return (default: 100)
            
        Returns:
            List of specialist dictionaries
        """
        try:
            results = self.collection.get(limit=limit)
            specialists = []
            
            if results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    specialist = {
                        "id": results['ids'][i] if results['ids'] else None,
                        "data": metadata
                    }
                    specialists.append(specialist)
            
            return specialists
        except Exception as e:
            print(f"Error getting specialists: {e}")
            return []


# Global instance
specialist_knowledge_manager = SpecialistKnowledgeManager()
