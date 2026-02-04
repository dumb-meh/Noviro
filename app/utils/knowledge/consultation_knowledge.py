"""
Consultation Knowledge Manager

Handles CRUD operations for consultations in the ChromaDB vector database.
Includes semantic search, add, update, and delete operations.
"""

from typing import List, Dict, Optional, Any
from app.vectordb.manager import vector_db_manager
from .knowledge_schema import ConsultationKnowledge, SearchResult, OperationResponse


class ConsultationKnowledgeManager:
    """Manages consultation entities in the vector database"""
    
    def __init__(self):
        self.collection = vector_db_manager.get_consultations_collection()
    
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
    
    def _create_searchable_text(self, consultation: ConsultationKnowledge) -> str:
        """
        Create searchable text representation for embeddings.
        Combines key consultation fields into a coherent text.
        """
        parts = []
        
        parts.append(f"Consultant: {consultation.name}")
        if consultation.title:
            parts.append(f"Title: {consultation.title}")
        parts.append(f"Description: {consultation.description}")
        parts.append(f"Category: {consultation.category}")
        
        if consultation.type:
            parts.append(f"Type: {consultation.type}")
        if consultation.experience:
            parts.append(f"Experience: {consultation.experience}")
        
        parts.append(f"Price: ${consultation.price}")
        parts.append(f"Duration: {consultation.duration} minutes")
        
        return " | ".join(parts)
    
    def add_consultation(self, consultation: ConsultationKnowledge) -> OperationResponse:
        """
        Add a new consultation to the vector database.
        
        Args:
            consultation: ConsultationKnowledge object with all consultation details
            
        Returns:
            OperationResponse with success status and consultation_id
        """
        try:
            searchable_text = self._create_searchable_text(consultation)
            consultation_dict = consultation.dict(exclude_none=True)
            flattened_metadata = self.flatten_metadata(consultation_dict)
            
            self.collection.add(
                documents=[searchable_text],
                metadatas=[flattened_metadata],
                ids=[consultation.consultation_id]
            )
            
            return OperationResponse(
                success=True,
                message="Consultation added successfully",
                data={"consultation_id": consultation.consultation_id}
            )
        except Exception as e:
            return OperationResponse(
                success=False,
                error=str(e)
            )
    
    def update_consultation(self, consultation_id: str, consultation: ConsultationKnowledge) -> OperationResponse:
        """
        Update an existing consultation in the vector database.
        
        Args:
            consultation_id: ID of the consultation to update
            consultation: ConsultationKnowledge object with updated details
            
        Returns:
            OperationResponse with success status
        """
        try:
            searchable_text = self._create_searchable_text(consultation)
            consultation_dict = consultation.dict(exclude_none=True)
            flattened_metadata = self.flatten_metadata(consultation_dict)
            
            self.collection.update(
                ids=[consultation_id],
                documents=[searchable_text],
                metadatas=[flattened_metadata]
            )
            
            return OperationResponse(
                success=True,
                message="Consultation updated successfully",
                data={"consultation_id": consultation_id}
            )
        except Exception as e:
            return OperationResponse(
                success=False,
                error=str(e)
            )
    
    def delete_consultation(self, consultation_id: str) -> OperationResponse:
        """
        Delete a consultation from the vector database.
        
        Args:
            consultation_id: ID of the consultation to delete
            
        Returns:
            OperationResponse with success status
        """
        try:
            self.collection.delete(ids=[consultation_id])
            return OperationResponse(
                success=True,
                message="Consultation deleted successfully",
                data={"consultation_id": consultation_id}
            )
        except Exception as e:
            return OperationResponse(
                success=False,
                error=str(e)
            )
    
    def search_consultations(
        self, 
        query: str, 
        n_results: int = 5, 
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Search for consultations using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Number of results to return (default: 5)
            filters: Optional metadata filters (e.g., {"category": "Business"})
            
        Returns:
            List of SearchResult objects with consultation data and relevance scores
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters if filters else None
            )
            
            consultations = []
            if results['metadatas'] and len(results['metadatas']) > 0:
                for i, metadata in enumerate(results['metadatas'][0]):
                    consultation = SearchResult(
                        id=results['ids'][0][i] if results['ids'] else "",
                        data=metadata,
                        relevance_score=1 - results['distances'][0][i] if results['distances'] else 0.0
                    )
                    consultations.append(consultation)
            
            return consultations
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_consultation_by_id(self, consultation_id: str) -> Optional[Dict]:
        """
        Get a specific consultation by ID.
        
        Args:
            consultation_id: Consultation ID to retrieve
            
        Returns:
            Consultation data dictionary or None if not found
        """
        try:
            results = self.collection.get(ids=[consultation_id])
            if results['metadatas'] and len(results['metadatas']) > 0:
                return results['metadatas'][0]
            return None
        except Exception as e:
            print(f"Error getting consultation: {e}")
            return None
    
    def get_all_consultations(self, limit: int = 100) -> List[Dict]:
        """
        Get all consultations from the database.
        
        Args:
            limit: Maximum number of consultations to return (default: 100)
            
        Returns:
            List of consultation dictionaries
        """
        try:
            results = self.collection.get(limit=limit)
            consultations = []
            
            if results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    consultation = {
                        "id": results['ids'][i] if results['ids'] else None,
                        "data": metadata
                    }
                    consultations.append(consultation)
            
            return consultations
        except Exception as e:
            print(f"Error getting consultations: {e}")
            return []


# Global instance
consultation_knowledge_manager = ConsultationKnowledgeManager()
