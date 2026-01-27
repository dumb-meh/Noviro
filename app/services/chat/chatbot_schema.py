from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from fastapi import Header

class HistoryItem(BaseModel):
    message: str
    response: str
    
class chatbot_request(BaseModel):
    history: Optional[List[HistoryItem]] = []
    user_id: str  # Fixed: should be a normal field, not Header
    message: str
    session_id: Optional[str] = None  # For conversation tracking
    metadata: Optional[Dict[str, Any]] = {}  # For extensibility
    
class chatbot_response(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = {}  # For returning additional info
