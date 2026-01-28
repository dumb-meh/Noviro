from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from fastapi import Header

class HistoryItem(BaseModel):
    message: str
    response: str
    
class chatbot_request(BaseModel):
    history: Optional[List[HistoryItem]] = []
    user_id: Header(str)
    message: str
    session_id: Optional[str] = None  
    metadata: Optional[Dict[str, Any]] = {}  
    
class chatbot_response(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = {}
