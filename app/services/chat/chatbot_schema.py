from pydantic import BaseModel
from typing import Optional,List, Any, Dict
from fastapi import Header

class HistoryItem(BaseModel):
    message: str
    response: str
    
class chatbot_request(BaseModel):
    history: Optional[List[HistoryItem]] = []
    user_id: Header(str)
    message: str
    
class chatbot_response(BaseModel):
    response:str
