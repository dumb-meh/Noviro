from pydantic import BaseModel
from typing import Optional, Any, Dict

class HistoryItem(BaseModel):
    message: str
    response: str
    
class chatbot_request(BaseModel):
    message: str
    user_id: Optional[str] = None  
    
class chatbot_response(BaseModel):
    response: str
