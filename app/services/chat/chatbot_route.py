from fastapi import APIRouter, HTTPException
from .chatbot import Chatbot
from .chatbot_schema import  chatbot_request,chatbot_response

router = APIRouter()
chatbot = Chatbot()


@router.post("/chatbot", response_model=chatbot_response)
async def generate_affirmations(request: chatbot_request):
    """Generate 12 affirmations based on quiz data and alignments."""
    try:
        response = chatbot.chatbot(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))