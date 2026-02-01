from fastapi import APIRouter, HTTPException, Header
from .chatbot_schema import chatbot_request, chatbot_response
from .chatbot import ecommerce_agent

router = APIRouter()


@router.post("/chat", response_model=chatbot_response)
async def chat(
    request: chatbot_request
):
    return ecommerce_agent.chat(request)