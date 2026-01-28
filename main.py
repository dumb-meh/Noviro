import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from app.services.chat.chatbot_route import router as chat_router
from app.utils.knowledge.knowledge_route import router as knowledge_router

load_dotenv()

app = FastAPI(
    title="AdrianaBrill AI Service",
    description="AI-powered e-commerce assistant with product suggestions and chat",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(knowledge_router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Noviro AI Service",
        "endpoints": {
            "ai_suggestions": "/api/ai_suggestions",
            "chat": "/api/chat",
            "knowledge": "/api/knowledge/products",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "service": "adrianabrill-ai"}
    )

# Error handlers
@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8085, 
        reload=True
    )