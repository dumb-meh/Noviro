import json
import os
from typing import Any, Dict

import openai
from dotenv import load_dotenv

from .chatbot_schema import chatbot_request, chatbot_response
from .langgraph_chatbot import ecommerce_agent

load_dotenv()


class Affirmation:
    """
    Original Affirmation class - ALL METHODS PRESERVED UNCHANGED.
    Added chat_with_agent method for LangGraph-based chatbot functionality.
    """
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.reasoning_model = "gpt-4.1"

    def generate_affirmations(self, request: chatbot_request) -> chatbot_response:
        """Generate 12 affirmations based on quiz data and alignments, avoiding past themes."""
        prompt = self._create_affirmation_prompt(request)
        response_data = self._get_openai_response(prompt)
        
       

    def _create_prompt(self, request: chatbot_request) -> str:
        prompt= """ test"""

        return prompt

    def _get_openai_response(self, prompt: str) -> Dict[str, Any] | None:
        """Call OpenAI API to generate affirmations."""
        try:
            payload_data = json.loads(prompt)
            system_content = payload_data.get("system", "")
            user_content = json.dumps(payload_data.get("payload", {}), ensure_ascii=False)

            response = self.client.chat.completions.create(
                model=self.reasoning_model,
                temperature=0.9,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
            )
            
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
        except Exception:
            return None
        return None
    
    # NEW METHOD: LangGraph-based chatbot
    def chat_with_agent(self, request: chatbot_request) -> chatbot_response:
        """
        NEW: Chat with the LangGraph-based e-commerce agent.
        
        This method uses the scalable LangGraph chatbot with:
        - Guardrails for e-commerce-only queries
        - Seven vector database integration points
        - State-based conversation management
        
        Args:
            request: chatbot_request with user message and history
            
        Returns:
            chatbot_response with generated response
        """
        return ecommerce_agent.chat(request)
