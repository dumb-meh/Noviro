import json
import requests
import time
from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.core.config import chatbot_config, settings
from app.utils.cache_manager import cache_manager
from app.utils.knowledge.product_knowledge import product_knowledge_manager
from app.utils.knowledge.service_knowledge import service_knowledge_manager
from app.utils.knowledge.consultation_knowledge import consultation_knowledge_manager
from app.utils.knowledge.specialist_knowledge import specialist_knowledge_manager
from .chatbot_schema import chatbot_request, chatbot_response, HistoryItem


class AbacusAIClient:
    """Simple Abacus AI client for LLM calls"""
    
    def __init__(self):
        self.api_key = settings.ABACUS_API_KEY
        self.deployment_id = settings.ABACUS_DEPLOYMENT_ID
        self.base_url = "https://api.abacus.ai/api"
    
    def get_conversation_response(self, message: str, conversation_id: Optional[str] = None) -> Dict:
        """Send message to Abacus AI with conversation context"""
        try:
            headers = {"apiKey": self.api_key}
            data = {
                "deploymentId": self.deployment_id,
                "message": message
            }
            if conversation_id:
                data["conversationId"] = conversation_id
            
            response = requests.post(
                f"{self.base_url}/getConversationResponse",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            return {
                "answer": result.get("result", {}).get("answer", ""),
                "conversation_id": result.get("result", {}).get("conversationId"),
                "success": True
            }
        except Exception as e:
            return {"answer": "", "conversation_id": conversation_id, "success": False, "error": str(e)}



class SessionManager:
    """Manages Abacus conversation IDs in Redis"""
    
    def __init__(self):
        self.cache = cache_manager
    
    def get_conversation_id(self, user_id: str) -> Optional[str]:
        """Get Abacus conversation_id from Redis"""
        if not self.cache.redis_client or not user_id:
            return None
        try:
            key = f"abacus_conversation:{user_id}"
            conv_id = self.cache.redis_client.get(key)
            return conv_id.decode('utf-8') if conv_id else None
        except:
            return None
    
    def set_conversation_id(self, user_id: str, conversation_id: str):
        """Store Abacus conversation_id in Redis"""
        if not self.cache.redis_client or not user_id:
            return
        try:
            key = f"abacus_conversation:{user_id}"
            ttl = settings.ABACUS_CONVERSATION_TTL_DAYS * 24 * 3600
            self.cache.redis_client.setex(key, ttl, conversation_id)
        except:
            pass



class ChatbotState(TypedDict):
    """LangGraph state"""
    messages: Annotated[Sequence[BaseMessage], add]
    user_query: str
    user_id: str
    conversation_id: Optional[str]
    is_ecommerce_query: bool
    is_followup: bool
    skip_retrieval: bool
    retrieved_contexts: Dict[str, List[str]]
    final_response: str
    metadata: Dict[str, Any]

class EcommerceChatbotAgent:
    """
    LangGraph E-commerce Chatbot
    
    - OpenAI for guardrails AND follow-up detection (single call)
    - Abacus AI for responses (main LLM)
    - Smart routing based on LLM classification
    """
    
    def __init__(self):
        self.config = chatbot_config
        
        # OpenAI for guardrails + follow-up detection
        self.guardrail_llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # Abacus AI for responses
        self.abacus = AbacusAIClient()
        
        # Helpers
        self.session_mgr = SessionManager()
        self.cache = cache_manager
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(ChatbotState)
        
        workflow.add_node("guardrail_check", self.guardrail_check_node)
        workflow.add_node("retrieve_knowledge", self.retrieve_knowledge_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("reject_query", self.reject_query_node)
        
        workflow.set_entry_point("guardrail_check")
        
        workflow.add_conditional_edges(
            "guardrail_check",
            self.route_after_guardrail,
            {
                "retrieve": "retrieve_knowledge",
                "direct_response": "generate_response",
                "reject": "reject_query"
            }
        )
        
        workflow.add_edge("retrieve_knowledge", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("reject_query", END)
        
        return workflow.compile()
    
    def guardrail_check_node(self, state: ChatbotState) -> ChatbotState:
        """
        Guardrail: Check e-commerce + detect follow-ups using OpenAI
        Single LLM call for both checks
        """
        # Get recent history for context
        recent_history = self.cache.get_history(state['user_id'])
        if recent_history:
            recent_history = recent_history[-3:]  # Last 3 exchanges
        
        # Build context from history
        history_context = ""
        if recent_history:
            history_lines = []
            for h in recent_history:
                history_lines.append(f"User: {h.message}")
                history_lines.append(f"Assistant: {h.response}")
            history_context = "\n".join(history_lines)
        
        # Combined prompt for both checks
        combined_prompt = f"""You are a classifier for an e-commerce chatbot. Analyze the user's query considering the conversation history.

Conversation History:
{history_context if history_context else "[No previous conversation]"}

Current User Query: {state['user_query']}

Please answer TWO questions:
1. Is this query a FOLLOW-UP question referencing the previous conversation? (e.g., "what about the price?", "show me more", "in blue")
2. Is this query related to E-COMMERCE topics? (products, services, shopping, pricing, delivery, etc.)

Respond with ONLY a JSON object:
{{"is_followup": true/false, "is_ecommerce": true/false, "reason": "brief explanation"}}"""
        
        try:
            messages = [
                SystemMessage(content="You are a precise classifier. Always respond with valid JSON."),
                HumanMessage(content=combined_prompt)
            ]
            
            response = self.guardrail_llm.invoke(messages)
            result = json.loads(response.content)
            
            state["is_followup"] = result.get("is_followup", False)
            state["is_ecommerce_query"] = result.get("is_ecommerce", True)
            state["skip_retrieval"] = result.get("is_followup", False)  # Skip vector search for follow-ups
            
        except Exception as e:
            # On error, default to safe values
            state["is_followup"] = False
            state["is_ecommerce_query"] = True
            state["skip_retrieval"] = False
        
        return state
    
    def retrieve_knowledge_node(self, state: ChatbotState) -> ChatbotState:
        """Search vector DBs (products, services, consultations, specialists)"""
        query = state['user_query']
        
        try:
            # Products
            products = product_knowledge_manager.search_products(query, n_results=3)
            if products:
                state['retrieved_contexts']['products'] = [
                    f"{p.data.get('name')} - ${p.data.get('price')} - {p.data.get('description', '')[:100]}"
                    for p in products
                ]
            
            # Services
            services = service_knowledge_manager.search_services(query, n_results=3)
            if services:
                state['retrieved_contexts']['services'] = [
                    f"{s.data.get('name')} - ${s.data.get('price')} - {s.data.get('description', '')[:100]}"
                    for s in services
                ]
            
            # Consultations
            consultations = consultation_knowledge_manager.search_consultations(query, n_results=2)
            if consultations:
                state['retrieved_contexts']['consultations'] = [
                    f"{c.data.get('name')} - ${c.data.get('price')} ({c.data.get('duration')} min)"
                    for c in consultations
                ]
            
            # Specialists
            specialists = specialist_knowledge_manager.search_specialists(query, n_results=2)
            if specialists:
                state['retrieved_contexts']['specialists'] = [
                    f"{s.data.get('name')} - {s.data.get('experience')} (Rating: {s.data.get('rating')})"
                    for s in specialists
                ]
        
        except Exception as e:
            state['metadata']['retrieval_error'] = str(e)
        
        return state
    
    def generate_response_node(self, state: ChatbotState) -> ChatbotState:
        """
        Generate response using Abacus AI
        Uses conversation_id for multi-turn context
        """
        # Prepare context
        context_parts = []
        for source, contexts in state['retrieved_contexts'].items():
            if contexts:
                context_parts.append(f"{source.upper()}:\n" + "\n".join(contexts))
        
        combined_context = "\n\n".join(context_parts) if context_parts else ""
        
        # Build message
        if combined_context:
            full_message = f"""Context from our database:
{combined_context}

User Query: {state['user_query']}

Please provide a helpful response based on the context."""
        else:
            full_message = state['user_query']
        
        try:
            # Get conversation_id
            conversation_id = self.session_mgr.get_conversation_id(state['user_id'])
            
            # Call Abacus AI
            result = self.abacus.get_conversation_response(full_message, conversation_id)
            
            if result['success']:
                state['final_response'] = result['answer']
                # Store new conversation_id
                if result['conversation_id']:
                    self.session_mgr.set_conversation_id(state['user_id'], result['conversation_id'])
            else:
                state['final_response'] = "I apologize, but I encountered an error. Please try again."
                state['metadata']['error'] = result.get('error')
        
        except Exception as e:
            state['final_response'] = "I apologize, but I encountered an error. Please try again."
            state['metadata']['error'] = str(e)
        
        return state
    
    def reject_query_node(self, state: ChatbotState) -> ChatbotState:
        """Reject non-e-commerce queries"""
        state['final_response'] = self.config.guardrail_config.rejection_message
        return state
    
    def route_after_guardrail(self, state: ChatbotState) -> str:
        """Route based on guardrail results"""
        if not state['is_ecommerce_query']:
            return "reject"
        
        if state['skip_retrieval'] or state['is_followup']:
            return "direct_response"  # Skip vector search for follow-ups
        
        return "retrieve"  # New query, search DBs
    
    def chat(self, request: chatbot_request) -> chatbot_response:
        """
        Main chat interface
        
        History is managed server-side:
        - Redis provides recent history for guardrail/follow-up detection
        - Abacus AI maintains full conversation via conversation_id
        """
        # Initialize state (no history from request needed)
        initial_state: ChatbotState = {
            "messages": [],  # Not used since Abacus handles history
            "user_query": request.message,
            "user_id": request.user_id,
            "conversation_id": None,
            "is_ecommerce_query": False,
            "is_followup": False,
            "skip_retrieval": False,
            "retrieved_contexts": {},
            "final_response": "",
            "metadata": {}
        }
        
        # Execute graph
        final_state = self.graph.invoke(initial_state)
        
        # Update Redis history
        self.cache.update_history(
            user_id=request.user_id,
            new_message=request.message,
            new_response=final_state['final_response']
        )
        
        # Return response (metadata removed from schema)
        return chatbot_response(
            response=final_state['final_response']
        )


# Global agent instance
ecommerce_agent = EcommerceChatbotAgent()
