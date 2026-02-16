import json
import requests
import time
import logging
from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from abacusai import ApiClient

from app.core.config import chatbot_config, settings
from app.utils.cache_manager import cache_manager
from app.utils.knowledge.product_knowledge import product_knowledge_manager
from app.utils.knowledge.service_knowledge import service_knowledge_manager
from app.utils.knowledge.consultation_knowledge import consultation_knowledge_manager
from app.utils.knowledge.specialist_knowledge import specialist_knowledge_manager
from .chatbot_schema import chatbot_request, chatbot_response, HistoryItem

# Configure logger
logger = logging.getLogger(__name__)


class OpenAIClient:
    """Simple OpenAI client for guardrail checks"""
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """Send chat completion request to OpenAI"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "temperature": temperature
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")


class AbacusAIClient:
    """Abacus AI client using official Python SDK"""
    
    def __init__(self):
        self.api_key = settings.ABACUS_API_KEY
        
        # Validate credentials
        if not self.api_key or self.api_key == "your-api-key":
            logger.error("[ABACUS] Missing or invalid ABACUS_API_KEY")
            self.client = None
        else:
            try:
                self.client = ApiClient(self.api_key)
                logger.info("[ABACUS] Successfully initialized Abacus.AI client")
            except Exception as e:
                logger.error(f"[ABACUS] Failed to initialize client: {str(e)}")
                self.client = None
    
    def get_conversation_response(self, message: str, conversation_id: Optional[str] = None) -> Dict:
        """
        Send message to Abacus AI using evaluate_prompt (no deployment needed)
        For conversation history, we'll manage it ourselves and include it in the prompt
        """
        try:
            # Validate client
            if not self.client:
                logger.error("[ABACUS] Client not initialized - check API key")
                return {"answer": "", "conversation_id": conversation_id, "success": False, "error": "Abacus client not initialized"}
            
            logger.info(f"[ABACUS] Message length: {len(message)} chars")
            logger.info(f"[ABACUS] Using evaluate_prompt (no deployment required)")
            
            # Call Abacus AI using evaluate_prompt
            # This method doesn't require a deployment ID
            result = self.client.evaluate_prompt(
                prompt=message
            )
            
            logger.info(f"[ABACUS] Success - received response")
            
            # evaluate_prompt returns a simple response
            answer = ""
            if hasattr(result, 'content'):
                answer = result.content
            elif hasattr(result, 'response'):
                answer = result.response
            elif isinstance(result, str):
                answer = result
            else:
                answer = str(result)
            
            logger.info(f"[ABACUS] Response length: {len(answer)} chars")
            
            # Since evaluate_prompt doesn't manage conversation IDs,
            # we return the same conversation_id that was passed in
            return {
                "answer": answer,
                "conversation_id": conversation_id,  # We manage this ourselves
                "success": True
            }
            
        except Exception as e:
            logger.error(f"[ABACUS] Error calling evaluate_prompt: {str(e)}")
            logger.error(f"[ABACUS] Error type: {type(e).__name__}")
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
    user_language: str
    english_query: str  # Translated query for vector search
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
        self.openai_client = OpenAIClient()
        
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
        logger.info(f"[GUARDRAIL] Starting guardrail check for user: {state['user_id']}")
        logger.info(f"[GUARDRAIL] User query: {state['user_query']}")
        
        # Get recent history for context
        recent_history = self.cache.get_history(state['user_id'])
        if recent_history:
            recent_history = recent_history[-3:]  # Last 3 exchanges
            logger.info(f"[GUARDRAIL] Found {len(recent_history)} history items")
        else:
            logger.info(f"[GUARDRAIL] No conversation history found")
        
        # Build context from history
        history_context = ""
        if recent_history:
            history_lines = []
            for h in recent_history:
                history_lines.append(f"User: {h.message}")
                history_lines.append(f"Assistant: {h.response}")
            history_context = "\n".join(history_lines)
        
        # Combined prompt for language detection, follow-up check, and e-commerce validation
        combined_prompt = f"""You are a classifier for an e-commerce chatbot. Analyze the user's query considering the conversation history.

Conversation History:
{history_context if history_context else "[No previous conversation]"}

Current User Query: {state['user_query']}

Please answer THREE questions:
1. What LANGUAGE is the user's query written in? (e.g., "English", "Arabic", "Spanish", "French", etc.)
2. Is this query a FOLLOW-UP question referencing the previous conversation? (e.g., "what about the price?", "show me more", "in blue")
3. Is this query related to E-COMMERCE topics? (products, services, shopping, pricing, delivery, etc.)

Respond with ONLY a JSON object:
{{"language": "<language_name>", "is_followup": true/false, "is_ecommerce": true/false, "reason": "brief explanation"}}"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a precise classifier. Always respond with valid JSON."},
                {"role": "user", "content": combined_prompt}
            ]
            
            response_content = self.openai_client.chat(messages)
            result = json.loads(response_content)
            
            state["user_language"] = result.get("language", "English")
            state["is_followup"] = result.get("is_followup", False)
            state["is_ecommerce_query"] = result.get("is_ecommerce", True)
            state["skip_retrieval"] = result.get("is_followup", False)  # Skip vector search for follow-ups
            
            logger.info(f"[GUARDRAIL] Classification result: {result}")
            logger.info(f"[GUARDRAIL] Detected language: {state['user_language']}")
            logger.info(f"[GUARDRAIL] is_followup={state['is_followup']}, is_ecommerce={state['is_ecommerce_query']}, skip_retrieval={state['skip_retrieval']}")
            
            # Translate query to English for vector search if needed
            if state['user_language'].lower() != 'english':
                try:
                    logger.info(f"[GUARDRAIL] Translating query to English for vector search")
                    translation_prompt = f"Translate this query to English. Only return the translation, nothing else: {state['user_query']}"
                    translation_messages = [{"role": "user", "content": translation_prompt}]
                    english_query = self.openai_client.chat(translation_messages, temperature=0.3)
                    state['english_query'] = english_query.strip()
                    logger.info(f"[GUARDRAIL] Translated query: {state['english_query']}")
                except Exception as e:
                    logger.error(f"[GUARDRAIL] Translation error: {str(e)}")
                    state['english_query'] = state['user_query']  # Fallback to original
            else:
                state['english_query'] = state['user_query']
                logger.info(f"[GUARDRAIL] Query already in English, no translation needed")
            
        except Exception as e:
            # On error, default to safe values
            logger.error(f"[GUARDRAIL] Error during classification: {str(e)}")
            state["user_language"] = "English"  # Default to English on error
            state["english_query"] = state['user_query']
            state["is_followup"] = False
            state["is_ecommerce_query"] = True
            state["skip_retrieval"] = False
        
        return state
    
    def retrieve_knowledge_node(self, state: ChatbotState) -> ChatbotState:
        """Search vector DBs (products, services, consultations, specialists)"""
        # Use English query for vector search (all DB content is in English)
        query = state['english_query']
        logger.info(f"[RETRIEVAL] Starting knowledge retrieval")
        logger.info(f"[RETRIEVAL] Original query: {state['user_query']}")
        logger.info(f"[RETRIEVAL] English query for vector search: {query}")
        
        try:
            # Products
            logger.info(f"[RETRIEVAL] Searching products...")
            try:
                products = product_knowledge_manager.search_products(query, n_results=3)
                if products:
                    state['retrieved_contexts']['products'] = [
                        f"{p.data.get('name')} - ${p.data.get('price')} - {p.data.get('description', '')[:100]}"
                        for p in products
                    ]
                    logger.info(f"[RETRIEVAL] Found {len(products)} products")
                else:
                    logger.info(f"[RETRIEVAL] No products found")
            except Exception as e:
                logger.error(f"[RETRIEVAL] Error searching products: {str(e)}")
                state['metadata']['product_error'] = str(e)
            
            # Services
            logger.info(f"[RETRIEVAL] Searching services...")
            try:
                services = service_knowledge_manager.search_services(query, n_results=3)
                if services:
                    state['retrieved_contexts']['services'] = [
                        f"{s.data.get('name')} - ${s.data.get('price')} - {s.data.get('description', '')[:100]}"
                        for s in services
                    ]
                    logger.info(f"[RETRIEVAL] Found {len(services)} services")
                else:
                    logger.info(f"[RETRIEVAL] No services found")
            except Exception as e:
                logger.error(f"[RETRIEVAL] Error searching services: {str(e)}")
                state['metadata']['service_error'] = str(e)
            
            # Consultations
            logger.info(f"[RETRIEVAL] Searching consultations...")
            try:
                consultations = consultation_knowledge_manager.search_consultations(query, n_results=2)
                if consultations:
                    state['retrieved_contexts']['consultations'] = [
                        f"{c.data.get('name')} - ${c.data.get('price')} ({c.data.get('duration')} min)"
                        for c in consultations
                    ]
                    logger.info(f"[RETRIEVAL] Found {len(consultations)} consultations")
                else:
                    logger.info(f"[RETRIEVAL] No consultations found")
            except Exception as e:
                logger.error(f"[RETRIEVAL] Error searching consultations: {str(e)}")
                state['metadata']['consultation_error'] = str(e)
            
            # Specialists
            logger.info(f"[RETRIEVAL] Searching specialists...")
            try:
                specialists = specialist_knowledge_manager.search_specialists(query, n_results=2)
                if specialists:
                    state['retrieved_contexts']['specialists'] = [
                        f"{s.data.get('name')} - {s.data.get('experience')} (Rating: {s.data.get('rating')})"
                        for s in specialists
                    ]
                    logger.info(f"[RETRIEVAL] Found {len(specialists)} specialists")
                else:
                    logger.info(f"[RETRIEVAL] No specialists found")
            except Exception as e:
                logger.error(f"[RETRIEVAL] Error searching specialists: {str(e)}")
                state['metadata']['specialist_error'] = str(e)
            
            # Summary
            total_contexts = sum(len(v) for v in state['retrieved_contexts'].values())
            logger.info(f"[RETRIEVAL] Completed. Total contexts retrieved: {total_contexts}")
            logger.info(f"[RETRIEVAL] Retrieved contexts by type: {', '.join([f'{k}:{len(v)}' for k, v in state['retrieved_contexts'].items()])}")
        
        except Exception as e:
            logger.error(f"[RETRIEVAL] Unexpected error in retrieval node: {str(e)}")
            state['metadata']['retrieval_error'] = str(e)
        
        return state
    
    def generate_response_node(self, state: ChatbotState) -> ChatbotState:
        """
        Generate response using Abacus AI
        Since evaluate_prompt doesn't manage conversation history automatically,
        we need to include recent history in the prompt
        """
        logger.info(f"[GENERATION] Starting response generation")
        
        # Prepare context from vector search
        context_parts = []
        for source, contexts in state['retrieved_contexts'].items():
            if contexts:
                context_parts.append(f"{source.upper()}:\n" + "\n".join(contexts))
        
        combined_context = "\n\n".join(context_parts) if context_parts else ""
        logger.info(f"[GENERATION] Context length: {len(combined_context)} chars, {len(context_parts)} sources")
        
        # Get conversation history for context (since evaluate_prompt doesn't manage it)
        conversation_history = ""
        recent_history = self.cache.get_history(state['user_id'])
        if recent_history and len(recent_history) > 0:
            recent_history = recent_history[-3:]  # Last 3 exchanges
            history_lines = []
            for h in recent_history:
                history_lines.append(f"User: {h.message}")
                history_lines.append(f"Assistant: {h.response}")
            conversation_history = "\n".join(history_lines)
            logger.info(f"[GENERATION] Including {len(recent_history)} previous exchanges in context")
        
        # Build comprehensive prompt with history and context
        prompt_parts = []
        
        if conversation_history:
            prompt_parts.append(f"Previous conversation:\n{conversation_history}\n")
        
        if combined_context:
            prompt_parts.append(f"Relevant information from our database:\n{combined_context}\n")
        
        prompt_parts.append(f"Current user query: {state['user_query']}\n")
        prompt_parts.append(f"IMPORTANT: The user is communicating in {state['user_language']}. You MUST respond in {state['user_language']} language.")
        prompt_parts.append("Please provide a helpful, friendly response based on the context and conversation history.")
        
        full_message = "\n".join(prompt_parts)
        
        logger.info(f"[GENERATION] Responding in language: {state['user_language']}")
        
        try:
            # Get conversation_id
            conversation_id = self.session_mgr.get_conversation_id(state['user_id'])
            logger.info(f"[GENERATION] Conversation ID: {conversation_id if conversation_id else 'None (new conversation)'}")
            
            # Call Abacus AI
            logger.info(f"[GENERATION] Calling Abacus AI...")
            result = self.abacus.get_conversation_response(full_message, conversation_id)
            
            if result['success']:
                state['final_response'] = result['answer']
                logger.info(f"[GENERATION] Successfully generated response (length: {len(result['answer'])} chars)")
                # Store new conversation_id
                if result['conversation_id']:
                    self.session_mgr.set_conversation_id(state['user_id'], result['conversation_id'])
                    logger.info(f"[GENERATION] Updated conversation ID: {result['conversation_id']}")
            else:
                logger.error(f"[GENERATION] Abacus AI call failed: {result.get('error')}")
                state['final_response'] = "I apologize, but I encountered an error. Please try again."
                state['metadata']['error'] = result.get('error')
        
        except Exception as e:
            logger.error(f"[GENERATION] Exception during response generation: {str(e)}")
            state['final_response'] = "I apologize, but I encountered an error. Please try again."
            state['metadata']['error'] = str(e)
        
        return state
    
    def reject_query_node(self, state: ChatbotState) -> ChatbotState:
        """Reject non-e-commerce queries in the user's language"""
        try:
            # Translate rejection message to user's language if not English
            rejection_message = self.config.guardrail_config.rejection_message
            
            if state['user_language'].lower() != 'english':
                logger.info(f"[REJECT] Translating rejection to {state['user_language']}")
                prompt = f"Translate this message to {state['user_language']}: {rejection_message}"
                messages = [{"role": "user", "content": prompt}]
                translated = self.openai_client.chat(messages, temperature=0.3)
                state['final_response'] = translated
            else:
                state['final_response'] = rejection_message
        except Exception as e:
            logger.error(f"[REJECT] Error translating rejection: {str(e)}")
            state['final_response'] = self.config.guardrail_config.rejection_message
        
        return state
    
    def route_after_guardrail(self, state: ChatbotState) -> str:
        """Route based on guardrail results"""
        if not state['is_ecommerce_query']:
            logger.info(f"[ROUTING] Query rejected - not e-commerce related")
            return "reject"
        
        if state['skip_retrieval'] or state['is_followup']:
            logger.info(f"[ROUTING] Routing to direct_response (followup: {state['is_followup']}, skip: {state['skip_retrieval']})")
            return "direct_response"  # Skip vector search for follow-ups
        
        logger.info(f"[ROUTING] Routing to retrieve - performing vector search")
        return "retrieve"  # New query, search DBs
    
    def chat(self, request: chatbot_request) -> chatbot_response:
        """
        Main chat interface
        
        History is managed server-side:
        - Redis provides recent history for guardrail/follow-up detection
        - Abacus AI maintains full conversation via conversation_id
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"[CHAT] New chat request from user: {request.user_id}")
        logger.info(f"[CHAT] Message: {request.message}")
        logger.info(f"{'='*80}")
        
        # Initialize state (no history from request needed)
        initial_state: ChatbotState = {
            "messages": [],  # Not used since Abacus handles history
            "user_query": request.message,
            "user_id": request.user_id,
            "conversation_id": None,
            "user_language": "English",  # Will be detected in guardrail_check
            "english_query": request.message,  # Will be translated if needed
            "is_ecommerce_query": False,
            "is_followup": False,
            "skip_retrieval": False,
            "retrieved_contexts": {},
            "final_response": "",
            "metadata": {}
        }
        
        # Execute graph
        logger.info(f"[CHAT] Starting LangGraph execution...")
        final_state = self.graph.invoke(initial_state)
        logger.info(f"[CHAT] LangGraph execution completed")
        
        # Update Redis history
        self.cache.update_history(
            user_id=request.user_id,
            new_message=request.message,
            new_response=final_state['final_response']
        )
        
        # Log final state
        logger.info(f"[CHAT] Final response length: {len(final_state['final_response'])} chars")
        if final_state['metadata']:
            logger.warning(f"[CHAT] Metadata (errors): {final_state['metadata']}")
        logger.info(f"{'='*80}\n")
        
        # Return response (metadata removed from schema)
        return chatbot_response(
            response=final_state['final_response']
        )


# Global agent instance
ecommerce_agent = EcommerceChatbotAgent()
