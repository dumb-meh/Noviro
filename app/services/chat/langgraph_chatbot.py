

import json
from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.core.config import chatbot_config
from .chatbot_schema import chatbot_request, chatbot_response, HistoryItem


class ChatbotState(TypedDict):
    """
    State schema for the chatbot agent.
    
    Attributes:
        messages: Conversation history as LangChain messages
        user_query: Current user query
        user_id: User identifier for personalization
        session_id: Session identifier for conversation tracking
        is_ecommerce_query: Flag from guardrail check
        guardrail_reason: Explanation from guardrail
        retrieved_contexts: Contexts from vector databases
        final_response: Generated response
        metadata: Additional metadata for extensibility
    """
    messages: Annotated[Sequence[BaseMessage], add]
    user_query: str
    user_id: str
    session_id: Optional[str]
    is_ecommerce_query: bool
    guardrail_reason: str
    retrieved_contexts: Dict[str, List[str]]
    final_response: str
    metadata: Dict[str, Any]


class EcommerceChatbotAgent:
    """
    LangGraph-based E-commerce Chatbot Agent
    
    This agent uses a state graph to process queries through multiple stages:
    1. Guardrail check - validates query is e-commerce related
    2. Vector database retrieval - fetches relevant context from 7 databases
    3. Response generation - synthesizes context into helpful response
    """
    
    def __init__(self):
        self.config = chatbot_config
        
        # Initialize LLM for guardrail (lower temperature for classification)
        self.guardrail_llm = ChatOpenAI(
            api_key=self.config.openai_api_key,
            model=self.config.model_config.model_name,
            temperature=self.config.guardrail_config.temperature
        )
        
        # Initialize LLM for response generation
        self.response_llm = ChatOpenAI(
            api_key=self.config.openai_api_key,
            model=self.config.model_config.model_name,
            temperature=self.config.response_config.temperature,
            max_tokens=self.config.response_config.max_tokens
        )
        
        # Build the state graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph with all nodes and conditional edges.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        workflow = StateGraph(ChatbotState)
        
        # Add nodes
        workflow.add_node("guardrail_check", self.guardrail_check_node)
        workflow.add_node("retrieve_products", self.retrieve_products_node)
        workflow.add_node("retrieve_product_specs", self.retrieve_product_specs_node)
        workflow.add_node("retrieve_reviews", self.retrieve_reviews_node)
        workflow.add_node("retrieve_inventory", self.retrieve_inventory_node)
        workflow.add_node("retrieve_pricing", self.retrieve_pricing_node)
        workflow.add_node("retrieve_shipping", self.retrieve_shipping_node)
        workflow.add_node("retrieve_support", self.retrieve_support_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("reject_query", self.reject_query_node)
        
        # Set entry point
        workflow.set_entry_point("guardrail_check")
        
        # Add conditional edge from guardrail
        workflow.add_conditional_edges(
            "guardrail_check",
            self.route_after_guardrail,
            {
                "proceed": "retrieve_products",
                "reject": "reject_query"
            }
        )
        
        # Chain retrieval nodes (parallel retrieval can be optimized later)
        workflow.add_edge("retrieve_products", "retrieve_product_specs")
        workflow.add_edge("retrieve_product_specs", "retrieve_reviews")
        workflow.add_edge("retrieve_reviews", "retrieve_inventory")
        workflow.add_edge("retrieve_inventory", "retrieve_pricing")
        workflow.add_edge("retrieve_pricing", "retrieve_shipping")
        workflow.add_edge("retrieve_shipping", "retrieve_support")
        workflow.add_edge("retrieve_support", "generate_response")
        
        # End nodes
        workflow.add_edge("generate_response", END)
        workflow.add_edge("reject_query", END)
        
        return workflow.compile()

    
    def guardrail_check_node(self, state: ChatbotState) -> ChatbotState:
        """
        Guardrail node: Validates if query is e-commerce related.
        
        Uses LLM to classify the query and determine if it should be processed.
        """
        if not self.config.guardrail_config.enabled:
            state["is_ecommerce_query"] = True
            state["guardrail_reason"] = "Guardrail disabled"
            return state
        
        try:
            messages = [
                SystemMessage(content=self.config.guardrail_config.guardrail_system_prompt),
                HumanMessage(content=f"Query: {state['user_query']}")
            ]
            
            response = self.guardrail_llm.invoke(messages)
            result = json.loads(response.content)
            
            state["is_ecommerce_query"] = result.get("is_ecommerce", False)
            state["guardrail_reason"] = result.get("reason", "Unknown")
            
        except Exception as e:
            # On error, default to allowing the query
            state["is_ecommerce_query"] = True
            state["guardrail_reason"] = f"Guardrail error: {str(e)}"
        
        return state
    
    def retrieve_products_node(self, state: ChatbotState) -> ChatbotState:
        """
        Retrieve from products catalog database.
        
        PLACEHOLDER: This will be implemented with actual vector DB retrieval.
        For now, returns empty context with clear integration point.
        """

        
        state['retrieved_contexts']['products'] = []
        return state
    
    def retrieve_product_specs_node(self, state: ChatbotState) -> ChatbotState:
        """
        Retrieve from product specifications database.
        
        PLACEHOLDER: Integration point for technical specifications vector DB.
        """
        # TODO: Implement product specs vector DB retrieval
        state['retrieved_contexts']['product_specs'] = []
        return state
    
    def retrieve_reviews_node(self, state: ChatbotState) -> ChatbotState:
        """
        Retrieve from customer reviews database.
        
        PLACEHOLDER: Integration point for reviews vector DB.
        """
        # TODO: Implement reviews vector DB retrieval
        state['retrieved_contexts']['reviews'] = []
        return state
    
    def retrieve_inventory_node(self, state: ChatbotState) -> ChatbotState:
        """
        Retrieve from inventory/stock database.
        
        PLACEHOLDER: Integration point for inventory vector DB.
        """
        # TODO: Implement inventory vector DB retrieval
        state['retrieved_contexts']['inventory'] = []
        return state
    
    def retrieve_pricing_node(self, state: ChatbotState) -> ChatbotState:
        """
        Retrieve from pricing and promotions database.
        
        PLACEHOLDER: Integration point for pricing vector DB.
        """
        # TODO: Implement pricing vector DB retrieval
        state['retrieved_contexts']['pricing'] = []
        return state
    
    def retrieve_shipping_node(self, state: ChatbotState) -> ChatbotState:
        """
        Retrieve from shipping information database.
        
        PLACEHOLDER: Integration point for shipping vector DB.
        """
        # TODO: Implement shipping vector DB retrieval
        state['retrieved_contexts']['shipping'] = []
        return state
    
    def retrieve_support_node(self, state: ChatbotState) -> ChatbotState:
        """
        Retrieve from support documentation database.
        
        PLACEHOLDER: Integration point for support docs vector DB.
        """
        # TODO: Implement support docs vector DB retrieval
        state['retrieved_contexts']['support'] = []
        return state
    
    def generate_response_node(self, state: ChatbotState) -> ChatbotState:
        """
        Generate final response using retrieved contexts.
        
        Synthesizes information from all vector databases into coherent response.
        """
        # Prepare context from all retrieved sources
        context_parts = []
        for source, contexts in state['retrieved_contexts'].items():
            if contexts:
                context_parts.append(f"{source.upper()}:\n" + "\n".join(contexts))
        
        combined_context = "\n\n".join(context_parts) if context_parts else "No specific context available."
        
        # Format the prompt
        prompt = self.config.response_config.system_prompt.format(
            context=combined_context,
            query=state['user_query']
        )
        
        # Generate response
        try:
            messages = [
                SystemMessage(content=prompt),
                *state['messages'][-10:],  # Include last 10 messages for context
                HumanMessage(content=state['user_query'])
            ]
            
            response = self.response_llm.invoke(messages)
            state['final_response'] = response.content
            
        except Exception as e:
            state['final_response'] = f"I apologize, but I encountered an error processing your request. Please try again."
            state['metadata']['error'] = str(e)
        
        return state
    
    def reject_query_node(self, state: ChatbotState) -> ChatbotState:
        """
        Handle rejected queries (non-e-commerce).
        
        Returns polite rejection message.
        """
        state['final_response'] = self.config.guardrail_config.rejection_message
        return state
    

    
    def route_after_guardrail(self, state: ChatbotState) -> str:
        """
        Conditional routing after guardrail check.
        
        Returns:
            "proceed" if query is e-commerce related
            "reject" if query should be rejected
        """
        return "proceed" if state['is_ecommerce_query'] else "reject"
    

    
    def chat(self, request: chatbot_request) -> chatbot_response:
        """
        Main chat interface for the agent.
        
        Args:
            request: chatbot_request with user message and history
            
        Returns:
            chatbot_response with generated response
        """
        # Convert history to LangChain messages
        messages = []
        for item in request.history or []:
            messages.append(HumanMessage(content=item.message))
            messages.append(AIMessage(content=item.response))
        
        # Initialize state
        initial_state: ChatbotState = {
            "messages": messages,
            "user_query": request.message,
            "user_id": request.user_id,
            "session_id": getattr(request, 'session_id', None),
            "is_ecommerce_query": False,
            "guardrail_reason": "",
            "retrieved_contexts": {},
            "final_response": "",
            "metadata": {}
        }
        
        # Execute the graph
        final_state = self.graph.invoke(initial_state)
        
        # Return response
        return chatbot_response(response=final_state['final_response'])


# Global agent instance
ecommerce_agent = EcommerceChatbotAgent()
