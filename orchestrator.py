import asyncio
import json
from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

from pydantic import BaseModel, Field

# Define our shared state structure
class AgentState(TypedDict):
    user_query: str
    route_decision: List[str]  # Which agents to call
    market_data: str
    fundamental_data: str
    news_data: str
    final_response: str


class Route(BaseModel):
    step: Literal["market_data", "news_analysis", "fundamental_analysis"] = Field(
        None, description="The next step in the routing process"
    )


llm = Ollama(
    model="qwen3",  # Qwen 2.5 model
    base_url="http://localhost:11434"  # default Ollama endpoint
)




class FinancialOrchestrator:
    def __init__(self):
        """Initialize the orchestrator with an LLM client (our Ollama client)"""
        self.router = llm.with_structured_output(Route)
        self.market_data_agent = None  # Will connect to our existing MCP client
        self.graph = self._build_graph()
    
    def set_market_data_agent(self, agent):
        """Inject our existing MCP client as the market data agent"""
        self.market_data_agent = agent
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._route_query)
        workflow.add_node("market_data", self._get_market_data)
        workflow.add_node("fundamental_analysis", self._get_fundamentals)
        workflow.add_node("news_analysis", self._get_news)
        workflow.add_node("synthesizer", self._synthesize_response)
        
        # Define the start node (router)
        workflow.set_entry_point("router")
        
        # Router decides which agents to call
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "market_data": "market_data",
                "news_analysis": "news_analysis",  # Start with market data for comprehensive analysis
                "fundamental_analysis": "fundamental_analysis"
            }
        )
        workflow.add_edge("market_data","synthesiser")
        workflow.add_edge("news_analysis","synthesiser")
        workflow.add_endge("fundamental_analysis","synthesiser")

        workflow.add_edge("synthesizer", END)
        workflow = workflow.compile()
        display(Image(workflow.get_graph().draw_mermaid_png()))
        return workflow.compile()
    
    async def _route_query(self, state: AgentState) -> Dict[str, Any]:
        """Analyze the user query and decide which agents to call"""
        
        route_decision = self.router.invoke(
            [
            SystemMessage(
                content="Route the input to market analysis, news analysis, or fundamental analysis based on the user input"
            ),
            HumanMessage(content=state["user_query"]),
        ]
        )
        
        print(f"Router decision: {route_decision.step}")
        return {"route_decision": route_decision.step}
    
    def _route_decision(self, state: AgentState) -> str:
        """Determine the routing path based on route decision"""
        routes = state["route_decision"]
        if routes == "market_analysis":
            return "_get_market_data"
        elif routes == "news_analysis":
            return "unknown"
        elif routes == "fundamental_analysis":
            return "unknown"
        else:
            return "_get_market_data"
        
    
    async def _get_market_data(self, state: AgentState) -> Dict[str, Any]:
        """Get market data using our existing MCP client"""
        if not self.market_data_agent:
            return {"market_data": "Market data agent not available"}
        
        try:
            # Extract stock symbol from query (simple approach)
            query = state["user_query"]
            symbol = self._extract_stock_symbol(query)
            
            if symbol:
                # Use our existing MCP client to get stock data
                result = await self.market_data_agent.process_query(f"Get stock quote for {symbol}")
                print(f"Market data result: {result}")
                return {"market_data": result}
            else:
                return {"market_data": "No stock symbol found in query"}
        except Exception as e:
            return {"market_data": f"Error getting market data: {str(e)}"}
    
       
    async def _get_fundamentals(self, state: AgentState) -> Dict[str, Any]:
        """Get fundamental analysis (placeholder for now)"""
        # TODO: Implement fundamental analysis agent/MCP server
        symbol = self._extract_stock_symbol(state["user_query"])
        return {"fundamental_data": f"Fundamental analysis for {symbol} - placeholder (P/E ratio, earnings, etc.)"}
    
    async def _get_news(self, state: AgentState) -> Dict[str, Any]:
        """Get news and sentiment analysis (placeholder for now)"""
        # TODO: Implement news analysis agent/MCP server
        symbol = self._extract_stock_symbol(state["user_query"])
        return {"news_data": f"Recent news for {symbol} - placeholder (sentiment analysis, recent events)"}
    
    async def _synthesize_response(self, state: AgentState) -> Dict[str, Any]:
        """Combine all agent results into a coherent response"""
        # Prepare context for the LLM
        context_parts = []
        
        if state.get("market_data"):
            context_parts.append(f"Market Data: {state['market_data']}")
        
        if state.get("fundamental_data"):
            context_parts.append(f"Fundamental Analysis: {state['fundamental_data']}")
        
        if state.get("news_data"):
            context_parts.append(f"News & Sentiment: {state['news_data']}")
        
        context = "\n\n".join(context_parts)
        
        # Use our Ollama client to synthesize the response
        synthesis_prompt = f"""
        User Question: {state['user_query']}
        
        Available Information:
        {context}
        
        Please provide a comprehensive, helpful response that synthesizes all the available information to answer the user's question about this stock/financial topic.
        """
        
        try:
            # This would use our existing Ollama client
            response = await self._call_llm_for_synthesis(synthesis_prompt)
            return {"final_response": response}
        except Exception as e:
            return {"final_response": f"Error synthesizing response: {str(e)}"}
    
    def _extract_stock_symbol(self, query: str) -> str:
        """Simple stock symbol extraction"""
        import re
        # Look for 1-5 uppercase letters (common stock symbol pattern)
        symbols = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        
        # Common stock symbols to prioritize
        common_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA']
        
        for symbol in symbols:
            if symbol in common_symbols:
                return symbol
        
        # Return first found symbol if no common ones
        return symbols[0] if symbols else "AAPL"  # Default for testing
    
    async def _call_llm_for_synthesis(self, prompt: str) -> str:
        """Call our LLM for response synthesis"""
        # This would integrate with our existing Ollama client
        # For now, placeholder
        return f"Synthesized response based on: {prompt[:100]}..."
    
    async def process_query(self, user_query: str) -> str:
        """Main entry point - process a user query through the workflow"""
        initial_state = AgentState(
            user_query=user_query,
            route_decision=[],
            market_data="",
            fundamental_data="",
            news_data="",
            final_response=""
        )
        
        try:
            result = await self.graph.ainvoke(initial_state)
            return result.get("final_response", "No response generated")
        except Exception as e:
            return f"Error processing query: {str(e)}"

# Example usage function
async def main():
    """Example of how to use the orchestrator"""
    
    # This would be our existing Ollama client
    class MockLLMClient:
        pass
    
    router = MockLLMClient()
    orchestrator = FinancialOrchestrator(router)
    
    # We would inject our existing MCP client here
    # orchestrator.set_market_data_agent(our_existing_mcp_client)
    
    # Test queries
    test_queries = [
        "What's the current price of AAPL?",
        "Should I buy Tesla stock?",
        "Give me an analysis of Microsoft"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = await orchestrator.process_query(query)
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())