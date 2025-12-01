"""Agent utility functions: nodes and tools."""

from typing import Dict, Any
from agent.config import AgentState, AGENT_SYSTEM_PROMPT, REPORT_TEMPLATE


# ===== Nodes =====
def consolidate_results(state: AgentState) -> AgentState:
    """Consolidate model results for agent analysis.
    
    This deterministic node extracts and formats:
    - Training metrics (convergence, iterations, log-likelihood)
    - Model results (factors, loadings, smoothed data)
    - Statistics (RMSE, factor contributions)
    - Forecasts and nowcasts
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with consolidated results
    """
    # Placeholder implementation
    # Will be fully implemented when model results are available
    
    consolidated = {
        "metrics": state.get("metrics", {}),
        "model_results": state.get("model_results", {}),
        "factors": state.get("factors"),
        "forecasts": state.get("forecasts"),
        "statistics": state.get("statistics", {})
    }
    
    state["model_results"] = consolidated
    
    return state


# ===== Tools =====
def search_tavily(query: str) -> str:
    """Search Tavily for research context (placeholder).
    
    Args:
        query: Search query
        
    Returns:
        Search results as string
    """
    # Placeholder - will be implemented when Tavily integration is added
    return f"Tavily search results for: {query} (not yet implemented)"


def generate_report_tool(state: AgentState) -> AgentState:
    """Generate report from consolidated results.
    
    Args:
        state: Agent state with consolidated results
        
    Returns:
        Updated state with generated report
    """
    # Placeholder implementation
    # Will use LLM to generate report from consolidated results
    
    results = state.get("model_results", {})
    
    # Generate placeholder report
    report = REPORT_TEMPLATE.format(
        summary="Model analysis completed",
        metrics=str(results.get("metrics", {})),
        factors=str(results.get("factors", {})),
        nowcasts="Nowcast results available",
        forecasts="Forecast results available",
        statistics=str(results.get("statistics", {})),
        context="Additional context from research"
    )
    
    # Add report to state
    state["report"] = report
    
    # Add response message
    from langchain_core.messages import AIMessage
    if "messages" not in state:
        state["messages"] = []
    state["messages"].append(AIMessage(content=report))
    
    return state


