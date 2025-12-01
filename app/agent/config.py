"""Agent configuration: state, schemas, and prompts."""

from typing import TypedDict, Annotated, Optional, Dict, Any, List
from pydantic import BaseModel

# ===== State Schema =====
class AgentState(TypedDict):
    """Agent state schema for LangGraph."""
    messages: Annotated[List[Any], "Chat messages"]
    model_results: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]
    factors: Optional[Any]
    forecasts: Optional[Any]
    statistics: Optional[Dict[str, Any]]


# ===== Input/Output Schemas =====
class AgentInput(BaseModel):
    """Input schema for agent."""
    query: str
    model_name: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class AgentOutput(BaseModel):
    """Output schema for agent."""
    response: str
    report: Optional[str] = None  # Markdown report content
    metadata: Optional[Dict[str, Any]] = None


# ===== Prompts =====
AGENT_SYSTEM_PROMPT = """You are a research assistant specialized in analyzing dynamic factor models (DFM) and deep dynamic factor models (DDFM) for macroeconomic nowcasting.

Your role is to:
1. Analyze model results including factors, forecasts, and nowcasts
2. Interpret training metrics and convergence information
3. Generate clear, concise reports based on the analysis
4. Provide insights about economic indicators and their relationships

When generating reports, include:
- Executive summary of key findings
- Model performance metrics
- Factor analysis and interpretation
- Nowcast and forecast results
- Statistical summaries
- Relevant context from web research (when available)

Be objective, data-driven, and focus on actionable insights."""

REPORT_TEMPLATE = """# Nowcasting Analysis Report

## Executive Summary
{summary}

## Model Performance
{metrics}

## Factor Analysis
{factors}

## Nowcast Results
{nowcasts}

## Forecast Results
{forecasts}

## Statistical Summary
{statistics}

## Additional Context
{context}
"""


