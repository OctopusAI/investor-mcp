"""
Octagon Investor MCP server for OpenAI agents tools.

This module provides a FastMCP server that exposes investor persona agents through the Model Context Protocol.
"""

import asyncio
from typing import Any, Dict, Optional
from pathlib import Path

from agents import Agent, Runner, trace, OpenAIResponsesModel
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


FRED_WILSON_PROFILE = (Path(__file__).parent / "investors/fred_wilson.md").read_text()
PETER_THIEL_PROFILE = (Path(__file__).parent / "investors/peter_thiel.md").read_text()

mcp = FastMCP(
    name="OpenAI Agents",
    instructions="""This MCP server provides access to Investor agents through the Model Context Protocol.""",
)


class AgentResponse(BaseModel):
    """Response from an OpenAI agent."""

    response: str = Field(..., description="The response from the agent")
    raw_response: Optional[Dict[str, Any]] = Field(
        None, description="The raw response data from the agent, if available"
    )


# --- Fred Wilson Agent ---
fred_wilson_agent = Agent(
    name="Fred Wilson (Union Square Ventures)",
    instructions=FRED_WILSON_PROFILE,
    tools=[],
)


# --- Peter Thiel Agent ---
pieter_thiel_agent = Agent(
    name="Peter Thiel (Founders Fund)",
    instructions=PETER_THIEL_PROFILE,
    tools=[],
)



# --- Updated Investor Persona Agent ---
@mcp.tool(
    name="investor_persona_agent",
    description="Consult with either Fred Wilson or Peter Thiel (choose one at a time) for investment insights.",
)
async def investor_persona_agent(
    query: str = Field(..., description="The investment-related question or query."),
    talk_to_fred: bool = Field(False, description="Set to TRUE to consult Fred Wilson"),
    talk_to_peter: bool = Field(False, description="Set to TRUE to consult Peter Thiel"),
) -> AgentResponse:
    """Use a specialized investor persona agent that delegates to a single investor personality."""
    try:
        # Validate only one investor is selected
        if sum([talk_to_fred, talk_to_peter]) != 1:
            return AgentResponse(
                response="Error: Please select exactly one investor to consult",
                raw_response={"error": "Invalid investor selection"}
            )

        # First get company research data
      

        # Determine selected investor and prepare analysis query
        selected_investor = None
        tools = []
       

        # Determine selected investor
        selected_investor = None
        tools = []
        
        if talk_to_fred:
            tools.append(
                fred_wilson_agent.as_tool(
                    tool_name="fred_wilson",
                    tool_description="Fred Wilson's perspective on community-focused ventures and NYC startups"
                )
            )
            selected_investor = "Fred Wilson (Union Square Ventures)"
            
        elif talk_to_peter:
            tools.append(
                pieter_thiel_agent.as_tool(
                    tool_name="peter_thiel",
                    tool_description="Peter Thiel's perspective on disruptive technology and zero-to-one innovations"
                )
            )
            selected_investor = "Peter Thiel (Founders Fund)"

        investor_agent = Agent(
            name=f"{selected_investor} Proxy",
            instructions=f"""You are exclusively channeling {selected_investor}. 
                Respond to the user's query directly in {selected_investor.split(' ')[0]}'s voice and style.
                
                Guidelines:
                1. Stay strictly within the investor's known philosophy
                2. Use historical examples from their investment portfolio
                3. Maintain their characteristic communication style
                4. Clearly state you're speaking as {selected_investor}""",
            tools=tools,
        )

        with trace("Single investor agent execution"):
            result = await Runner.run(investor_agent, query)

        return AgentResponse(
            response=result.final_output,
            raw_response={"investor": selected_investor, "items": [str(item) for item in result.new_items]},
        )

    except Exception as e:
        print(f"Error running investor persona agent: {e}")
        return AgentResponse(
            response=f"An error occurred while processing your request: {str(e)}", 
            raw_response=None
        )