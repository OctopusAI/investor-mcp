"""
Octagon Investor MCP server with handoff-driven chain-of-thought orchestration.

Fred Wilson and Peter Thiel orchestrate their analysis using tool handoffs to domain-specific agents,
with comparison capabilities and traceable reasoning.
"""

import asyncio
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass

from agents import (
    Agent,
    Runner,
    trace,
    handoff,
    RunHooks,
    RunContextWrapper,
    OpenAIResponsesModel,
)
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from agents_mcp_server.cli import octagon_client

# --- Load Instruction Profiles ---
FRED_WILSON_PROFILE = (Path(__file__).parent / "investors/fred_wilson.md").read_text()
PETER_THIEL_PROFILE = (Path(__file__).parent / "investors/peter_thiel.md").read_text()

# --- Shared Context (no memory) ---
@dataclass
class InvestorContext:
    query: str
    user_id: Optional[str] = None

# --- MCP Initialization ---
mcp = FastMCP(
    name="OpenAI Agents",
    instructions="""This MCP server provides access to Investor agents through the Model Context Protocol.""",
)

# --- Response Schema ---
class AgentResponse(BaseModel):
    response: str = Field(..., description="The response from the agent")
    raw_response: Optional[Dict[str, Any]] = Field(
        None, description="The raw response data from the agent, if available"
    )

# --- Logging Hooks ---
class InvestorHooks(RunHooks[InvestorContext]):
    async def on_handoff(self, context: RunContextWrapper[InvestorContext], from_agent, to_agent):
        print(f"🔁 Handoff from {from_agent.name} to {to_agent.name}")

    async def on_agent_start(self, context: RunContextWrapper[InvestorContext], agent: Agent):
        print(f"🚀 Starting agent: {agent.name}")

    async def on_agent_end(self, context: RunContextWrapper[InvestorContext], agent: Agent, output: Any):
        print(f"✅ Finished agent: {agent.name}")


# --- Domain-Specific Agents ---
companies_agent = Agent(
    name="Companies Agent",
    instructions="Retrieve detailed company information from Octagon's companies database.",
    model=OpenAIResponsesModel(model="octagon-companies-agent", openai_client=octagon_client),
)

funding_agent = Agent(
    name="Funding Agent",
    instructions="Retrieve detailed funding information from Octagon's companies database.",
    model=OpenAIResponsesModel(model="octagon-funding-agent", openai_client=octagon_client),
)

investors_agent = Agent(
    name="Investors Agent",
    instructions="Retrieve detailed investor information from Octagon's investors database.",
    model=OpenAIResponsesModel(model="octagon-investors-agent", openai_client=octagon_client),
)


# --- Handoff Tool Definitions ---
company_handoff = handoff(
    agent=companies_agent,
    tool_name_override="transfer_to_company_agent",
    tool_description_override="Handoff to get company insight",
)

funding_handoff = handoff(
    agent=funding_agent,
    tool_name_override="transfer_to_funding_agent",
    tool_description_override="Handoff to get funding insight",
)

investor_handoff = handoff(
    agent=investors_agent,
    tool_name_override="transfer_to_investor_agent",
    tool_description_override="Handoff to get investor insight",
)


# --- Fred Wilson Orchestrator ---
@mcp.tool(
    name="fred_wilson_orchestrator",
    description="Fred Wilson's investment assistant. Provides analysis and answers using Fred's investment philosophy.",
)
async def fred_wilson_orchestrator(
    query: str = Field(..., description="The investment-related question or query."),
) -> AgentResponse:
    try:
        context = InvestorContext(query=query)

        fred_agent = Agent[InvestorContext](
            name="Fred Wilson Orchestrator",
            instructions=f"""You are Fred Wilson, a community-focused investor known for supporting NYC startups.

Follow this process:
1. Understand the user query.
2. Think step-by-step.
3. Use tools to research company, funding, or investor info as needed.
4. Once all info is gathered, give your final investment opinion in your unique style.""",
            tools=[company_handoff, funding_handoff, investor_handoff],
            hooks=InvestorHooks(),
        )

        with trace("Fred Wilson orchestrator execution"):
            result = await Runner.run(
                starting_agent=fred_agent,
                input=query,
                context=context,
            )

        return AgentResponse(
            response=result.final_output,
            raw_response={"source": "Fred Wilson", "items": [str(item) for item in result.new_items]},
        )

    except Exception as e:
        print(f"Error in Fred Wilson Orchestrator: {e}")
        return AgentResponse(
            response=f"An error occurred while processing Fred's analysis: {str(e)}",
            raw_response=None,
        )


# --- Peter Thiel Orchestrator ---
@mcp.tool(
    name="peter_thiel_orchestrator",
    description="Peter Thiel's investment assistant. Provides analysis and answers using Peter's investment philosophy.",
)
async def peter_thiel_orchestrator(
    query: str = Field(..., description="The investment-related question or query."),
) -> AgentResponse:
    try:
        context = InvestorContext(query=query)

        peter_agent = Agent[InvestorContext](
            name="Peter Thiel Orchestrator",
            instructions=f"""You are Peter Thiel, a visionary investor known for contrarian, zero-to-one thinking.

Approach:
1. Break down the query.
2. Consider what information is needed to form a bold thesis.
3. Use the tools to gather data on company, funding, or investors.
4. Give your view, staying true to your distinct analytical style.""",
            tools=[company_handoff, funding_handoff, investor_handoff],
            hooks=InvestorHooks(),
        )

        with trace("Peter Thiel orchestrator execution"):
            result = await Runner.run(
                starting_agent=peter_agent,
                input=query,
                context=context,
            )

        return AgentResponse(
            response=result.final_output,
            raw_response={"source": "Peter Thiel", "items": [str(item) for item in result.new_items]},
        )

    except Exception as e:
        print(f"Error in Peter Thiel Orchestrator: {e}")
        return AgentResponse(
            response=f"An error occurred while processing Peter's analysis: {str(e)}",
            raw_response=None,
        )


# --- Compare Investor Opinions Tool ---
@mcp.tool(
    name="compare_investors",
    description="Ask both Fred Wilson and Peter Thiel the same investment question and compare their perspectives.",
)
async def compare_investors(
    query: str = Field(..., description="The investment-related question or scenario."),
) -> AgentResponse:
    try:
        context = InvestorContext(query=query)

        with trace("Run Fred Wilson"):
            fred_result = await Runner.run(
                starting_agent=Agent[InvestorContext](
                    name="Fred Wilson",
                    instructions=FRED_WILSON_PROFILE,
                    tools=[company_handoff, funding_handoff, investor_handoff],
                    hooks=InvestorHooks(),
                ),
                input=query,
                context=context,
            )

        with trace("Run Peter Thiel"):
            peter_result = await Runner.run(
                starting_agent=Agent[InvestorContext](
                    name="Peter Thiel",
                    instructions=PETER_THIEL_PROFILE,
                    tools=[company_handoff, funding_handoff, investor_handoff],
                    hooks=InvestorHooks(),
                ),
                input=query,
                context=context,
            )

        with trace("Synthesizing comparison"):
            combined_summary = f"""📊 **Fred Wilson's Perspective:**
{fred_result.final_output}

🚀 **Peter Thiel's Perspective:**
{peter_result.final_output}

🤔 **Key Differences:**
- Fred may focus more on community, network effects, and long-term viability.
- Peter may emphasize breakthrough tech, monopoly potential, and contrarian theses.
"""

        return AgentResponse(
            response=combined_summary,
            raw_response={
                "fred": fred_result.final_output,
                "peter": peter_result.final_output,
            },
        )

    except Exception as e:
        print(f"Error comparing investors: {e}")
        return AgentResponse(
            response=f"An error occurred during investor comparison: {str(e)}",
            raw_response=None,
        )
