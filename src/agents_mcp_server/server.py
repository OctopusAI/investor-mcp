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

from agents_mcp_server.cli import octagon_client

FRED_WILSON_PROFILE = (Path(__file__).parent / "investors/fred_wilson.md").read_text()
PETER_THIEL_PROFILE = (Path(__file__).parent / "investors/peter_thiel.md").read_text()
COMPANY_REPORT_SHEET = (Path(__file__).parent / "reports/company_sheet.md").read_text()

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


# --- Base Agents ---
fred_wilson_agent = Agent(
    name="Fred Wilson (Union Square Ventures)",
    instructions=FRED_WILSON_PROFILE,
    tools=[],
)

pieter_thiel_agent = Agent(
    name="Peter Thiel (Founders Fund)",
    instructions=PETER_THIEL_PROFILE,
    tools=[],
)

companies_agent = Agent(
    name="Companies Agent",
    instructions="Retrieve detailed company information from Octagon's companies database.",
    model=OpenAIResponsesModel(model="octagon-companies-agent", openai_client=octagon_client),
    tools=[],
)

funding_agent = Agent(
    name="Octagon Funding Agent",
    instructions="Retrieve detailed funding information from Octagon's companies database.",
    model=OpenAIResponsesModel(model="octagon-funding-agent", openai_client=octagon_client),
    tools=[],
)

company_report_agent = Agent(
    name="Company Report Agent",
    instructions=f"Retrieve detailed company information from Octagon's companies database. Fill in the company report sheet: {COMPANY_REPORT_SHEET}",
    tools=[],
)


# --- Shared Report Workflow ---
async def generate_final_company_report(query: str) -> str:
    """Handles company data, funding data, and generates the final report."""
    # Step 1: Company data
    with trace("Company research"):
        company_result = await Runner.run(companies_agent, query)
        company_data = company_result.final_output

    # Step 2: Initial report from company data
    with trace("Initial company report"):
        initial_prompt = f"""
Use the following company research data to fill in a structured company report.

{COMPANY_REPORT_SHEET}

Company Research Data:
{company_data}
"""
        initial_result = await Runner.run(company_report_agent, initial_prompt)
        partial_report = initial_result.final_output

    # Step 3: Funding data
    with trace("Funding research"):
        funding_result = await Runner.run(funding_agent, query)
        funding_data = funding_result.final_output

    # Step 4: Final report with funding info
    with trace("Final company report with funding"):
        final_prompt = f"""
You are provided with a partially completed company report and new funding data.

Update and complete the report using this additional information:

--- Existing Report ---
{partial_report}

--- Funding Data ---
{funding_data}

Use the following template for consistency:

{COMPANY_REPORT_SHEET}
"""
        final_result = await Runner.run(company_report_agent, final_prompt)
        return final_result.final_output


# --- Fred Wilson MCP Tool ---
@mcp.tool(
    name="fred_wilson_agent",
    description="Get investment insights from Fred Wilson of Union Square Ventures.",
)
async def fred_wilson_agent_tool(
    query: str = Field(..., description="The investment-related question or query."),
) -> AgentResponse:
    try:
        final_report = await generate_final_company_report(query)

        agent = Agent(
            name="Fred Wilson Proxy",
            instructions=f"""You are Fred Wilson from Union Square Ventures.

Respond to the investment analysis request below using your unique investment philosophy, voice, and portfolio experience.

Company Report:
{final_report}

Investment Analysis Request:
{query}

Guidelines:
1. Focus on community, NYC startups, and platforms.
2. Use examples from your known investments.
3. Be candid and concise like Fred Wilson's blog style.
4. Say you're Fred Wilson.""",
            tools=[],
        )

        with trace("Fred Wilson analysis"):
            result = await Runner.run(agent, query)

        return AgentResponse(
            response=result.final_output,
            raw_response={
                "investor": "Fred Wilson",
                "final_report": final_report,
                "items": [str(item) for item in result.new_items],
            },
        )

    except Exception as e:
        print(f"Fred Wilson error: {e}")
        return AgentResponse(response=f"Error: {str(e)}", raw_response=None)


# --- Peter Thiel MCP Tool ---
@mcp.tool(
    name="peter_thiel_agent",
    description="Get investment insights from Peter Thiel of Founders Fund.",
)
async def peter_thiel_agent_tool(
    query: str = Field(..., description="The investment-related question or query."),
) -> AgentResponse:
    try:
        final_report = await generate_final_company_report(query)

        agent = Agent(
            name="Peter Thiel Proxy",
            instructions=f"""You are Peter Thiel from Founders Fund.

Respond to the investment analysis request below using your known contrarian philosophy, voice, and zero-to-one mindset.

Company Report:
{final_report}

Investment Analysis Request:
{query}

Guidelines:
1. Focus on breakthrough innovation, monopoly building, and long-term vision.
2. Use references to PayPal, Palantir, and other examples.
3. Be bold, logical, and provocative like Peter.
4. Say you're Peter Thiel.""",
            tools=[],
        )

        with trace("Peter Thiel analysis"):
            result = await Runner.run(agent, query)

        return AgentResponse(
            response=result.final_output,
            raw_response={
                "investor": "Peter Thiel",
                "final_report": final_report,
                "items": [str(item) for item in result.new_items],
            },
        )

    except Exception as e:
        print(f"Peter Thiel error: {e}")
        return AgentResponse(response=f"Error: {str(e)}", raw_response=None)
