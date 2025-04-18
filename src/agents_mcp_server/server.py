import asyncio
from typing import Any, Dict, Optional
from pathlib import Path

from agents import Agent, Runner, trace, OpenAIResponsesModel
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from agents_mcp_server.cli import octagon_client

# Load investor profiles
FRED_WILSON_PROFILE = (Path(__file__).parent / "investors/fred_wilson.md").read_text()
PETER_THIEL_PROFILE = (Path(__file__).parent / "investors/peter_thiel.md").read_text()

# Initialize the MCP server
mcp = FastMCP(
    name="OpenAI Agents",
    instructions="This MCP server provides access to Investor agents through the Model Context Protocol.",
)

# Define the companies agent
octagon_companies_agent = Agent(
    name="Companies Agent",
    instructions="Retrieve detailed company information from Octagon's companies database.",
    model=OpenAIResponsesModel(model="octagon-companies-agent", openai_client=octagon_client),
    tools=[],
)

# Define the investors info agent
octagon_investors_agent = Agent(
    name="Investors Info Agent",
    instructions="Provides information about other investors and their investment history.",
    model=OpenAIResponsesModel(model="octagon-investors-agent", openai_client=octagon_client),
    tools=[],
)

# Define the funding info agent
octagon_funding_agent = Agent(
    name="Funding Info Agent",
    instructions="Provides information about funding rounds and financial data.",
    model=OpenAIResponsesModel(model="octagon-funding-agent", openai_client=octagon_client),
    tools=[],
)

# Define the Fred Wilson agent with access to the additional agents
fred_wilson_agent = Agent(
    name="Fred Wilson (Union Square Ventures)",
    instructions=FRED_WILSON_PROFILE,
    model=OpenAIResponsesModel(model="octagon-investors-agent", openai_client=octagon_client),
    tools=[
        octagon_companies_agent.as_tool(
            tool_name="octagon_companies_agent",
            tool_description="Provides detailed company information from Octagon's database."
        ),
        octagon_investors_agent.as_tool(
            tool_name="octagon_investors_agent",
            tool_description="Provides information about other investors and their investment history."
        ),
        octagon_funding_agent.as_tool(
            tool_name="octagon_funding_agent",
            tool_description="Provides information about funding rounds and financial data."
        ),
    ],
)

# Define the Peter Thiel agent with access to the additional agents
peter_thiel_agent = Agent(
    name="Peter Thiel (Founders Fund)",
    instructions=PETER_THIEL_PROFILE,
    model=OpenAIResponsesModel(model="octagon-investors-agent", openai_client=octagon_client),
    tools=[
        octagon_companies_agent.as_tool(
            tool_name="octagon_companies_agent",
            tool_description="Provides detailed company information from Octagon's database."
        ),
        octagon_investors_agent.as_tool(
            tool_name="octagon_investors_agent",
            tool_description="Provides information about other investors and their investment history."
        ),
        funding_info_agent.as_tool(
            tool_name="funding_info_agent",
            tool_description="Provides information about funding rounds and financial data."
        ),
    ],
)


# Define the response model
class AgentResponse(BaseModel):
    response: str = Field(..., description="The response from the agent")
    raw_response: Optional[Dict[str, Any]] = Field(
        None, description="The raw response data from the agent, if available"
    )

# Define the investor persona agent tool
@mcp.tool(
    name="investor_persona_agent",
    description="Consult with either Fred Wilson or Peter Thiel (choose one at a time) for investment insights.",
)
async def investor_persona_agent(
    query: str = Field(..., description="The investment-related question or query."),
    talk_to_fred: bool = Field(False, description="Set to TRUE to consult Fred Wilson"),
    talk_to_peter: bool = Field(False, description="Set to TRUE to consult Peter Thiel"),
) -> AgentResponse:
    try:
        # Validate only one investor is selected
        if sum([talk_to_fred, talk_to_peter]) != 1:
            return AgentResponse(
                response="Error: Please select exactly one investor to consult",
                raw_response={"error": "Invalid investor selection"}
            )

        # Determine selected investor
        selected_agent = fred_wilson_agent if talk_to_fred else peter_thiel_agent
        selected_investor = selected_agent.name

        # Run the selected investor agent with the query
        with trace(f"{selected_investor} agent execution"):
            result = await Runner.run(selected_agent, query)

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
