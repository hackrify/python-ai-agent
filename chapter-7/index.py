from typing import Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio


# Define a tool that searches the web for information.
async def search_web_tool(query: str) -> str:
    """Find information on the web"""
    return "Charminar, golkonda fort, paradise biryani"

async def search_places(query: str) -> str:
    return "Charminar, golkonda fort, paradise biryani"

planning_agent = AssistantAgent(
    "TripPlanningAgent",
    description="An trip planner agent for planning trips, this agent should be the first to engage when given a new task.",
    system_message="""
    You are a trip planning agent.
    Your job is to break down a trip plan into day wise intenary.
    Your team members are:
        PlacesSearchAgent: Search for information for places,aveage time required to explore place, GPS location of place and prepare a list 
        IntenaryBuildAgent : Plan day wise itenary for places from Web Search Agent
    
    After all tasks are complete, summarize the findings and end with "TERMINATE".
    """,
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini",
        api_key="Open AI Key"
        )
    
)


Places_serach_agent = AssistantAgent(
    "PlacesSearchAgent",
    description="You are preparing a list of places in city",
    system_message="""
    You are places search agent, you job is to create list of places available on internet, average number of hours required to visit this place and GPS location of places. 
    Stack rank the places in order of high significace at top, add famous restaurants in list as well. 
    
    When creating list of places, use this format
    1. Place, average number of hours required to explore place, gps location of place

    """,
    
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini",
        api_key="Open AI Key"
        )
)


internary_build_agent = AssistantAgent(
    "IntenaryBuildAgent",
    description="You are internary planner agent, You are planning a iternary based on user preferences.",
    system_message="""
    You are itenary planner agent, you take list of places as input, and planning a day wise itenary.  
    Add transit time as well into consideration from one place to another.
    Use GPS information and optimize intenary to reduce travel time.
    """,
    tools=[search_web_tool],
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini",
        api_key="Open AI Key"
        )
)


async def main() -> None:
    # Define an agent

    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination | max_messages_termination
   
    while True:
        # Get user input from the console.
        user_input = input("Enter a message (type 'exit' to leave): ")
        if user_input.strip().lower() == "exit":
            break
        # Run the team and stream messages to the console.
        team = SelectorGroupChat(
        [planning_agent, Places_serach_agent, internary_build_agent],
        termination_condition=termination,
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini",
        api_key="Open AI Key"
        ),

        )

        await Console(team.run_stream(task=user_input))


asyncio.run(main())
