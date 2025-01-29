# agent/agent.py

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from utils.state import State  
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Callable, Literal, Optional, List
from uuid import uuid4
from datetime import datetime
import json
import uuid
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0))
STREAMING = os.getenv("STREAMING", "True").lower() in ("true", "1", "t")

llm = ChatOpenAI(temperature=TEMPERATURE, streaming=STREAMING, model=MODEL_NAME)

# Import utilities and tools
from utils.tools import (
    get_customer_info,
    get_user_info,
    update_customer_profile,
    get_albums_by_artist,
    get_tracks_by_artist,
    check_for_songs,
    artist_retriever,
    song_retriever,
)
from utils.nodes import (
    fetch_user_info,
    handle_tool_error,
    create_entry_node,
    pop_dialog_state,
    route_to_workflow,
)
from utils.state import State

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage
)

from langgraph.graph.message import add_messages

# Initialize Logger
from utils.logger import get_logger
logger = get_logger(__name__)

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)]
    )

# Define Assistants and Prompts

extra_customer_security = """
You will be given the customer data when the chats begins. Do not ask them for any clarification. 
Do not under any circumstance share customer information of another customer. 
IMPORTANT: If the customer wants to make any modification to their profile, you must ask them to confirm their current email and phone number.  
IMPORTANT: If the customer wants to inquire about the information that you have for them on file, you must ask them to confirm their current email and phone number.
                          """

# Customer Assistance Prompt
customer_assistance_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Your role is to assist a user in viewing or updating their profile information.

            You have access to the following tools to perform these tasks:
            1. `get_customer_info`: Retrieve information about a specific customer using their unique customer ID.
            - Required input: `customer_id`
            - Example: "What is your customer ID so I can retrieve your profile?"

            2. `update_customer_profile`: Update a specific field in the customer's profile.
            - Required inputs:
                - `customer_id`: The unique ID of the customer.
                - `field`: The name of the profile field to update (e.g., "FirstName", "Email", "Phone").
                - `new_value`: The new value to assign to the specified field.
            - Example: "Could you confirm your customer ID and provide the new email address you'd like to update?"

            Steps to assist the user:
            - Always verify you have all the required inputs before using a tool.
            - If a user wants to update a field (e.g., their email or phone number), ask for the relevant inputs:
                - For first name updates: Ask for the new first name.
                - For email updates: Ask for the new email address.
                - For phone updates: Ask for the new phone number.
                - For other fields: Clarify the field name and the new value.

            Example interactions:
            1. To retrieve profile information:
            - "What is your customer ID so I can look up your profile?"
            2. To update the profile:
            - "What is your customer ID, and which field would you like to update? Please provide the new value for that field."

            Important Notes:
            - If the user requests to update a field that is invalid or unavailable, inform them politely of the valid fields:
            - Allowed fields: `FirstName`, `LastName`, `Company`, `Address`, `City`, `State`, `Country`, `PostalCode`, `Phone`, `Fax`, `Email`, and `SupportRepId`.
            - If you are unable to assist with the user's request, politely suggest they contact customer support.
            """ + extra_customer_security
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

# Music Assistance Prompt
music_assistance_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Your job is to assist a customer in finding any songs they are looking for.

            You have access to specific tools to look up songs, albums, or artists. If a customer asks for something you cannot assist with, 
            politely inform them of the limitations and let them know what you can help with.

            When looking up artists and songs, sometimes the exact match may not be found. In such cases, the tools are designed to return 
            information about similar songs or artists. This is intentional and helps provide relevant recommendations.
            """ 
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

# Primary Assistant Prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Your job is to serve as a polite and helpful customer service representative for a music store.

            You can assist customers in the following ways:
            1. **Updating user information**:
                - If the customer wants to update or access their information in the user database, route them to `customer`.
            2. **Recommending music**:
                - If the customer wants to find music or learn about music, route them to `music`.

            Routing Instructions:
            - If the user mentions updating or accessing their personal information, call the router with 'customer'.
            - If the user mentions music recommendations or any music-related inquiry, call the router with 'music'.
            - For any other inquiries, respond politely and explain what you can assist with.

            Always aim to be polite and clear in your interactions.
            """
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

# Define CompleteOrEscalate Model
class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }

# Define Router Model
class Router(BaseModel):
    """
    A routing model responsible for determining the appropriate agent
    to handle user inquiries based on the context of their message.

    Attributes:
    ----------
    choice : str
        Specifies the agent to route to. Should be one of:
        - 'music' for inquiries related to music recommendations or music information.
        - 'customer' for inquiries related to updating or accessing user information.
    """
    choice: str = Field(description="Should be one of: 'music', 'customer'.")

# Define Transfer Models
class ToCustomerAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle customer profile queries."""

    request: str = Field(
        description="Any necessary follow-up questions to retrieve or update customer information should clarify before proceeding."
    )

class ToMusicAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle music-related queries."""

    request: str = Field(
        description="Any necessary follow-up questions to give information about music should clarify before proceeding."
    )

# Initialize Assistants
customer_tools = [get_customer_info, update_customer_profile]
customer_safe_tools = [get_customer_info]
customer_sensitive_tools = [update_customer_profile]
music_tools = [check_for_songs, get_tracks_by_artist, get_albums_by_artist]
primary_tools = [Router, ToCustomerAssistant, ToMusicAssistant]

customer_runnable = customer_assistance_prompt | llm.bind_tools(
    customer_tools + [CompleteOrEscalate]
)

music_runnable = music_assistance_prompt | llm.bind_tools(
    music_tools + [CompleteOrEscalate]
)

primary_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_tools + [ToMusicAssistant, ToCustomerAssistant]
)

# Define Assistant Classes
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
    
def user_info(state: State) -> State:
    """
    Retrieve a customer_id from state["configurable"]["customer_id"],
    call get_user_info with the correct 'RunnableConfig' signature,
    and store the result in state["user_info"].
    """
    return {"user_info": get_user_info.invoke({})}

# Build State Graph
def build_graph() -> StateGraph:
    builder = StateGraph(State)
    memory = MemorySaver()

    # Fetch User Info
    builder.add_node("fetch_user_info", user_info)
    builder.add_edge(START, "fetch_user_info")

    # Primary Assistant
    builder.add_node("primary_assistant", Assistant(primary_runnable))
    builder.add_node("primary_assistant_tools", create_tool_node_with_fallback(primary_tools))
    # builder.add_edge("fetch_user_info", "primary_assistant")

    builder.add_conditional_edges(
    "fetch_user_info", 
    route_to_workflow,  # A function dynamically deciding the next step
    ["primary_assistant", "music_assistant", "customer_assistant"]
    )


    def route_primary_assistant(
        state: State,
    ):
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        if tool_calls:
            if tool_calls[0]["name"] == ToCustomerAssistant.__name__:
                return "enter_customer_profile"
            elif tool_calls[0]["name"] == ToMusicAssistant.__name__:
                return "enter_music"
            return "primary_assistant_tools"
        raise ValueError("Invalid route")


    builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    [
        "enter_customer_profile",
        "enter_music",
        "primary_assistant_tools",
        "leave_skill",  # Ensure we can return to the primary assistant
        END,
    ],
)

    builder.add_edge("primary_assistant_tools", "primary_assistant")

    # Customer Profile Assistant
    builder.add_node("enter_customer_profile", create_entry_node("Customer Assistant", "customer_assistant"))
    builder.add_node("customer_assistant", Assistant(customer_runnable))
    builder.add_node("customer_safe_tools", create_tool_node_with_fallback(customer_safe_tools))
    builder.add_node("customer_sensitive_tools", create_tool_node_with_fallback(customer_sensitive_tools))
    builder.add_edge("enter_customer_profile", "customer_assistant")
    builder.add_edge("customer_safe_tools", "customer_assistant")
    builder.add_edge("customer_sensitive_tools", "customer_assistant")


    def route_customer_assistant(
        state: State,
    ):
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
        if did_cancel:
            return "leave_skill"
        safe_toolnames = [t.name for t in customer_safe_tools]
        if all(tc["name"] in safe_toolnames for tc in tool_calls):
            return "customer_safe_tools"
        return "customer_sensitive_tools"

    builder.add_conditional_edges(
        "customer_assistant",
        route_customer_assistant,
        ["customer_safe_tools", "customer_sensitive_tools", "leave_skill", END],
    )

    # Music Assistant
    builder.add_node("enter_music", create_entry_node("Music Assistant", "music_assistant"))
    builder.add_node("music_assistant", Assistant(music_runnable))
    builder.add_node("music_tools", create_tool_node_with_fallback(music_tools))
    builder.add_edge("enter_music", "music_assistant")
    builder.add_edge("music_tools", "music_assistant")


    def route_music_assistant(state: State):
        route = tools_condition(state)
        if route == END:
            return END
        
        tool_calls = state["messages"][-1].tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
        
        if did_cancel:
            return "leave_skill"

        safe_toolnames = [t.name for t in music_tools]
        
        # If only safe tools were called, return "music_tools"
        if all(tc["name"] in safe_toolnames for tc in tool_calls):
            return "music_tools"

        # Otherwise, introduce differentiation (future-proofing)
        return "music_tools"

  

    builder.add_conditional_edges(
        "music_assistant",
        route_music_assistant,
        ["music_tools", "leave_skill", END],
    )

    # Leave Skill
    builder.add_node("leave_skill", pop_dialog_state)
    builder.add_edge("leave_skill", "primary_assistant")


    # Compile Graph
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["customer_sensitive_tools"],
    )
    return graph


from IPython.display import Image

def save_graph_visualization(graph: StateGraph, save_path: str):
    """
    Save the graph visualization as a PNG file.
    
    Parameters:
    ----------
    graph : StateGraph
        The constructed state graph.
    save_path : str
        The file path to save the visualization PNG.
    """
    try:
        # Generate graph visualization as a PNG image
        graph_image = graph.get_graph(xray=True).draw_mermaid_png()
        with open(save_path, "wb") as f:
            f.write(graph_image)
        print(f"Graph visualization saved successfully at {save_path}")
    except Exception as e:
        print(f"Failed to save graph visualization: {str(e)}")

output_path = "/Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/docs/v2_graph_visualization.png"
graph = build_graph()
save_graph_visualization(graph, output_path)