# utils/nodes.py

from typing import Callable, Optional
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from utils.tools import get_customer_info, update_customer_profile
from utils.state import State  
from typing import Literal



def fetch_user_info(state: State) -> dict:
    """
    Retrieve user information and update the state.
    """
    customer_id = state.get("configurable", {}).get("customer_id")
    if not customer_id:
        return {"error": "No customer_id provided."}
    
    user_info = get_customer_info.invoke(RunnableConfig(configurable={"customer_id": customer_id}))
    return {"user_info": user_info}

def handle_tool_error(state: State) -> dict:
    """
    Handle errors from tool invocations.
    """
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease fix your request.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=(
                        f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                        f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                        " and any other action is not complete until after you have successfully invoked the appropriate tool."
                        " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                        " Do not mention who you are - just act as the proxy for the assistant."
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }
    return entry_node

def pop_dialog_state(state: State) -> dict:
    """
    Pop the dialog stack and return to the main assistant.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        messages.append(
            ToolMessage(
                content=(
                    "Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed."
                ),
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }

def route_to_workflow(state: State) -> Literal["primary_assistant", "music", "customer_assistant"]:
    """
    Route to the appropriate workflow based on the current dialog state.
    """
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]

