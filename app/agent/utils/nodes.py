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
    Retrieves user information based on the `customer_id` stored in the state.

    This function extracts the `customer_id` from the configurable state parameters
    and invokes a function to fetch user details. If no `customer_id` is provided,
    an error message is returned.

    Args:
        state (State): The current state of the conversation, containing
                       configurable parameters such as `customer_id`.

    Returns:
        dict: A dictionary containing either the user information under
              the key `"user_info"` or an error message if no `customer_id` is found.
    """
    customer_id = state.get("configurable", {}).get("customer_id")
    if not customer_id:
        return {"error": "No customer_id provided."}

    user_info = get_customer_info.invoke(
        RunnableConfig(configurable={"customer_id": customer_id})
    )
    return {"user_info": user_info}


def handle_tool_error(state: State) -> dict:
    """
    Handles errors from tool invocations and generates error messages.

    This function retrieves the latest tool calls from the conversation state and
    constructs error messages for each failed tool invocation. The messages instruct
    the user to fix their request.

    Args:
        state (State): The current state of the conversation, containing
                       error details and tool call history.

    Returns:
        dict: A dictionary containing formatted error messages, each linked
              to the corresponding tool call ID.
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
    """
    Creates an entry node function for transitioning to a new assistant.

    This function generates a callable node that constructs system messages
    guiding the assistant's behavior when transitioning from the primary assistant
    to a specialized assistant (e.g., Customer Assistant, Music Assistant).

    The assistant is instructed to:
    - Act as the assigned `assistant_name` and use appropriate tools.
    - Avoid revealing its identity and instead serve as a proxy.
    - Use `CompleteOrEscalate` if the user’s request changes.

    Args:
        assistant_name (str): The name of the assistant to transition into.
        new_dialog_state (str): The state identifier for the new assistant.

    Returns:
        Callable: A function that, when executed, returns a dictionary containing:
                  - A system message instructing the assistant on how to behave.
                  - The updated dialog state.
    """

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
    Pops the current dialog state, returning control to the primary assistant.

    This function removes the most recent dialog state from the stack and
    resumes interaction with the primary assistant. If the latest message
    contains tool calls, a transition message is added.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing:
              - The `"dialog_state"` set to `"pop"` to indicate a state transition.
              - A list of messages informing the user that the primary assistant
                is resuming control.
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


def route_to_workflow(
    state: State,
) -> Literal["primary_assistant", "music_assistant", "customer_assistant"]:
    """
    Determines the appropriate workflow to transition into.

    This function inspects the current dialog state and routes the conversation
    to the correct assistant. If no prior state exists, it defaults to the
    `"primary_assistant"`.

    Args:
        state (State): The current state of the conversation, which contains
                       the dialog state history.

    Returns:
        Literal["primary_assistant", "music_assistant", "customer_assistant"]:
        The assistant to which the workflow should transition.
    """
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]
