# agent/utils/state.py

from typing import List, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """
    Updates the dialog state stack by pushing or popping states.

    This function manages the state transitions of a dialog by modifying
    the `dialog_state` list. It follows these rules:
    - If `right` is `None`, the state remains unchanged.
    - If `right` is `"pop"`, the most recent state is removed (if any).
    - Otherwise, `right` is appended to the stack, representing a transition
      to a new state.

    Args:
        left (list[str]): The current list representing the dialog state stack.
        right (Optional[str]): The state to be pushed onto the stack, or `"pop"`
                               to remove the last state.

    Returns:
        list[str]: The updated dialog state stack after applying the operation.
    """
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    """
    Represents the structured state of the dialog system.

    This state object tracks the conversation history, user information,
    and the current assistant handling the conversation.

    Attributes:
        messages (Annotated[list[AnyMessage], add_messages]):
            A list of messages exchanged during the conversation,
            with `add_messages` handling updates.

        user_info (str):
            Information about the user, typically retrieved at the start
            of the conversation.

        dialog_state (Annotated[list[Literal["assistant", "music", "customer"]], update_dialog_stack]):
            A stack representing the current conversation flow, where:
            - `"assistant"` represents the primary assistant.
            - `"music"` represents the music assistant.
            - `"customer"` represents the customer support assistant.
            - The `update_dialog_stack` function manages additions and removals
              from this list.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[Literal["assistant", "music", "customer"]],
        update_dialog_stack,
    ]
