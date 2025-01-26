from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig, Runnable

def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """
    Pushes or pops a dialog state from the stack.

    Parameters
    ----------
    left : list of str
        The current stack of states (e.g., ["assistant", "music"]).
    right : str or None
        - If None, returns the stack unchanged.
        - If "pop", removes the last state (if any).
        - Otherwise, appends the new state to the stack.

    Returns
    -------
    list of str
        The updated dialog stack.
    """
    if right is None:
        return left
    if right == "pop":
        # Safely pop the last element if the stack isn't empty
        return left[:-1] if left else left
    return left + [right]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[Literal["assistant", "music", "customer"]],
        update_dialog_stack
    ]


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
    
    