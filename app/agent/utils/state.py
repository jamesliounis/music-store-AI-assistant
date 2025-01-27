# agent/utils/state.py

from typing import List, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    messages: List[AnyMessage]
    user_info: Optional[dict]
    dialog_state: List[Literal["assistant", "music", "customer"]]

