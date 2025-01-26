# src/langgraph/nodes/conversation_graph_builder.py

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.graph import START, END
import uuid

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from customer_tools import get_user_info

from langgraph.utils import State, pop_dialog_state  # Importing the pop_dialog_state function

# Assuming create_entry_node and create_tool_node_with_fallback are defined elsewhere
from utils import create_entry_node, create_tool_node_with_fallback  # Adjust the import path accordingly

class ConversationGraphBuilder:
    """
    A class to build and compile the conversation graph integrating various assistants and tools.
    """

    def __init__(
        self,
        primary_assistant,
        customer_assistant,
        music_assistant,
        customer_safe_tools,
        customer_sensitive_tools,
        music_tools,
        primary_assistant_tools
    ):
        """
        Initializes the ConversationGraphBuilder with the necessary assistants and tools.

        Parameters:
        ----------
        primary_assistant : Assistant
            The primary assistant runnable.
        customer_assistant : Assistant
            The customer assistant runnable.
        music_assistant : Assistant
            The music assistant runnable.
        customer_safe_tools : List[Tool]
            List of safe tools for the customer assistant.
        customer_sensitive_tools : List[Tool]
            List of sensitive tools for the customer assistant.
        music_tools : List[Tool]
            List of tools for the music assistant.
        primary_assistant_tools : List[Tool]
            List of tools for the primary assistant.
        """
        self.primary_assistant = primary_assistant
        self.customer_assistant = customer_assistant
        self.music_assistant = music_assistant
        self.customer_safe_tools = customer_safe_tools
        self.customer_sensitive_tools = customer_sensitive_tools
        self.music_tools = music_tools
        self.primary_assistant_tools = primary_assistant_tools

        self.builder = StateGraph(State)

    def build_graph(self):
        """
        Constructs the conversation graph by adding nodes and edges.

        Returns:
        -------
        StateGraph
            The compiled conversation graph.
        """
        # Fetch user information
        user_info = get_user_info()
        self.builder.add_node("fetch_user_info", user_info)
        self.builder.add_edge(START, "fetch_user_info")

        # Music Assistant Workflow
        self.builder.add_node(
            "enter_music",
            create_entry_node("Music Assistant", "music"),
        )
        self.builder.add_node("music", self.music_assistant)

        self.builder.add_edge("enter_music", "music")

        self.builder.add_node(
            "music_tools",
            create_tool_node_with_fallback(self.music_tools),
        )

        def route_music(state: State):
            route = tools_condition(state)
            if route == END:
                return END

            tool_calls = state["messages"][-1].tool_calls
            did_cancel = any(tc["name"] == "CompleteOrEscalate" for tc in tool_calls)

            if did_cancel:
                return "leave_skill"

            # Route directly to music_tools if only one group exists
            safe_toolnames = [t.name for t in self.music_tools]
            if all(tc["name"] in safe_toolnames for tc in tool_calls):
                return "music_tools"
            return "music_tools"

        self.builder.add_conditional_edges(
            "music",
            route_music,
            ["music_tools", "leave_skill", END],
        )

        self.builder.add_node("leave_skill", RunnableLambda(pop_dialog_state))
        self.builder.add_edge("leave_skill", "primary_assistant")

        # Customer Assistant Workflow
        self.builder.add_node(
            "enter_customer_profile",
            create_entry_node("Customer Assistant", "customer_assistant"),
        )
        self.builder.add_node("customer_assistant", self.customer_assistant)
        self.builder.add_edge("enter_customer_profile", "customer_assistant")

        self.builder.add_node(
            "customer_safe_tools",
            create_tool_node_with_fallback(self.customer_safe_tools),
        )
        self.builder.add_node(
            "customer_sensitive_tools",
            create_tool_node_with_fallback(self.customer_sensitive_tools),
        )

        def route_customer_assistant(state: State):
            route = tools_condition(state)
            if route == END:
                return END
            tool_calls = state["messages"][-1].tool_calls
            did_cancel = any(tc["name"] == "CompleteOrEscalate" for tc in tool_calls)
            if did_cancel:
                return "leave_skill"
            safe_toolnames = [t.name for t in self.customer_safe_tools]
            if all(tc["name"] in safe_toolnames for tc in tool_calls):
                return "customer_safe_tools"
            return "customer_sensitive_tools"

        self.builder.add_edge("customer_sensitive_tools", "customer_assistant")
        self.builder.add_edge("customer_safe_tools", "customer_assistant")
        self.builder.add_conditional_edges(
            "customer_assistant",
            route_customer_assistant,
            [
                "customer_safe_tools",
                "customer_sensitive_tools",
                "leave_skill",
                END,
            ],
        )

        # Primary Assistant Workflow
        self.builder.add_node("primary_assistant", self.primary_assistant)
        self.builder.add_node(
            "primary_assistant_tools", create_tool_node_with_fallback(self.primary_assistant_tools)
        )

        def route_primary_assistant(state: State):
            route = tools_condition(state)
            if route == END:
                return END
            tool_calls = state["messages"][-1].tool_calls
            if tool_calls:
                if tool_calls[0]["name"] == "ToCustomerAssistant":
                    return "enter_customer_profile"
                elif tool_calls[0]["name"] == "ToMusicAssistant":
                    return "enter_music"
                return "primary_assistant_tools"
            raise ValueError("Invalid route")

        # The assistant can route to one of the delegated assistants,
        # directly use a tool, or directly respond to the user
        self.builder.add_conditional_edges(
            "primary_assistant",
            route_primary_assistant,
            [
                "enter_customer_profile",
                "enter_music",
                "primary_assistant_tools",
                END,
            ],
        )
        self.builder.add_edge("primary_assistant_tools", "primary_assistant")

        # Each delegated workflow can directly respond to the user
        # When the user responds, we want to return to the currently active workflow
        def route_to_workflow(state: State) -> Literal[
            "primary_assistant",
            "music",
            "customer_assistant"
        ]:
            """If we are in a delegated state, route directly to the appropriate assistant."""
            dialog_state = state.get("dialog_state")
            if not dialog_state:
                return "primary_assistant"
            return dialog_state[-1]

        self.builder.add_conditional_edges("fetch_user_info", route_to_workflow)

        # Compile graph
        memory = MemorySaver()
        graph = self.builder.compile(
            checkpointer=memory,
            # Let the user approve or deny the use of sensitive tools
            interrupt_before=[
                "customer_sensitive_tools"
            ],
        )

        return graph
    
    def save_graph_visualization(graph: 'StateGraph', save_path: str):
        """
        Saves the graph visualization as a PNG image to the specified path.

        Parameters:
        ----------
        graph : StateGraph
            The compiled conversation graph.
        save_path : str
            The file path where the visualization will be saved.
        """
        try:
            # Generate the Mermaid PNG image bytes
            image_bytes = graph.get_graph(xray=True).draw_mermaid_png()
            
            # Write the image bytes to the specified file
            with open(save_path, "wb") as f:
                f.write(image_bytes)
            print(f"Graph visualization saved to {save_path}")
        except Exception as e:
            print(f"Failed to save graph visualization: {e}")
            # This requires some extra dependencies and is optional
            pass
