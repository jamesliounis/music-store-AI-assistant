from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.graph import START, END
import uuid

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda

from utils import State, pop_dialog_state, create_entry_node, create_tool_node_with_fallback, Assistant
from tools.customer_tools import CustomerProfileManager

from assistants.primary.primary_runnable import PrimaryRunnable
from assistants.customer.customer_runnable import CustomerRunnable
from assistants.music.music_runnable import MusicRunnable

from src.assistants.primary.router import Router, ToCustomerAssistant, ToMusicAssistant
from src.tools.customer_tools import CustomerProfileManager
from src.tools.music_tools import MusicRetrieverManager 

class ConversationGraphBuilder:
    """
    A class to build and compile the conversation graph integrating various assistants and tools.
    """

    def __init__(
        self,
        # primary_assistant,
        # customer_assistant,
        # music_assistant,
        # customer_safe_tools,
        # customer_sensitive_tools,
        # music_tools,
        # primary_assistant_tools
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
        
        # Assistants
        self.primary_assistant = PrimaryRunnable().get_runnable()
        self.customer_assistant = CustomerRunnable().get_runnable()
        self.music_assistant = MusicRunnable().get_runnable()

        # Tools
        self.music_retriever = MusicRetrieverManager()
        self.music_tools = [self.music_retriever.check_for_songs, self.music_retriever.get_tracks_by_artist, self.music_retriever.get_albums_by_artist]

        self.customer_profile_manager = CustomerProfileManager()
        self.customer_safe_tools = [self.customer_profile_manager.get_customer_info]
        self.customer_sensitive_tools = [self.customer_profile_manager.update_customer_profile]
        
        self.primary_assistant_tools = [Router, ToCustomerAssistant, ToMusicAssistant]

        self.builder = StateGraph(State)

    def user_info(self, state: State) -> State:
        """
        Retrieve a customer_id from state["configurable"]["customer_id"],
        call get_user_info with the correct 'RunnableConfig' signature,
        and store the result in state["user_info"].
        """
        return {"user_info": self.customer_profile_manager.get_user_info.invoke({})}

    def build_graph(self):
        """
        Constructs the conversation graph by adding nodes and edges.

        Returns:
        -------
        StateGraph
            The compiled conversation graph.
        """
        try:
            print("Adding node: fetch_user_info")
            self.builder.add_node("fetch_user_info", self.user_info)
            self.builder.add_edge(START, "fetch_user_info")

            # Music Assistant Workflow
            print("Adding node: enter_music")
            self.builder.add_node(
                "enter_music",
                create_entry_node("Music Assistant", "music"),
            )

            print("Adding node: music")

            # self.builder.add_node("music", self.music_assistant)
            self.builder.add_node("music", Assistant(self.music_assistant))
            self.builder.add_edge("enter_music", "music")

            print("Adding node: music_tools")
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

                safe_toolnames = [t.name for t in self.music_tools]
                if all(tc["name"] in safe_toolnames for tc in tool_calls):
                    return "music_tools"
                return "music_tools"

            print("Adding conditional edges for music...")
            self.builder.add_conditional_edges(
                "music",
                route_music,
                ["music_tools", "leave_skill", END],
            )

            print("Adding node: leave_skill")
            self.builder.add_node("leave_skill", RunnableLambda(pop_dialog_state))
            self.builder.add_edge("leave_skill", "primary_assistant")

            # Customer Assistant Workflow
            print("Adding node: enter_customer_profile")
            self.builder.add_node(
                "enter_customer_profile",
                create_entry_node("Customer Assistant", "customer_assistant"),
            )

            print("Adding node: customer_assistant")
            self.builder.add_node("customer_assistant", Assistant(self.customer_assistant))
            self.builder.add_edge("enter_customer_profile", "customer_assistant")

            print("Adding node: customer_safe_tools")
            self.builder.add_node(
                "customer_safe_tools",
                create_tool_node_with_fallback(self.customer_safe_tools),
            )

            print("Adding node: customer_sensitive_tools")
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

            print("Adding conditional edges for customer assistant...")
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

            self.builder.add_edge("customer_sensitive_tools", "customer_assistant")
            self.builder.add_edge("customer_safe_tools", "customer_assistant")

            # Primary Assistant Workflow
            print("Adding node: primary_assistant")
            self.builder.add_node("primary_assistant", Assistant(self.primary_assistant))

            print("Adding node: primary_assistant_tools")
            self.builder.add_node(
                "primary_assistant_tools",
                create_tool_node_with_fallback(self.primary_assistant_tools),
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

            print("Adding conditional edges for primary assistant...")
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

            print("Adding conditional edges for workflow routing...")
            self.builder.add_conditional_edges("fetch_user_info", route_to_workflow)

            print("Compiling the graph...")
            memory = MemorySaver()
            graph = self.builder.compile(
                checkpointer=memory,
                interrupt_before=[
                    "customer_sensitive_tools"
                ],
            )

            print("Graph built successfully!")
            return graph

        except Exception as e:
            print(f"Error while building graph: {e}")
            raise

    
    def save_graph_visualization(self, graph: 'StateGraph', save_path: str):
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


if __name__ == "__main__":
    # Path to save the graph visualization
    save_path = "/Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/docs/conversation_graph.png"
    graph_builder = ConversationGraphBuilder()
    graph = graph_builder.build_graph()

    # Save the graph visualization
    graph_builder.save_graph_visualization(graph, save_path)


