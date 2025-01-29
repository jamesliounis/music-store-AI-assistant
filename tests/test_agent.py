import unittest
from langgraph.graph import StateGraph
from agent.agent import build_graph
from utils.state import State
from langchain_core.messages import HumanMessage, AIMessage
import os

class TestMusicStoreAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the graph for testing before running any tests."""
        cls.graph = build_graph()

    def test_graph_initialization(self):
        """Test if the state graph is built successfully."""
        self.assertIsInstance(self.graph, StateGraph)

    def test_initial_state(self):
        """Ensure that the graph starts from `fetch_user_info`."""
        state = {"messages": [], "dialog_state": [], "user_info": {}}
        next_states = self.graph.get_next_states("fetch_user_info", state)
        self.assertIn("primary_assistant", next_states)

    def test_primary_assistant_routing(self):
        """Check that the primary assistant routes correctly based on user input."""
        state = {
            "messages": [HumanMessage(content="I want to update my profile.")],
            "dialog_state": [],
            "user_info": {"customer_id": "12345"}
        }
        next_states = self.graph.get_next_states("primary_assistant", state)
        self.assertIn("enter_customer_profile", next_states)

        state = {
            "messages": [HumanMessage(content="Find me songs by The Beatles.")],
            "dialog_state": [],
            "user_info": {}
        }
        next_states = self.graph.get_next_states("primary_assistant", state)
        self.assertIn("enter_music", next_states)

    def test_customer_assistant_tool_execution(self):
        """Ensure customer assistant tools (get_customer_info, update_customer_profile) work."""
        state = {
            "messages": [HumanMessage(content="What is my email?")],
            "dialog_state": ["customer"],
            "user_info": {"customer_id": "12345"}
        }
        next_states = self.graph.get_next_states("customer_assistant", state)
        self.assertIn("customer_safe_tools", next_states)

    def test_music_assistant_tool_execution(self):
        """Ensure music assistant tools (get_tracks_by_artist) execute properly."""
        state = {
            "messages": [HumanMessage(content="Show me songs by The Beatles.")],
            "dialog_state": ["music"],
        }
        next_states = self.graph.get_next_states("music_assistant", state)
        self.assertIn("music_tools", next_states)

    def test_leave_skill_transition(self):
        """Ensure that leaving the skill transitions back to the primary assistant."""
        state = {
            "messages": [AIMessage(content="Exiting...")],
            "dialog_state": ["customer"]
        }
        next_states = self.graph.get_next_states("leave_skill", state)
        self.assertIn("primary_assistant", next_states)

    def test_error_handling(self):
        """Ensure the system correctly handles unexpected inputs or failures."""
        state = {
            "messages": [HumanMessage(content="Some random nonsense")],
            "dialog_state": []
        }
        with self.assertRaises(ValueError):
            self.graph.get_next_states("primary_assistant", state)

    def test_graph_visualization_saving(self):
        """Test that the graph visualization saves correctly."""
        output_path = "test_graph_visualization.png"
        try:
            from agent.agent import save_graph_visualization
            save_graph_visualization(self.graph, output_path)
            self.assertTrue(os.path.exists(output_path))
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)  # Cleanup test file

if __name__ == "__main__":
    unittest.main()

