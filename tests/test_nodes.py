import unittest
from utils.nodes import (
    fetch_user_info,
    handle_tool_error,
    create_entry_node,
    pop_dialog_state,
    route_to_workflow,
)
from langchain_core.messages import ToolMessage, HumanMessage
from utils.state import State
from unittest.mock import patch

class TestNodes(unittest.TestCase):

    def setUp(self):
        """Set up a mock state for testing."""
        self.state_with_customer_id = {
            "configurable": {"customer_id": "12345"},
            "messages": [],
        }
        self.state_without_customer_id = {
            "configurable": {},
            "messages": [],
        }
        self.state_with_error = {
            "error": "Some tool error",
            "messages": [{"tool_calls": [{"id": "tool123"}]}],
        }
        self.state_with_tool_calls = {
            "messages": [{"tool_calls": [{"id": "tool123"}]}],
        }
        self.state_with_dialog_stack = {
            "dialog_state": ["primary_assistant", "customer_assistant"]
        }
        self.state_without_dialog_stack = {
            "dialog_state": []
        }

    @patch("utils.tools.get_customer_info.invoke")
    def test_fetch_user_info_with_customer_id(self, mock_get_customer_info):
        """Test that fetch_user_info retrieves customer data when customer_id is provided."""
        mock_get_customer_info.return_value = {"name": "John Doe", "email": "john@example.com"}
        result = fetch_user_info(self.state_with_customer_id)
        self.assertIn("user_info", result)
        self.assertEqual(result["user_info"], {"name": "John Doe", "email": "john@example.com"})

    def test_fetch_user_info_without_customer_id(self):
        """Test that fetch_user_info returns an error when no customer_id is provided."""
        result = fetch_user_info(self.state_without_customer_id)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "No customer_id provided.")

    def test_handle_tool_error(self):
        """Test that handle_tool_error returns formatted error messages for failed tool calls."""
        result = handle_tool_error(self.state_with_error)
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].tool_call_id, "tool123")
        self.assertTrue("Error: 'Some tool error'" in result["messages"][0].content)

    def test_create_entry_node(self):
        """Test that create_entry_node returns an entry message and correct dialog state."""
        entry_node = create_entry_node("Customer Assistant", "customer_assistant")
        result = entry_node(self.state_with_tool_calls)
        self.assertIn("messages", result)
        self.assertIn("dialog_state", result)
        self.assertEqual(result["dialog_state"], "customer_assistant")
        self.assertTrue("The assistant is now the Customer Assistant" in result["messages"][0].content)
        self.assertEqual(result["messages"][0].tool_call_id, "tool123")

    def test_pop_dialog_state_with_tool_calls(self):
        """Test that pop_dialog_state correctly transitions back to the primary assistant."""
        result = pop_dialog_state(self.state_with_tool_calls)
        self.assertEqual(result["dialog_state"], "pop")
        self.assertEqual(len(result["messages"]), 1)
        self.assertTrue("Resuming dialog with the host assistant" in result["messages"][0].content)
        self.assertEqual(result["messages"][0].tool_call_id, "tool123")

    def test_route_to_workflow_with_dialog_state(self):
        """Test that route_to_workflow returns the last dialog state when it exists."""
        result = route_to_workflow(self.state_with_dialog_stack)
        self.assertEqual(result, "customer_assistant")

    def test_route_to_workflow_without_dialog_state(self):
        """Test that route_to_workflow defaults to primary_assistant if no state exists."""
        result = route_to_workflow(self.state_without_dialog_stack)
        self.assertEqual(result, "primary_assistant")


if __name__ == "__main__":
    unittest.main()

