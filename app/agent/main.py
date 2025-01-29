# agent/main.py

import json
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import AnyMessage
from agent import build_graph  
from langgraph.graph.message import add_messages
from utils.state import State  
from utils.logger import get_logger 
import traceback  
from typing import Optional

# Initialize Logger
logger = get_logger(__name__)

def load_config(config_path: str) -> dict:
    """
    Load configuration from a JSON file.

    Parameters:
    ----------
    config_path : str
        The file path to the configuration JSON.

    Returns:
    -------
    dict
        The loaded configuration data.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found.")
        raise FileNotFoundError(f"{config_path} does not exist.")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    logger.info(f"Configuration loaded from {config_path}.")
    return config_data

def build_system_message(user_info: dict) -> SystemMessage:
    """
    Build a system message that references the user's info.

    Parameters:
    ----------
    user_info : dict
        The user's information retrieved from the database.

    Returns:
    -------
    SystemMessage
        The constructed system message.
    """
    customer_id = user_info.get('CustomerID', 'N/A')
    return SystemMessage(
        content=(
            f"You are an AI Chatbot. You are speaking with this customer: {user_info}, "
            f"a valued customer. You must absolutely greet them by name. Pay attention to their unique customer ID: {customer_id}."
        )
    )

def print_latest_event(events, user_input=None):
    """
    Helper to print only the most recent AI message and the user's input.
    It skips repeated system or irrelevant responses.
    """
    latest_message = None

    for event in events:
        if "messages" in event:
            for msg in event["messages"]:
                if isinstance(msg, AIMessage):  # Focus on AI responses
                    latest_message = f"ASSISTANT: {msg.content}"
                elif isinstance(msg, ToolMessage):  # Include tool updates if relevant
                    latest_message = f"TOOL: {msg.content}"

    if user_input:
        print(f"YOU: {user_input}")
    if latest_message:
        print(latest_message) 

def main():
    """
    The main function to initialize and run the LangGraph-based customer support chatbot.
    """
    try:
        # Load environment variables from the .env file
        load_dotenv()
        logger.info("Environment variables loaded.")

        # Determine the base directory (app root)
        base_dir = Path(__file__).resolve().parent.parent  # Adjusted to account for the directory structure
        logger.debug(f"Base directory set to: {base_dir}")

        # Path to config.json
        config_path = base_dir / "config.json"
        logger.debug(f"Config file path resolved to: {config_path}")

        # Load configuration
        config_data = load_config(str(config_path))

        # Dynamically inject thread_id and set customer_id with default
        thread_id = str(uuid.uuid4())
        config_data["configurable"]["thread_id"] = thread_id
        config_data["configurable"]["customer_id"] = config_data["configurable"].get("customer_id", 1)
        logger.info(f"Injected thread_id: {thread_id} and set customer_id: {config_data['configurable']['customer_id']}")

        # Build the LangGraph
        graph = build_graph()
        logger.info("LangGraph built successfully.")

        # Initialize the graph (trigger the first node)
        graph.invoke({"messages": []}, config_data)
        logger.info("Graph initialized with empty input.")

        # Retrieve user_info from the state
        snapshot = graph.get_state(config_data)
        user_info_data = snapshot.values["user_info"]
        logger.debug(f"User info retrieved: {user_info_data}")

        # Build system message
        system_msg = build_system_message(user_info_data)

        # Insert the system message and print greeting
        init_result = graph.invoke({"messages": [system_msg]}, config_data)
        logger.info("System message inserted into the graph.")

        if isinstance(init_result, dict) and "messages" in init_result:
            print_latest_event(init_result["messages"])


        # Start the interactive chat loop
        while True:
            try:
                user_input = input("YOU: ")
                if user_input.strip().lower() in {"q", "quit"}:
                    print("Exiting chat. Goodbye!")
                    logger.info("Chat session terminated by user.")
                    break

                # Pass the new user message to the graph
                events = graph.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config_data,
                    stream_mode="values"
                )
                print_latest_event(events, user_input=user_input)
                logger.debug("User input processed and response printed.")

                # Check for interrupts (human-in-the-loop approvals)
                snapshot = graph.get_state(config_data)
                while snapshot.next:
                    print("\n**INTERRUPT**: The chatbot wants to perform a sensitive action.\n")
                    user_decision = input("Approve? (y/n or type reason): ").strip().lower()
                    logger.debug(f"User decision on interrupt: {user_decision}")

                    if user_decision == "y":
                        resumed_output = graph.invoke(None, config_data)
                        if isinstance(resumed_output, dict) and "messages" in resumed_output:
                            print_latest_event([resumed_output])
                            logger.info("Sensitive action approved by user.")
                    else:
                        denial_msg = ToolMessage(
                            content=f"Action denied by user. Reason: '{user_decision}'. Please adapt."
                        )
                        resumed_output = graph.invoke({"messages": [denial_msg]}, config_data)
                        if isinstance(resumed_output, dict) and "messages" in resumed_output:
                            print_latest_event([resumed_output])
                            logger.info("Sensitive action denied by user.")

                    snapshot = graph.get_state(config_data)

                print("\n" + "=" * 40 + "\n")

            except KeyboardInterrupt:
                print("\nExiting chat. Goodbye!")
                logger.info("Chat session terminated via KeyboardInterrupt.")
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                print("An unexpected error occurred. Please try again.")

    except Exception as e:
        logger.critical(f"Failed to start the chatbot: {e}")
        logger.critical(traceback.format_exc())
        print("Failed to start the chatbot. Please check the logs for more details.")

if __name__ == "__main__":
    main()
