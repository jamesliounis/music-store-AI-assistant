import json
import uuid
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from graph import ConversationGraphBuilder


def initialize_graph():
    """
    Initialize the conversation graph and load the configuration.

    Returns:
    --------
    graph : StateGraph
        The compiled conversation graph.
    config_data : dict
        The configuration data with a unique thread_id.
    """
    # Load the configuration
    with open("/Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/data/config.json", "r") as f:
        config_data = json.load(f)

    # Add a unique thread_id
    thread_id = str(uuid.uuid4())
    config_data["configurable"]["thread_id"] = thread_id

    # Build the graph
    graph_builder = ConversationGraphBuilder()
    graph = graph_builder.build_graph()

    # Initialize the graph with an empty input to populate the state
    if "customer_id" not in config_data["configurable"]:
        raise ValueError("Missing customer_id in config_data['configurable']")

    graph.invoke({"messages": []}, config_data)

    return graph, config_data


def print_latest_event(events, user_input=None):
    """
    Helper to print only the most recent AI message and the user's input.
    Skips repeated system or irrelevant responses.
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
    # Initialize the graph and configuration
    graph, config_data = initialize_graph()

    # Retrieve the user_info from the state
    snapshot = graph.get_state(config_data)
    user_info_data = snapshot.values['user_info']

    # Build a system message referencing the user’s info
    system_msg = SystemMessage(
        content=(
            f"You are an AI Chatbot. You are speaking with this customer: {user_info_data}, "
            f"a valued customer. Greet them by name. Pay attention to their unique customer ID: {user_info_data.get('CustomerId')}."
        )
    )

    # Insert the system message into the graph
    init_result = graph.invoke({"messages": [system_msg]}, config_data)

    # Print the initial greeting from the assistant
    if isinstance(init_result, dict):
        print("\n-- Initial Assistant Response --\n")
        print_latest_event([init_result])

    print("\n-- Chat initialized successfully --\n")

    # Start the interactive loop
    while True:
        user_input = input("YOU: ")
        if user_input.strip().lower() in {"q", "quit"}:
            print("Exiting chat. Goodbye!")
            break

        # Pass the new user message to the graph
        events = graph.stream({"messages": [HumanMessage(content=user_input)]}, config_data, stream_mode="values")
        print_latest_event(events, user_input=user_input)

        # Check for interrupts
        snapshot = graph.get_state(config_data)
        while snapshot.next:
            print("\n**INTERRUPT**: The chatbot wants to perform a sensitive action.\n")
            user_decision = input("Approve? (y/n or type changes): ").strip().lower()

            if user_decision == "y":
                resumed_output = graph.invoke(None, config_data)
                if isinstance(resumed_output, dict) and "messages" in resumed_output:
                    print_latest_event([resumed_output])
            else:
                denial_msg = ToolMessage(
                    content=f"Action denied by user. Reason: '{user_decision}'. Please adapt."
                )
                resumed_output = graph.invoke({"messages": [denial_msg]}, config_data)
                if isinstance(resumed_output, dict) and "messages" in resumed_output:
                    print_latest_event([resumed_output])

            snapshot = graph.get_state(config_data)

        print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    main()
