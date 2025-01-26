import json
import uuid
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import ast
from graph import ConversationGraphBuilder

# 1) Load config from JSON, add thread_id at runtime
with open("/Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/data/config.json", "r") as f:
        config_data = json.load(f)

graph_builder =  ConversationGraphBuilder()
graph = graph_builder.build_graph()

thread_id = str(uuid.uuid4())
config_data["configurable"]["thread_id"] = thread_id

# 2) Initialize the graph with an empty input to populate the state
graph.invoke({"messages": []}, config_data)  # Trigger the graph's first node

# 3) Retrieve the user_info from the state
snapshot = graph.get_state(config_data)
user_info_data = snapshot.values["user_info"]

# 4) Build a single system message that references the user’s info
system_msg = SystemMessage(
    content=(
        f"You are an AI Chatbot. You are speaking with this customer: {user_info_data}, "
        f"a valued customer. Greet them by name. Pay attention to their unique customer ID: {user_info_data['CustomerId']}."
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


# 5) Insert the system message once, then print the resulting greeting (if any)
init_result = graph.invoke({"messages": [system_msg]}, config_data)

# Print the AI's response (greeting) if it exists
if isinstance(init_result, dict):
    print("\n-- Initial Assistant Response --\n")
    print_latest_event([init_result])

print("\n-- Done preloading system message into the state --\n")

# 6) Start the interactive loop where the user can type
while True:
    user_input = input("YOU: ")
    if user_input.strip().lower() in {"q", "quit"}:
        print("Exiting chat. Goodbye!")
        break

    # Pass only the new user message to the graph
    events = graph.stream({"messages": [HumanMessage(content=user_input)]}, config_data, stream_mode="values")
    print_latest_event(events, user_input=user_input)

    # Check for interrupts as usual
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
