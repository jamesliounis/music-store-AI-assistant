from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from src.utils import Assistant, CompleteOrEscalate, State
from src.tools.customer_tools import CustomerProfileManager
from typing import List
from langchain_community.utilities.sql_database import SQLDatabase



class CustomerRunnable:
    """
    A class to encapsulate the Customer Assistant runnable, handling
    customer-related interactions such as retrieving and updating customer profiles.
    """

    def __init__(self, model_name: str = "gpt-4-turbo-preview", temperature: float = 0):
        """
        Initializes the CustomerAssistant with the specified language model and temperature.

        Parameters:
        ----------
        model_name : str
            The name of the language model to use (default: "gpt-4-turbo-preview").
        temperature : float
            The temperature setting for the language model (default: 0).
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(
            temperature=self.temperature,
            streaming=True,
            model=self.model_name
        )
        self.customer_profile_manager = CustomerProfileManager()
        self.customer_tools = [self.customer_profile_manager.get_customer_info, self.customer_profile_manager.update_customer_profile]
        self.assistant = self._create_assistant()
        self.database = SQLDatabase.from_uri("sqlite:////Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/data/chinook.db")

    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Creates the customer assistance prompt with additional security instructions.

        Returns:
        -------
        ChatPromptTemplate
            The configured chat prompt template for the Customer Assistant.
        """
        extra_customer_security = """
        You will be given the customer data when the chat begins. Do not ask them for any clarification. 
        Do not under any circumstance share customer information of another customer. 
        IMPORTANT: If the customer wants to make any modification to their profile, you must ask them to confirm their current email and phone number.  
        IMPORTANT: If the customer wants to inquire about the information that you have for them on file, you must ask them to confirm their current email and phone number.
        """

        system_message = f"""
        Your role is to assist a user in viewing or updating their profile information.

        You have access to the following tools to perform these tasks:
        1. `get_customer_info`: Retrieve information about a specific customer using their unique customer ID.
           - Required input: `customer_id`
           - Example: "What is your customer ID so I can retrieve your profile?"

        2. `update_customer_profile`: Update a specific field in the customer's profile.
           - Required inputs:
               - `customer_id`: The unique ID of the customer.
               - `field`: The name of the profile field to update (e.g., "FirstName", "Email", "Phone").
               - `new_value`: The new value to assign to the specified field.
           - Example: "Could you confirm your customer ID and provide the new email address you'd like to update?"

        Steps to assist the user:
        - Always verify you have all the required inputs before using a tool.
        - If a user wants to update a field (e.g., their email or phone number), ask for the relevant inputs:
            - For first name updates: Ask for the new first name.
            - For email updates: Ask for the new email address.
            - For phone updates: Ask for the new phone number.
            - For other fields: Clarify the field name and the new value.

        Example interactions:
        1. To retrieve profile information:
           - "What is your customer ID so I can look up your profile?"
        2. To update the profile:
           - "What is your customer ID, and which field would you like to update? Please provide the new value for that field."

        Important Notes:
        - If the user requests to update a field that is invalid or unavailable, inform them politely of the valid fields:
          - Allowed fields: `FirstName`, `LastName`, `Company`, `Address`, `City`, `State`, `Country`, `PostalCode`, `Phone`, `Fax`, `Email`, and `SupportRepId`.
        - If you are unable to assist with the user's request, politely suggest they contact customer support.
        """ + extra_customer_security

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("placeholder", "{messages}"),
            ]
        ).partial(time=datetime.now())

    def _create_assistant(self) -> Assistant:
        """
        Creates and configures the Assistant instance with the prompt and tools.

        Returns:
        -------
        Assistant
            The configured Assistant instance for handling customer interactions.
        """
        prompt = self._create_prompt()
        runnable = prompt | self.llm.bind_tools(
            self.customer_tools + [CompleteOrEscalate]
        )
        return Assistant(runnable)

    def get_runnable(self) -> Assistant:
        """
        Provides access to the configured Assistant runnable.

        Returns:
        -------
        Assistant
            The configured Assistant instance.
        """
        return self.assistant


def create_customer_runnable() -> Assistant:
    """
    Factory function to create and return the Customer Assistant.

    Returns:
        Assistant: An instance of the Customer Assistant.
    """
    return CustomerRunnable().get_runnable()

if __name__ == "__main__":
    # Simple test to ensure the assistant is created correctly
    customer_assistant = create_customer_runnable()
    print("Customer Assistant successfully created:", customer_assistant)