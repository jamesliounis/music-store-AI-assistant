# src/langgraph/nodes/primary_assistant.py

from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.utils import Assistant, CompleteOrEscalate, State
from langgraph.router import Router, ToCustomerAssistant, ToMusicAssistant  # Adjust import path as needed
from typing import List


class PrimaryRunnable:
    """
    A class to encapsulate the Primary Assistant runnable, handling
    general Q&A and delegating tasks to specialized assistants (Customer and Music).
    """

    def __init__(self, model_name: str = "gpt-4-turbo-preview", temperature: float = 0):
        """
        Initializes the PrimaryAssistant with the specified language model and temperature.

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
        self.primary_tools = [Router]
        self.assistant = self._create_assistant()

    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Creates the primary assistant prompt with routing instructions.

        Returns:
        -------
        ChatPromptTemplate
            The configured chat prompt template for the Primary Assistant.
        """
        system_message = """
        Your job is to serve as a polite and helpful customer service representative for a music store.

        You can assist customers in the following ways:
        1. **Updating user information**:
            - If the customer wants to update or access their information in the user database, route them to `customer`.
        2. **Recommending music**:
            - If the customer wants to find music or learn about music, route them to `music`.

        Routing Instructions:
        - If the user mentions updating or accessing their personal information, call the router with 'customer'.
        - If the user mentions music recommendations or any music-related inquiry, call the router with 'music'.
        - For any other inquiries, respond politely and explain what you can assist with.

        Always aim to be polite and clear in your interactions.
        """

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
            The configured Assistant instance for handling general Q&A and routing.
        """
        prompt = self._create_prompt()
        runnable = prompt | self.llm.bind_tools(
            self.primary_tools + [
                ToMusicAssistant,
                ToCustomerAssistant,
            ]
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

