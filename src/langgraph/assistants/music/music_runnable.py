# src/langgraph/nodes/music_assistant.py

from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.utils import Assistant, CompleteOrEscalate, State
from langgraph.router import Router, ToCustomerAssistant, ToMusicAssistant  # Adjust import path as needed
from src.tools.query_handler import check_for_songs, get_tracks_by_artist, get_albums_by_artist  # Adjust import path as needed
from typing import List


class MusicRunnable:
    """
    A class to encapsulate the Music Assistant runnable, handling
    music-related interactions such as searching for songs, albums, and artists.
    """

    def __init__(self, model_name: str = "gpt-4-turbo-preview", temperature: float = 0):
        """
        Initializes the MusicAssistant with the specified language model and temperature.

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
        self.music_tools = [check_for_songs, get_tracks_by_artist, get_albums_by_artist]
        self.assistant = self._create_assistant()

    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Creates the music assistance prompt.

        Returns:
        -------
        ChatPromptTemplate
            The configured chat prompt template for the Music Assistant.
        """
        system_message = """
        Your job is to assist a customer in finding any songs they are looking for.

        You have access to specific tools to look up songs, albums, or artists. If a customer asks for something you cannot assist with, 
        politely inform them of the limitations and let them know what you can help with.

        When looking up artists and songs, sometimes the exact match may not be found. In such cases, the tools are designed to return 
        information about similar songs or artists. This is intentional and helps provide relevant recommendations.
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
            The configured Assistant instance for handling music interactions.
        """
        prompt = self._create_prompt()
        runnable = prompt | self.llm.bind_tools(
            self.music_tools + [CompleteOrEscalate]
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
