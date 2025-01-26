from pydantic import BaseModel, Field

class Router(BaseModel):
    """
    A routing model responsible for determining the appropriate agent
    to handle user inquiries based on the context of their message.

    Attributes:
    ----------
    choice : str
        Specifies the agent to route to. Should be one of:
        - 'music' for inquiries related to music recommendations or music information.
        - 'customer' for inquiries related to updating or accessing user information.
    """
    choice: str = Field(description="Should be one of: 'music', 'customer'.")


class ToCustomerAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle customer profile queries."""

    request: str = Field(
        description="Any necessary followup questions to retrieve or update customer information should clarify before proceeding."
    )


class ToMusicAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle music-related queries."""

    request: str = Field(
        description="Any necessary followup questions to give information about music should clarify before proceeding."
    )
