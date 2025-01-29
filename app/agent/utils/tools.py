# agent/utils/tools.py

from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.runnables import RunnableConfig
from typing import List, Dict, Any, Optional
import sqlite3
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler(
            "/Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/app/agent/logs/app.log"
        ),
    ],
)

logger = logging.getLogger(__name__)

# Initialize Database
DATABASE_FILE = (
    "/Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/data/chinook.db"
)
DATABASE_URI = f"sqlite:///{os.path.abspath(DATABASE_FILE)}"
db = SQLDatabase.from_uri(DATABASE_URI)

# Initialize LLM
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0))
STREAMING = os.getenv("STREAMING", "True").lower() in ("true", "1", "t")

llm = ChatOpenAI(temperature=TEMPERATURE, streaming=STREAMING, model=MODEL_NAME)

# Initialize Embeddings
embeddings = OpenAIEmbeddings()


@tool
def get_customer_info(customer_id: int):
    """
    Retrieve customer details by customer ID.
    
    Args:
        customer_id (int): Unique customer identifier.

    Returns:
        list | dict: Customer data or an error message.
    """

    # Validate that a customer ID is provided
    if not isinstance(customer_id, int) or customer_id <= 0:
        return {
            "error": "Invalid customer ID. Please provide a valid positive integer."
        }

    # Query the database for customer information
    try:
        result = db.run(f"SELECT * FROM customers WHERE CustomerID = {customer_id};")
        if result:
            return result
        else:
            return {"error": f"No customer found with ID {customer_id}."}
    except Exception as e:
        return {"error": f"An error occurred while fetching customer info: {str(e)}"}


@tool
def get_user_info(config: RunnableConfig) -> dict:
    """
    Retrieve customer info using `customer_id` from config.

    Args:
        config (RunnableConfig): Config containing `customer_id`.

    Returns:
        dict: Customer details or an error message.
    """
    configuration = config.get("configurable", {})
    customer_id = configuration.get("customer_id", None)
    if not customer_id:
        raise ValueError("No customer_id configured in config['configurable'].")

    # Connect to your DB
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Query the customer row
    query = "SELECT * FROM customers WHERE CustomerID = ?"
    cursor.execute(query, (customer_id,))
    row = cursor.fetchone()
    column_names = (
        [desc[0] for desc in cursor.description] if cursor.description else []
    )

    if row:
        # Convert the tuple row to a dictionary
        row_dict = dict(zip(column_names, row))
        result = row_dict
    else:
        result = {"error": f"No customer found with ID {customer_id}"}

    cursor.close()
    conn.close()

    return result


@tool
def update_customer_profile(customer_id: int, field: str, new_value: str):
    """
    Update a specific customer profile field.

    Args:
        customer_id (int): Unique customer ID.
        field (str): Field name to update.
        new_value (str): New field value.

    Returns:
        dict: Success or error message.
    """
    print(
        f"Received inputs - Customer ID: {customer_id}, Field: {field}, New Value: {new_value}"
    )

    # Validate inputs
    if not isinstance(customer_id, int) or customer_id <= 0:
        return {
            "error": "Invalid customer ID. Please provide a valid positive integer."
        }

    valid_fields = [
        "FirstName",
        "LastName",
        "Company",
        "Address",
        "City",
        "State",
        "Country",
        "PostalCode",
        "Phone",
        "Fax",
        "Email",
        "SupportRepId",
    ]
    if field not in valid_fields:
        return {
            "error": f"Invalid field '{field}'. Allowed fields are: {', '.join(valid_fields)}"
        }

    if not isinstance(new_value, str) or len(new_value.strip()) == 0:
        return {"error": "Invalid value. Please provide a valid new value."}

    try:
        # Connect to the database directly
        # conn = sqlite3.connect(DATABASE_URI)
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        # Parameterized query
        query = f"UPDATE customers SET {field} = ? WHERE CustomerID = ?;"
        print(f"Executing query: {query} with params: ({new_value}, {customer_id})")

        # Execute query
        cursor.execute(query, (new_value, customer_id))
        conn.commit()  # Commit the changes
        rows_affected = cursor.rowcount

        cursor.close()
        conn.close()

        # Check if the update was successful
        if rows_affected == 0:
            return {
                "error": f"No rows updated. Ensure Customer ID {customer_id} exists in the database."
            }

        return {
            "success": f"{field} for customer ID {customer_id} updated to '{new_value}'."
        }

    except sqlite3.Error as e:
        error_message = (
            f"An error occurred while updating the field '{field}': {str(e)}"
        )
        print(f"Error details: {error_message}")
        return {"error": error_message}

    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        return {"error": "An unexpected error occurred."}


@tool
def get_albums_by_artist(artist_name: str):
    """
    Retrieve albums by an artist or similar artists.

    Args:
        artist_name (str): Name of the artist.

    Returns:
        list | dict: Album details or an error message.
    """
    try:
        # Find relevant artists using the retriever
        docs = artist_retriever.get_relevant_documents(artist_name)

        # Check if any artists were found
        if not docs:
            return {
                "error": f"No artists found matching '{artist_name}'. Please try another name."
            }

        # Extract artist IDs from the retrieved documents
        artist_ids = ", ".join([str(d.metadata["ArtistId"]) for d in docs])

        # Query the database for albums by the retrieved artists
        query = f"""
        SELECT 
            albums.Title AS Title, 
            artists.Name AS Name 
        FROM 
            albums 
        LEFT JOIN 
            artists 
        ON 
            albums.ArtistId = artists.ArtistId 
        WHERE 
            albums.ArtistId IN ({artist_ids});
        """
        result = db.run(query, include_columns=True)

        # Check if any albums were found
        if not result:
            return {
                "message": f"No albums found for artists similar to '{artist_name}'."
            }

        return result

    except Exception as e:
        return {"error": f"An error occurred while fetching albums: {str(e)}"}


@tool
def get_tracks_by_artist(artist_name: str):
    """
    Retrieve tracks by an artist or similar artists.

    Args:
        artist_name (str): Name of the artist.

    Returns:
        list | dict: Track details or an error message.
    """
    # Validate retriever initialization
    if artist_retriever is None:
        return {
            "error": "Artist retriever is not initialized. Please ensure the retrievers are set up."
        }

    try:
        # Retrieve relevant artists using the retriever
        docs = artist_retriever.get_relevant_documents(artist_name)

        artist_ids = ", ".join([str(doc.metadata["ArtistId"]) for doc in docs])

        # Query the database for tracks by the retrieved artists (case-insensitive)
        query = f"""
        SELECT 
            tracks.Name AS SongName, 
            artists.Name AS ArtistName 
        FROM 
            albums 
        LEFT JOIN 
            artists 
        ON 
            albums.ArtistId = artists.ArtistId 
        LEFT JOIN 
            tracks 
        ON 
            tracks.AlbumId = albums.AlbumId 
        WHERE 
            LOWER(artists.Name) = LOWER('{artist_name}')
            OR albums.ArtistId IN ({artist_ids});
        """
        result = db.run(query, include_columns=True)

        if not result:
            return {
                "message": f"No tracks found for artists similar to '{artist_name}'."
            }

        return result

    except Exception as e:
        return {"error": f"An error occurred while fetching tracks: {str(e)}"}


@tool
def check_for_songs(song_title: str):
    """
    Search for songs by title using approximate matching.

    Args:
        song_title (str): Title of the song.

    Returns:
        list | dict: Song details or an error message.
    """
    try:
        # Retrieve relevant songs using the retriever
        songs = song_retriever.get_relevant_documents(song_title)

        # Check if any songs were found
        if not songs:
            return {
                "message": f"No songs found matching '{song_title}'. Please try another title."
            }

        return songs

    except Exception as e:
        return {"error": f"An error occurred while searching for songs: {str(e)}"}


def create_music_retrievers(database):
    """
    Create retrievers for artist and track searches.

    Args:
        database (SQLDatabase): Connected database instance.

    Returns:
        tuple: (artist_retriever, song_retriever)
    """
    try:
        # Query the database for artists and tracks
        artists = database._execute("SELECT * FROM artists")
        songs = database._execute("SELECT * FROM tracks")

        # Validate query results
        if not artists:
            raise ValueError("No artists found in the database.")
        if not songs:
            raise ValueError("No tracks found in the database.")

        # Extract artist and track names for embedding
        artist_names = [artist["Name"] for artist in artists]
        track_names = [track["Name"] for track in songs]

        # Create retrievers for artists and songs
        artist_retriever = SKLearnVectorStore.from_texts(
            texts=artist_names, embedding=OpenAIEmbeddings(), metadatas=artists
        ).as_retriever()

        song_retriever = SKLearnVectorStore.from_texts(
            texts=track_names, embedding=OpenAIEmbeddings(), metadatas=songs
        ).as_retriever()

        return artist_retriever, song_retriever

    except Exception as e:
        raise RuntimeError(f"Error creating music retrievers: {str(e)}")


# Define global variables
artist_retriever = None
song_retriever = None


def initialize_retrievers(database):
    """
    Initialize global retrievers for music-related queries.

    Args:
        database (SQLDatabase): Database connection.

    Returns:
        None
    """
    global artist_retriever, song_retriever
    artist_retriever, song_retriever = create_music_retrievers(database)


# Call this once during setup
initialize_retrievers(db)
