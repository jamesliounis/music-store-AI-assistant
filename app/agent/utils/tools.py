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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("/Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/app/agent/logs/app.log"),  
    ]
)

logger = logging.getLogger(__name__)

# Initialize Database
DATABASE_URI ="sqlite:////Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/data/chinook.db"
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
    Retrieve customer information from the database using their unique customer ID.

    This tool queries the 'customers' table in the Chinook database to fetch information 
    about a specific customer. It is essential to ensure that the customer ID is provided 
    and is valid before invoking this function.

    Parameters:
    ----------
    customer_id : int
        The unique identifier for the customer in the database.


    Returns:
    -------
    list or dict
        If a customer record is found, returns a list containing one tuple (the row).
        If no record is found or there's an error, returns a dict with an "error" key.
        
    Notes:
    ------
    - ALWAYS confirm that the customer ID is available and valid before calling this function.
    - If the customer ID is invalid or does not exist, the function will return an appropriate 
      error message.
    - Example usage:
        `get_customer_info(1)` will information for the customer with ID 1.
    """
    # Validate that a customer ID is provided
    if not isinstance(customer_id, int) or customer_id <= 0:
        return {"error": "Invalid customer ID. Please provide a valid positive integer."}
    
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
    Retrieve customer information from the 'customers' table
    using a 'customer_id' read from 'config["configurable"]["customer_id"]'.

    If the record is found, returns a dictionary of column_name -> value.
    Otherwise, returns {'error': ...}.

    Example usage:
    --------------
    config = {"configurable": {"customer_id": 1}}
    result = get_user_info.invoke(config)

    The function will look up the customer with ID=1.
    """
    configuration = config.get("configurable", {})
    customer_id = configuration.get("customer_id", None)
    if not customer_id:
        raise ValueError("No customer_id configured in config['configurable'].")

    # Connect to your DB
    conn = sqlite3.connect(DATABASE_URI)
    cursor = conn.cursor()

    # Query the customer row
    query = "SELECT * FROM customers WHERE CustomerID = ?"
    cursor.execute(query, (customer_id,))
    row = cursor.fetchone()
    column_names = [desc[0] for desc in cursor.description] if cursor.description else []

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
    Update a specific field in a customer's profile.

    Parameters:
    ----------
    customer_id : int
        The unique identifier for the customer in the database.
    field : str
        The name of the field to update (e.g., 'FirstName', 'LastName', 'Email').
    new_value : str
        The new value to update in the specified field.

    Returns:
    -------
    dict
        A dictionary containing a success message or an error message.
    """
    print(f"Received inputs - Customer ID: {customer_id}, Field: {field}, New Value: {new_value}")

    # Validate inputs
    if not isinstance(customer_id, int) or customer_id <= 0:
        return {"error": "Invalid customer ID. Please provide a valid positive integer."}

    valid_fields = [
        "FirstName", "LastName", "Company", "Address", "City", "State",
        "Country", "PostalCode", "Phone", "Fax", "Email", "SupportRepId"
    ]
    if field not in valid_fields:
        return {"error": f"Invalid field '{field}'. Allowed fields are: {', '.join(valid_fields)}"}
    
    if not isinstance(new_value, str) or len(new_value.strip()) == 0:
        return {"error": "Invalid value. Please provide a valid new value."}

    try:
        # Connect to the database directly
        conn = sqlite3.connect(DATABASE_URI)  
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
            return {"error": f"No rows updated. Ensure Customer ID {customer_id} exists in the database."}
        
        return {"success": f"{field} for customer ID {customer_id} updated to '{new_value}'."}

    except sqlite3.Error as e:
        error_message = f"An error occurred while updating the field '{field}': {str(e)}"
        print(f"Error details: {error_message}")
        return {"error": error_message}

    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        return {"error": "An unexpected error occurred."}

@tool
def get_albums_by_artist(artist_name: str):
    """
    Retrieve a list of albums by a given artist or similar artists using approximate matching.

    This tool leverages the `artist_retriever` to find artists whose names closely match 
    the provided input and then queries the database to get the albums associated with 
    those artists.

    Parameters:
    ----------
    artist_name : str
        The name of the artist to search for.

    Returns:
    -------
    list
        A list of dictionaries, where each dictionary contains:
        - Title: The album title.
        - Name: The artist's name.

    Notes:
    ------
    - If no matching artists are found, an appropriate message will be returned.
    - Uses approximate matching via the `artist_retriever` to handle typos and partial matches.
    - Example usage:
        `get_albums_by_artist("The Beatles")` retrieves albums for "The Beatles" or similar artists.

    Example Response:
    -----------------
        [
            {"Title": "Revolver", "Name": "The Beatles"},
            {"Title": "Abbey Road", "Name": "The Beatles"}
        ]
    """
    try:
        # Find relevant artists using the retriever
        docs = artist_retriever.get_relevant_documents(artist_name)
        
        # Check if any artists were found
        if not docs:
            return {"error": f"No artists found matching '{artist_name}'. Please try another name."}
        
        # Extract artist IDs from the retrieved documents
        artist_ids = ", ".join([str(d.metadata['ArtistId']) for d in docs])
        
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
            return {"message": f"No albums found for artists similar to '{artist_name}'."}
        
        return result

    except Exception as e:
        return {"error": f"An error occurred while fetching albums: {str(e)}"}

@tool
def get_tracks_by_artist(artist_name: str):
    """
    Retrieve a list of tracks by a given artist or similar artists using approximate matching.
    """
    # Validate retriever initialization
    if artist_retriever is None:
        return {"error": "Artist retriever is not initialized. Please ensure the retrievers are set up."}

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
            return {"message": f"No tracks found for artists similar to '{artist_name}'."}
        
        return result

    except Exception as e:
        return {"error": f"An error occurred while fetching tracks: {str(e)}"}

@tool
def check_for_songs(song_title: str):
    """
    Search for songs by title using approximate matching.

    This tool uses the `song_retriever` to find songs whose titles closely match 
    the provided input. It returns relevant information about the songs.

    Parameters:
    ----------
    song_title : str
        The title of the song to search for.

    Returns:
    -------
    list
        A list of dictionaries with song details or a message if no matches are found.

    Notes:
    ------
    If no exact match is found, it returns similar titles.
    """
    try:
        # Retrieve relevant songs using the retriever
        songs = song_retriever.get_relevant_documents(song_title)
        
        # Check if any songs were found
        if not songs:
            return {"message": f"No songs found matching '{song_title}'. Please try another title."}
        
        return songs

    except Exception as e:
        return {"error": f"An error occurred while searching for songs: {str(e)}"}

def create_music_retrievers(database):
    """
    Create retrievers for looking up artists and tracks using approximate matching.

    This function uses a vector-based search mechanism to create retrievers for artists and 
    track names from the Chinook database. It enables efficient and error-tolerant lookups 
    of artist and track names without requiring exact spelling.

    Parameters:
    ----------
    database : SQLDatabase
        An instance of the SQLDatabase connected to the Chinook database.

    Returns:
    -------
    tuple
        A tuple containing:
        - artist_retriever: A retriever for searching artist names.
        - song_retriever: A retriever for searching track names.

    Notes:
    ------
    - The function queries the database for artists and tracks, retrieves their names, 
      and indexes them into separate retrievers.
    - OpenAI embeddings are used for generating vector representations of the names.
    - The retrievers allow for approximate matching, making it user-friendly for misspelled 
      or partially remembered artist/track names.

    Example Usage:
    --------------
        db = SQLDatabase.from_uri(DB_URI)
        artist_retriever, song_retriever = create_music_retrievers(db)
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
        artist_names = [artist['Name'] for artist in artists]
        track_names = [track['Name'] for track in songs]

        # Create retrievers for artists and songs
        artist_retriever = SKLearnVectorStore.from_texts(
            texts=artist_names,
            embedding=OpenAIEmbeddings(),
            metadatas=artists
        ).as_retriever()

        song_retriever = SKLearnVectorStore.from_texts(
            texts=track_names,
            embedding=OpenAIEmbeddings(),
            metadatas=songs
        ).as_retriever()

        return artist_retriever, song_retriever

    except Exception as e:
        raise RuntimeError(f"Error creating music retrievers: {str(e)}")

# Define global variables
artist_retriever = None
song_retriever = None

def initialize_retrievers(database):
    global artist_retriever, song_retriever
    artist_retriever, song_retriever = create_music_retrievers(database)

# Call this once during setup
initialize_retrievers(db)

