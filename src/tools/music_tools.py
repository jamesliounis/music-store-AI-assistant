from typing import List, Dict, Union
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.tools import tool

# Hardcoded absolute path to the chinook.db file
db_uri = "sqlite:////Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/data/chinook.db"

# Initialize SQLDatabase with the correct URI
sql_db = SQLDatabase.from_uri(db_uri)


class MusicRetrieverManager:
    """
    A manager class responsible for creating and using retrievers
    for artists and songs via approximate matching in the Chinook database.

    This class wraps the functionality for:
      - Creating artist and song retrievers (vector-based)
      - Finding albums and tracks by artist
      - Searching for songs by approximate title match

    Usage:
    ------
        db = SQLDatabase.from_uri("sqlite:///path/to/chinook.db")
        manager = MusicRetrieverManager(db)
        
        # Retrieve albums by approximate artist name
        albums = manager.get_albums_by_artist("Beatles")

        # Retrieve tracks by approximate artist name
        tracks = manager.get_tracks_by_artist("Foo Fghters")

        # Check for songs by approximate title
        songs = manager.check_for_songs("Rehab")
    """

    def __init__(self, database: SQLDatabase = sql_db, db_path: str = "/Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/data/chinook.db"):
        """
        Initialize the MusicRetrieverManager by querying the database
        for artists and tracks, then building approximate-matching retrievers.
        
        Parameters:
        -----------
        database : SQLDatabase
            An instance of the SQLDatabase connected to the Chinook database.
        """
        self._db = database
        # Will store vector-based retrievers
        self._artist_retriever = None
        self._song_retriever = None

        self._create_retrievers()

    def _create_retrievers(self) -> None:
        """
        Internal method to query the database for artists and tracks,
        then build SKLearnVectorStore-based retrievers for approximate matching.
        """
        try:
            # Query the DB for all artists and tracks
            artists = self._db._execute("SELECT * FROM artists")
            songs = self._db._execute("SELECT * FROM tracks")

            if not artists:
                raise ValueError("No artists found in the database.")
            if not songs:
                raise ValueError("No tracks found in the database.")

            # Extract lists of names for embedding
            artist_names = [artist["Name"] for artist in artists]
            track_names = [track["Name"] for track in songs]

            # Build vector-based retrievers
            artist_store = SKLearnVectorStore.from_texts(
                texts=artist_names,
                embedding=OpenAIEmbeddings(),
                metadatas=artists
            )
            self._artist_retriever = artist_store.as_retriever()

            song_store = SKLearnVectorStore.from_texts(
                texts=track_names,
                embedding=OpenAIEmbeddings(),
                metadatas=songs
            )
            self._song_retriever = song_store.as_retriever()

        except Exception as e:
            raise RuntimeError(f"Error creating music retrievers: {str(e)}")

    def get_albums_by_artist(self, artist_name: str) -> Union[List[Dict], Dict]:
        """
        Retrieve a list of albums by a given artist or similar artists using approximate matching.

        Parameters:
        -----------
        artist_name : str
            The name of the artist to search for.

        Returns:
        --------
        list of dict or dict with an error/message:
            Each dict in the returned list typically includes:
            - "Title": The album title
            - "Name": The artist's name

        Example:
        --------
            manager.get_albums_by_artist("The Beatles") 
            -> [ {"Title": "Revolver", "Name": "The Beatles"}, ... ]
        """
        if not self._artist_retriever:
            return {"error": "Artist retriever is not initialized."}

        try:
            # Use approximate match to find relevant artists
            docs = self._artist_retriever.get_relevant_documents(artist_name)
            if not docs:
                return {"error": f"No artists found matching '{artist_name}'."}

            # Build a list of matching ArtistIds
            artist_ids = ", ".join([str(d.metadata["ArtistId"]) for d in docs])

            # Query all albums belonging to these artist IDs
            query = f"""
            SELECT 
                albums.Title AS Title, 
                artists.Name AS Name 
            FROM 
                albums 
            JOIN 
                artists ON albums.ArtistId = artists.ArtistId
            WHERE 
                albums.ArtistId IN ({artist_ids});
            """
            result = self._db.run(query, include_columns=True)
            if not result:
                return {"message": f"No albums found for artists similar to '{artist_name}'."}

            return result

        except Exception as e:
            return {"error": f"An error occurred while fetching albums: {str(e)}"}

    def get_tracks_by_artist(self, artist_name: str) -> Union[List[Dict], Dict]:
        """
        Retrieve a list of tracks by a given artist or similar artists using approximate matching.
        """
        if not self._artist_retriever:
            return {"error": "Artist retriever is not initialized."}

        try:
            docs = self._artist_retriever.get_relevant_documents(artist_name)
            if not docs:
                return {"message": f"No tracks found for artists similar to '{artist_name}'."}

            artist_ids = ", ".join(str(d.metadata["ArtistId"]) for d in docs)

            query = f"""
            SELECT 
                tracks.Name AS SongName, 
                artists.Name AS ArtistName 
            FROM 
                albums 
            LEFT JOIN 
                artists ON albums.ArtistId = artists.ArtistId
            LEFT JOIN 
                tracks ON tracks.AlbumId = albums.AlbumId
            WHERE 
                LOWER(artists.Name) = LOWER('{artist_name}')
                OR albums.ArtistId IN ({artist_ids});
            """
            result = self._db.run(query, include_columns=True)
            if not result:
                return {"message": f"No tracks found for artists similar to '{artist_name}'."}

            return result

        except Exception as e:
            return {"error": f"An error occurred while fetching tracks: {str(e)}"}

    def check_for_songs(self, song_title: str) -> Union[List[Dict], Dict]:
        """
        Search for songs by approximate title matching using the track retriever.
        """
        if not self._song_retriever:
            return {"error": "Song retriever is not initialized."}

        try:
            docs = self._song_retriever.get_relevant_documents(song_title)
            if not docs:
                return {"message": f"No songs found matching '{song_title}'."}
            return docs  # Typically a list of docs with metadata
        except Exception as e:
            return {"error": f"An error occurred while searching for songs: {str(e)}"}
