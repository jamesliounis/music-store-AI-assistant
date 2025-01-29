import unittest
from unittest.mock import patch, MagicMock
from utils.tools import (
    get_customer_info,
    get_user_info,
    update_customer_profile,
    get_albums_by_artist,
    get_tracks_by_artist,
    check_for_songs,
    create_music_retrievers,
    initialize_retrievers,
)
from langchain_core.runnables import RunnableConfig
import sqlite3


class TestTools(unittest.TestCase):

    @patch("utils.tools.db.run")
    def test_get_customer_info_valid_id(self, mock_db_run):
        """Test retrieving customer info with a valid customer ID."""
        mock_db_run.return_value = [{"CustomerId": 1, "Name": "John Doe"}]

        result = get_customer_info(1)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["CustomerId"], 1)
        self.assertEqual(result[0]["Name"], "John Doe")

    def test_get_customer_info_invalid_id(self):
        """Test get_customer_info with an invalid customer ID."""
        result = get_customer_info(-1)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid customer ID. Please provide a valid positive integer.")

    @patch("sqlite3.connect")
    def test_get_user_info_valid_id(self, mock_connect):
        """Test retrieving user info using get_user_info with a valid customer ID."""
        mock_conn = MagicMock()
        mock_cursor = mock_conn.cursor.return_value
        mock_cursor.fetchone.return_value = (1, "John Doe", "john@example.com")
        mock_cursor.description = [("CustomerId",), ("Name",), ("Email",)]
        mock_connect.return_value = mock_conn

        config = {"configurable": {"customer_id": 1}}
        result = get_user_info(config)

        self.assertEqual(result["CustomerId"], 1)
        self.assertEqual(result["Name"], "John Doe")
        self.assertEqual(result["Email"], "john@example.com")

    @patch("sqlite3.connect")
    def test_get_user_info_invalid_id(self, mock_connect):
        """Test get_user_info with an invalid customer ID."""
        mock_conn = MagicMock()
        mock_cursor = mock_conn.cursor.return_value
        mock_cursor.fetchone.return_value = None
        mock_cursor.description = None
        mock_connect.return_value = mock_conn

        config = {"configurable": {"customer_id": 999}}
        result = get_user_info(config)

        self.assertIn("error", result)
        self.assertEqual(result["error"], "No customer found with ID 999")

    @patch("sqlite3.connect")
    def test_update_customer_profile_success(self, mock_connect):
        """Test updating a customer's profile field successfully."""
        mock_conn = MagicMock()
        mock_cursor = mock_conn.cursor.return_value
        mock_cursor.rowcount = 1  # Simulating one row updated
        mock_connect.return_value = mock_conn

        result = update_customer_profile(1, "Email", "new@example.com")
        self.assertIn("success", result)
        self.assertEqual(result["success"], "Email for customer ID 1 updated to 'new@example.com'.")

    @patch("sqlite3.connect")
    def test_update_customer_profile_invalid_field(self, mock_connect):
        """Test update_customer_profile with an invalid field."""
        result = update_customer_profile(1, "InvalidField", "value")
        self.assertIn("error", result)
        self.assertTrue("Invalid field" in result["error"])

    @patch("sqlite3.connect")
    def test_update_customer_profile_no_rows_updated(self, mock_connect):
        """Test update_customer_profile when no rows are updated."""
        mock_conn = MagicMock()
        mock_cursor = mock_conn.cursor.return_value
        mock_cursor.rowcount = 0  # Simulating no rows updated
        mock_connect.return_value = mock_conn

        result = update_customer_profile(1, "Email", "new@example.com")
        self.assertIn("error", result)
        self.assertTrue("No rows updated" in result["error"])

    @patch("utils.tools.db.run")
    @patch("utils.tools.artist_retriever.get_relevant_documents")
    def test_get_albums_by_artist_success(self, mock_retriever, mock_db_run):
        """Test retrieving albums by an artist successfully."""
        mock_retriever.return_value = [{"metadata": {"ArtistId": 1}}]
        mock_db_run.return_value = [
            {"Title": "Revolver", "Name": "The Beatles"},
            {"Title": "Abbey Road", "Name": "The Beatles"},
        ]

        result = get_albums_by_artist("The Beatles")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["Title"], "Revolver")

    @patch("utils.tools.db.run")
    @patch("utils.tools.artist_retriever.get_relevant_documents")
    def test_get_tracks_by_artist_success(self, mock_retriever, mock_db_run):
        """Test retrieving tracks by an artist successfully."""
        mock_retriever.return_value = [{"metadata": {"ArtistId": 1}}]
        mock_db_run.return_value = [
            {"SongName": "Hey Jude", "ArtistName": "The Beatles"},
            {"SongName": "Let It Be", "ArtistName": "The Beatles"},
        ]

        result = get_tracks_by_artist("The Beatles")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["SongName"], "Hey Jude")

    @patch("utils.tools.song_retriever.get_relevant_documents")
    def test_check_for_songs_success(self, mock_retriever):
        """Test searching for songs successfully."""
        mock_retriever.return_value = [{"metadata": {"Title": "Hey Jude"}}]

        result = check_for_songs("Hey Jude")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["metadata"]["Title"], "Hey Jude")

    @patch("utils.tools.create_music_retrievers")
    def test_initialize_retrievers(self, mock_create_retrievers):
        """Test initializing the global music retrievers."""
        mock_create_retrievers.return_value = ("mock_artist_retriever", "mock_song_retriever")

        initialize_retrievers("mock_db")

        from utils.tools import artist_retriever, song_retriever
        self.assertEqual(artist_retriever, "mock_artist_retriever")
        self.assertEqual(song_retriever, "mock_song_retriever")


if __name__ == "__main__":
    unittest.main()

