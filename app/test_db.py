# test_db_connection.py

import sqlite3
from pathlib import Path
import os
from dotenv import load_dotenv
import logging


def test_db_connection(db_uri: str) -> None:
    """
    Test the database connection and list all tables in the SQLite database.

    Parameters:
    ----------
    db_uri : str
        The database URI in the format 'sqlite:////absolute/path/to/db'.
    logger : logging.Logger
        The logger instance for logging messages.
    """
    # Extract the path from the SQLite URI
    if db_uri.startswith("sqlite:///"):
        db_path = db_uri.replace("sqlite:///", "")
    else:
        logger.error("Unsupported database URI. Only SQLite URIs are supported.")
        raise ValueError("Unsupported database URI. Only SQLite URIs are supported.")
    
    # Handle absolute paths (with four slashes)
    if db_uri.startswith("sqlite:////"):
        db_path = db_uri.replace("sqlite:////", "/")
    


    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        logger.info(f"Connected to the database at {db_path}.")
        
        # Execute a query to list all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logger.info(f"Tables in the database: {tables}")
        print(f"Connected to the database. Tables: {tables}")
        
    except sqlite3.Error as e:
        if "unable to open database file" in str(e):
            logger.critical(f"SQLite error: {e}. Check if the database file exists and has correct permissions.")
            print(f"SQLite error: {e}. Please check if the database file exists and has the correct permissions.")
        else:
            logger.error(f"SQLite error: {e}")
            print(f"SQLite error: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
            logger.debug("Cursor closed.")
        if 'conn' in locals():
            conn.close()
            logger.debug("Connection closed.")

def main():
    """
    Main function to load environment variables and test the database connection.
    """
    
    # Determine the path to the .env file
    env_path = Path(__file__).parent.parent / ".env"
    
    # Load environment variables from the .env file
    load_dotenv(dotenv_path=env_path)
    
    # Retrieve the DATABASE_URI from environment variables
    db_uri = os.getenv("DATABASE_URI")
    

    

    
    # Test the database connection
    test_db_connection(db_uri)

if __name__ == "__main__":
    main()

