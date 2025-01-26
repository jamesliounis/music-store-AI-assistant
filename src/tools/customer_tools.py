import os
import sqlite3
from typing import Any, Dict, List, Union
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase

# Hardcoded absolute path to the chinook.db file
db_uri = "sqlite:////Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/data/chinook.db"

# Initialize SQLDatabase with the correct URI
sql_db = SQLDatabase.from_uri(db_uri)

class CustomerProfileManager:
    """
    A manager class for fetching and updating customer information in the Chinook database.

    This class consolidates:
      - get_user_info (config-based retrieval)
      - get_customer_info (ID-based retrieval)
      - update_customer_profile (updating a single profile field)

    References the Chinook DB through either direct sqlite3 calls or via a SQLDatabase object.

    Usage Example:
    -------------
        db = SQLDatabase.from_uri("sqlite:///path/to/chinook.db")
        manager = CustomerProfileManager(db)

        config_data = {"configurable": {"customer_id": 1}}
        user_info = manager.get_user_info(config_data)
        print(user_info)

        # or direct retrieval by ID:
        info = manager.get_customer_info(1)

        # update a field:
        update_result = manager.update_customer_profile(1, "Email", "new_email@example.com")
        print(update_result)
    """

    def __init__(self, database: SQLDatabase = sql_db, db_path: str = "/Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/data/chinook.db"):
        """
        Initialize the CustomerProfileManager with:
          - A SQLDatabase instance for advanced queries
          - A fallback db_path for direct sqlite3 access if needed

        Parameters:
        -----------
        database : SQLDatabase
            An instance of the SQLDatabase connected to the Chinook database
        db_path : str
            Fallback path to the SQLite database for direct connections (default: "/Users/jamesliounis/Desktop/langchain/music-store-AI-assistant/data/chinook.db")
        """
        self._db = database
        self._db_path = db_path

    @tool
    def get_user_info(self, config: RunnableConfig) -> dict:
        """
        Retrieve customer information from the 'customers' table
        using a 'customer_id' read from 'config["configurable"]["customer_id"]'.

        Returns:
        --------
        dict:
            - A dictionary of column_name -> value if found.
            - {"error": "..."} if no record or invalid input.

        Example:
        --------
            config = {"configurable": {"customer_id": 1}}
            result = get_user_info.invoke(config)
        """
        configuration = config.get("configurable", {})
        customer_id = configuration.get("customer_id", None)
        if not customer_id:
            raise ValueError("No customer_id found in config['configurable'].")

        # direct sqlite3 connection
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM customers WHERE CustomerID = ?"
        cursor.execute(query, (customer_id,))
        row = cursor.fetchone()
        column_names = [desc[0] for desc in cursor.description] if cursor.description else []

        if row:
            result = dict(zip(column_names, row))
        else:
            result = {"error": f"No customer found with ID {customer_id}"}

        cursor.close()
        conn.close()
        return result

    @tool
    def get_customer_info(self, customer_id: int) -> Union[List[Any], Dict[str, str]]:
        """
        Retrieve customer info by ID from the 'customers' table. 
        Returns a list of rows or an error message.

        Parameters:
        -----------
        customer_id : int
            The unique ID of the customer in the database.

        Returns:
        --------
        list of tuples or dict with error:
            If a customer record is found, returns a list of row data.
            Otherwise, returns {"error": "..."}.
        """
        if not isinstance(customer_id, int) or customer_id <= 0:
            return {"error": "Invalid customer ID. Must be a positive integer."}
        
        try:
            query = f"SELECT * FROM customers WHERE CustomerID = {customer_id};"
            result = self._db.run(query, include_columns=False)
            if result:
                return result
            else:
                return {"error": f"No customer found with ID {customer_id}."}
        except Exception as e:
            return {"error": f"An error occurred while fetching customer info: {str(e)}"}

    @tool
    def update_customer_profile(self, customer_id: int, field: str, new_value: str) -> Dict[str, str]:
        """
        Update a specific field (like 'Email', 'Phone', etc.) in a customer's profile.

        Parameters:
        -----------
        customer_id : int
            The customer's unique identifier.
        field : str
            The name of the field to update (e.g., "FirstName", "Email").
        new_value : str
            The new value to assign to the field.

        Returns:
        --------
        dict:
            - {"success": "..."} if update is successful
            - {"error": "..."} if something fails
        """
        print(f"Received inputs - Customer ID: {customer_id}, Field: {field}, New Value: {new_value}")

        # Basic validation
        if not isinstance(customer_id, int) or customer_id <= 0:
            return {"error": "Invalid customer ID. Must be a positive integer."}

        valid_fields = [
            "FirstName", "LastName", "Company", "Address", "City", "State",
            "Country", "PostalCode", "Phone", "Fax", "Email", "SupportRepId"
        ]
        if field not in valid_fields:
            return {"error": f"Invalid field '{field}'. Allowed fields: {', '.join(valid_fields)}"}

        if not isinstance(new_value, str) or not new_value.strip():
            return {"error": "Invalid 'new_value'. Provide a non-empty string."}

        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            query = f"UPDATE customers SET {field} = ? WHERE CustomerID = ?;"
            print(f"Executing query: {query} with params: ({new_value}, {customer_id})")

            cursor.execute(query, (new_value, customer_id))
            conn.commit()
            rows_affected = cursor.rowcount

            cursor.close()
            conn.close()

            if rows_affected == 0:
                return {"error": f"No rows updated. Ensure Customer ID {customer_id} exists in the database."}

            return {"success": f"{field} for customer ID {customer_id} updated to '{new_value}'."}

        except sqlite3.Error as e:
            error_message = f"SQLite error updating field '{field}': {str(e)}"
            print(f"Error details: {error_message}")
            return {"error": error_message}
        except Exception as e:
            print(f"Unhandled exception: {str(e)}")
            return {"error": "An unexpected error occurred."}
