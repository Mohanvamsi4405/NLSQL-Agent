from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import duckdb
import os
import json
import re
from io import StringIO
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Retrieve the API key from the environment, assuming it's set in production.
# If not found, fall back to loading from a .env file for local development.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        load_dotenv()
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found. Please set it in your environment or a .env file.")
    except Exception as e:
        # Fallback for environments where dotenv might not be available
        raise ValueError(f"GROQ_API_KEY not found and dotenv failed to load it: {e}")

app = Flask(__name__)
CORS(app)

# A global dictionary to store the database connection for the session.
# This will persist across different requests within the same process.
db_session = {}

def get_db_connection(uploaded_file):
    """Creates and returns an in-memory DuckDB connection with the data."""
    try:
        # Read the uploaded file's content into a Pandas DataFrame
        string_io = StringIO(uploaded_file.read().decode('utf-8'))
        df = pd.read_csv(string_io)

        # Create an in-memory DuckDB connection
        con = duckdb.connect(':memory:')
        
        # Register the DataFrame as a table in the in-memory database
        con.register('data_table', df)
        
        return con
    except Exception as e:
        raise Exception(f"Failed to create database from file: {str(e)}")

def format_schema_for_prompt(con, table_name):
    """Retrieves and formats the table schema for the LLM prompt."""
    try:
        schema_query = f"PRAGMA table_info('{table_name}')"
        schema = con.execute(schema_query).fetchall()
        
        formatted_schema = f"Table '{table_name}' has the following columns:\n"
        for row in schema:
            col_name = row[1]
            col_type = row[2]
            formatted_schema += f"- {col_name} ({col_type})\n"
        return formatted_schema
    except Exception as e:
        return f"Could not retrieve schema: {str(e)}"

@app.route("/")
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file upload, reads schema, and stores it in-memory."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request."}), 400

    file = request.files['file']
    table_name = request.form.get("table_name", "data_table")

    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected."}), 400

    try:
        # Create an in-memory DuckDB connection and load the data
        con = get_db_connection(file)
        
        columns = [{"name": row[1], "type": row[2]} for row in con.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
        
        # Get total row count
        total_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        preview_data = con.execute(f"SELECT * FROM {table_name} LIMIT 20").fetchall()
        
        # A unique ID for this session
        session_id = os.urandom(16).hex()
        db_session[session_id] = {
            'con': con, 
            'table_name': table_name
        }

        return jsonify({
            "status": "success",
            "message": "File uploaded and schema read successfully.",
            "session_id": session_id,
            "columns": columns,
            "total_rows": total_rows,
            "preview_data": preview_data,
            "table_name": table_name
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask_query():
    """Processes a natural language query and returns SQL, explanation, and results."""
    question = request.form.get("question", "").lower()
    session_id = request.form.get("session_id")
    execute_sql_str = request.form.get("execute_sql", "false")
    execute_sql = execute_sql_str.lower() == 'true'

    if not session_id or session_id not in db_session:
        return jsonify({"status": "error", "message": "Session not found. Please upload a dataset first."}), 404

    session_data = db_session[session_id]
    con = session_data['con']
    table_name = session_data['table_name']
    
    # Check for schema-related queries and bypass the LLM
    schema_keywords = ["schema", "columns", "data types", "table info", "structure", "describe"]
    if any(word in question for word in schema_keywords):
        sql_display = f"PRAGMA table_info('{table_name}')"
        sql_execution = sql_display
        explanation = f"This query retrieves the schema for the '{table_name}' table, showing column names and data types."
        try:
            cursor = con.execute(sql_execution)
            headers = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            results = [headers] + rows
            execution_error = None
        except Exception as e:
            execution_error = str(e)
            results = None
            
        return jsonify({
            "status": "success",
            "sql_query": sql_display,
            "explanation": explanation,
            "results": results,
            "execution_error": execution_error
        })

    # Normal LLM-based query processing
    try:
        formatted_schema = format_schema_for_prompt(con, table_name)
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=GROQ_API_KEY)
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", 
                     """
You are an expert SQL query generator. Your task is to analyze the following natural language question about the table '{table_name}' and provide a single, valid DuckDB SQL query. This query should be a SELECT statement. Also, provide a clear and concise explanation of what the query does.

The table schema is as follows:
{schema}

Do not return the table schema or PRAGMA queries in the response, even if the user asks for it. The schema has already been provided to you. Only generate SELECT statements.

When referencing a column name that contains a space or other special characters, always enclose the column name in double quotes (").

Respond with a JSON object containing the `sql_query` and `explanation`.
Example format:
```json
{{
    "sql_query": "SELECT ...",
    "explanation": "This query does..."
}}
```
                    """),
                ("human", "Natural language question: {question}")
            ]
        )
        
        chain = prompt_template | llm
        response = chain.invoke({
            "question": question, 
            "table_name": table_name, 
            "schema": formatted_schema
        })

        json_match = re.search(r'```json\n(.*?)```', response.content, re.DOTALL)
        if json_match:
            generated_json = json_match.group(1).strip()
        else:
            generated_json = response.content.strip()

        try:
            parsed_response = json.loads(generated_json)
            sql_query = parsed_response.get('sql_query', 'Error generating SQL.')
            explanation = parsed_response.get('explanation', 'Failed to generate a valid explanation.')
        except json.JSONDecodeError:
            sql_query = "Error generating SQL."
            explanation = f"LLM returned invalid JSON. Response was: {generated_json}"
        
        results = None
        execution_error = None
        if execute_sql and sql_query and "Error" not in sql_query:
            try:
                # Execute the query and get the result cursor
                cursor = con.execute(sql_query)
                
                # Get the column names from the cursor's description, which always exists
                headers = [desc[0] for desc in cursor.description]
                
                # Fetch all rows
                rows = cursor.fetchall()
                
                # Combine headers and rows for the final result
                results = [headers] + rows
            except Exception as e:
                execution_error = str(e)

        return jsonify({
            "status": "success",
            "sql_query": sql_query,
            "explanation": explanation,
            "results": results,
            "execution_error": execution_error
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/clear_session", methods=["POST"])
def clear_session():
    """Clears the session data."""
    session_id = request.form.get("session_id")
    if session_id in db_session:
        session_data = db_session[session_id]
        con = session_data['con']
        con.close()
        del db_session[session_id]
        return jsonify({"status": "success", "message": "Session cleared."})
    return jsonify({"status": "error", "message": "Session not found."}), 404
