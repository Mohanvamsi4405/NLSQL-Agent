from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import duckdb
import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from io import BytesIO
import logging

# Set up logging for better debugging on Render
logging.basicConfig(level=logging.INFO)
logging.getLogger('werkzeug').setLevel(logging.INFO)

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if the API key is available
if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY not found. Please set it in your .env file or Render environment variables.")

app = Flask(__name__)
# Enable CORS for all origins, which is useful for development and some production scenarios
CORS(app)

# A global dictionary to store the DataFrame for the session.
# WARNING: This approach is not persistent. Data will be lost on service restart.
# It is suitable for a single-user, session-based application.
db_session = {}

def format_schema_for_prompt(con, table_name):
    """Retrieves and formats the table schema for the LLM prompt."""
    try:
        schema = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        
        formatted_schema = f"Table '{table_name}' has the following columns:\n"
        for row in schema:
            col_name = row[1]
            col_type = row[2]
            formatted_schema += f"- {col_name} ({col_type})\n"
        return formatted_schema
    except Exception as e:
        logging.error(f"Could not retrieve schema: {e}")
        return f"Could not retrieve schema: {str(e)}"

@app.route("/")
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file upload, reads data into memory, and stores it in the session."""
    logging.info("Received file upload request.")
    if 'file' not in request.files:
        logging.error("No file part in the request.")
        return jsonify({"status": "error", "message": "No file part in the request."}), 400

    file = request.files['file']
    table_name = request.form.get("table_name")

    if file.filename == '':
        logging.error("No file selected.")
        return jsonify({"status": "error", "message": "No file selected."}), 400
    if not table_name:
        logging.error("Table name not provided.")
        return jsonify({"status": "error", "message": "Table name not provided."}), 400

    try:
        file_stream = BytesIO(file.read())
        df = pd.read_csv(file_stream)
        
        session_id = os.urandom(16).hex()
        db_session[session_id] = {
            'dataframe': df,
            'table_name': table_name
        }
        logging.info(f"File uploaded and data loaded. Session ID: {session_id}")

        columns = [{"name": col, "type": str(df.dtypes[col])} for col in df.columns]
        total_rows = len(df)
        preview_data = df.head(5).values.tolist()
        
        return jsonify({
            "status": "success",
            "message": "File uploaded and data loaded into memory.",
            "session_id": session_id,
            "columns": columns,
            "total_rows": total_rows,
            "preview_data": preview_data,
            "table_name": table_name
        })
    except Exception as e:
        logging.error(f"File upload failed: {e}")
        return jsonify({"status": "error", "message": f"Upload failed: {e}"}), 500

@app.route("/ask", methods=["POST"])
def ask_query():
    """Processes a natural language query and returns SQL, explanation, and results."""
    logging.info("Received natural language query request.")
    question = request.form.get("question", "").lower()
    session_id = request.form.get("session_id")
    execute_sql_str = request.form.get("execute_sql", "false")
    execute_sql = execute_sql_str.lower() == 'true'

    if not session_id or session_id not in db_session:
        logging.error("Session not found.")
        return jsonify({"status": "error", "message": "Session not found. Please upload a dataset first."}), 404

    session_data = db_session[session_id]
    df = session_data['dataframe']
    table_name = session_data['table_name']
    
    # Handle schema queries without hitting the LLM
    if any(word in question for word in ["schema", "columns", "data types", "table info", "structure", "describe"]):
        sql_display = f"SELECT * FROM PRAGMA_TABLE_INFO('{table_name}');"
        explanation = f"This query retrieves the schema for the '{table_name}' table, showing column names and their data types."
        try:
            con = duckdb.connect(':memory:')
            con.register(table_name, df)
            cursor = con.execute(sql_display)
            headers = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            results = [headers] + rows
            execution_error = None
        except Exception as e:
            logging.error(f"Schema query execution failed: {e}")
            execution_error = str(e)
            results = None
        finally:
            con.close()
        
        return jsonify({
            "status": "success", "sql_query": sql_display, "explanation": explanation,
            "results": results, "execution_error": execution_error
        })

    # LLM-based query processing
    try:
        con = duckdb.connect(':memory:')
        con.register(table_name, df)
        formatted_schema = format_schema_for_prompt(con, table_name)
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set.")

        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=GROQ_API_KEY)
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", 
                    """You are an expert SQL query generator. Your task is to analyze the following natural language question about the table '{table_name}' and provide a single, valid DuckDB SQL query. This query should be a SELECT statement. Also, provide a clear and concise explanation of what the query does.

The table schema is as follows:
{schema}

Do not return the table schema or PRAGMA queries in the response. Only generate SELECT statements.

When referencing a column name that contains a space or other special characters, always enclose the column name in double quotes (").

Respond with a JSON object containing the `sql_query` and `explanation`.
Example format:
```json
{{
    "sql_query": "SELECT ...",
    "explanation": "This query does..."
}}
```"""),
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
                cursor = con.execute(sql_query)
                headers = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                results = [headers] + rows
            except Exception as e:
                logging.error(f"SQL query execution failed: {e}")
                execution_error = str(e)
    except Exception as e:
        logging.error(f"An error occurred during query processing: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if 'con' in locals():
            con.close()

    return jsonify({
        "status": "success", "sql_query": sql_query, "explanation": explanation,
        "results": results, "execution_error": execution_error
    })

@app.route("/clear_session", methods=["POST"])
def clear_session():
    """Clears the session data from memory."""
    session_id = request.form.get("session_id")
    if session_id in db_session:
        del db_session[session_id]
        logging.info(f"Session {session_id} cleared.")
        return jsonify({"status": "success", "message": "Session cleared."})
    logging.warning("Session not found during clear request.")
    return jsonify({"status": "error", "message": "Session not found."}), 404
