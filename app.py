from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import duckdb
import os
import json
from tempfile import NamedTemporaryFile
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

app = Flask(__name__)
CORS(app)

# A global dictionary to store the database connection for the session.
# This will persist across different requests.
db_session = {}

def get_db_info(file_path, table_name):
    """Creates and returns a DuckDB connection and the table name."""
    try:
        # Determine the correct reading function based on file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Use a temporary, file-based DuckDB database
        db_file_path = file_path.replace(file_ext, '.duckdb')
        con = duckdb.connect(database=db_file_path, read_only=False)
        
        if file_ext == '.csv':
            read_function = f"read_csv_auto('{file_path}')"
        
        else:
            raise Exception(f"Unsupported file format: {file_ext}")
            
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {read_function}")
        return con, table_name, db_file_path
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
    table_name = request.form.get("table_name")

    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected."}), 400
    if not table_name:
        return jsonify({"status": "error", "message": "Table name not provided."}), 400

    temp_file_path = None
    try:
        file_ext = os.path.splitext(file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        con, table_name_actual, db_file_path = get_db_info(temp_file_path, table_name)
        
        columns = [{"name": row[1], "type": row[2]} for row in con.execute(f"PRAGMA table_info('{table_name_actual}')").fetchall()]
        
        # Get total row count
        total_rows = con.execute(f"SELECT COUNT(*) FROM {table_name_actual}").fetchone()[0]
        preview_data = con.execute(f"SELECT * FROM {table_name_actual} LIMIT 20").fetchall()
        
        session_id = os.urandom(16).hex()
        db_session[session_id] = {
            'con': con, 
            'file_path': temp_file_path, 
            'db_file_path': db_file_path,
            'table_name': table_name_actual
        }

        return jsonify({
            "status": "success",
            "message": "File uploaded and schema read successfully.",
            "session_id": session_id,
            "columns": columns,
            "total_rows": total_rows,
            "preview_data": preview_data,
            "table_name": table_name_actual
        })
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask_query():
    """Processes a natural language query and returns SQL, explanation, and results."""
    question = request.form.get("question", "").lower()  # Convert to lowercase for case-insensitive matching
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
        # This is the query that will be shown to the user in the UI.
        sql_display = f"DESCRIBE {table_name};"
        # This is the query that will be executed by DuckDB.
        sql_execution = f"SELECT * FROM PRAGMA_TABLE_INFO('{table_name}')"
        explanation = f"This query retrieves the schema for the '{table_name}' table, showing column names, data types, and other properties."
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
    """Clears the session data and deletes the temporary files."""
    session_id = request.form.get("session_id")
    if session_id in db_session:
        session_data = db_session[session_id]
        con = session_data['con']
        con.close()
        
        file_path = session_data['file_path']
        db_file_path = session_data['db_file_path']
        
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(db_file_path):
            os.remove(db_file_path)
            
        del db_session[session_id]
        return jsonify({"status": "success", "message": "Session cleared."})
    return jsonify({"status": "error", "message": "Session not found."}), 404

if __name__ == "__main__":
    app.run(debug=True, port=8000)
