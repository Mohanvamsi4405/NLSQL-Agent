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
from tempfile import NamedTemporaryFile

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

app = Flask(__name__)
CORS(app)

# A global dictionary to store the DataFrame for the session.
# This makes the application stateless with respect to the filesystem.
db_session = {}

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
    """Handles file upload, reads data into memory, and stores it in the session."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request."}), 400

    file = request.files['file']
    table_name = request.form.get("table_name")

    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected."}), 400
    if not table_name:
        return jsonify({"status": "error", "message": "Table name not provided."}), 400

    try:
        # Read the uploaded file directly into a BytesIO stream
        file_stream = BytesIO(file.read())
        
        # Read the stream into a Pandas DataFrame. This is the key change.
        df = pd.read_csv(file_stream)
        
        session_id = os.urandom(16).hex()
        # Store the DataFrame itself, not a file path
        db_session[session_id] = {
            'dataframe': df,
            'table_name': table_name
        }

        # Get metadata directly from the DataFrame
        columns = [{"name": col, "type": str(df.dtypes[col])} for col in df.columns]
        total_rows = len(df)
        preview_data = df.head(20).values.tolist()
        
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
    df = session_data['dataframe']
    table_name = session_data['table_name']
    
    # Check for schema-related queries and bypass the LLM
    schema_keywords = ["schema", "columns", "data types", "table info", "structure", "describe"]
    if any(word in question for word in schema_keywords):
        sql_display = f"DESCRIBE {table_name};"
        explanation = f"This query retrieves the schema for the '{table_name}' table, showing column names, data types, and other properties."
        
        try:
            # Connect to a new in-memory DB for this specific request
            con = duckdb.connect(':memory:')
            con.register(table_name, df)
            
            # Execute the query
            cursor = con.execute(f"SELECT * FROM PRAGMA_TABLE_INFO('{table_name}')")
            headers = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            results = [headers] + rows
            execution_error = None
        except Exception as e:
            execution_error = str(e)
            results = None
        finally:
            con.close()
        
        return jsonify({
            "status": "success",
            "sql_query": sql_display,
            "explanation": explanation,
            "results": results,
            "execution_error": execution_error
        })

    # Normal LLM-based query processing
    try:
        # Connect to a new in-memory DB for this specific request
        con = duckdb.connect(':memory:')
        con.register(table_name, df)
        
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
                # Execute the query on the in-memory connection
                cursor = con.execute(sql_query)
                headers = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                results = [headers] + rows
            except Exception as e:
                execution_error = str(e)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        # Ensure the connection is closed after every request
        if 'con' in locals():
            con.close()

    return jsonify({
        "status": "success",
        "sql_query": sql_query,
        "explanation": explanation,
        "results": results,
        "execution_error": execution_error
    })

@app.route("/clear_session", methods=["POST"])
def clear_session():
    """Clears the session data from memory."""
    session_id = request.form.get("session_id")
    if session_id in db_session:
        # Simply delete the entry from the in-memory dictionary
        del db_session[session_id]
        return jsonify({"status": "success", "message": "Session cleared."})
    return jsonify({"status": "error", "message": "Session not found."}), 404

if __name__ == "__main__":
    app.run(debug=True, port=os.environ.get("PORT", 8000))
