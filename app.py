from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pandas as pd
from io import BytesIO
import json
import duckdb

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Flask app setup with explicit CORS configuration (allow all origins for now)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global dict for session data
db_session = {}

# Retrieve API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your environment variables.")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part in the request."}), 400

        file = request.files['file']
        table_name = request.form.get("table_name")

        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected."}), 400
        if not table_name:
            return jsonify({"status": "error", "message": "Table name not provided."}), 400

        # Read file into Pandas DataFrame
        file_stream = BytesIO(file.read())
        df = pd.read_csv(file_stream)

        # Generate a session ID and store DataFrame in memory
        session_id = os.urandom(16).hex()
        db_session[session_id] = {
            'dataframe': df,
            'table_name': table_name
        }

        # Prepare response metadata
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
        # Return error details for debugging (consider hiding in production)
        return jsonify({"status": "error", "message": f"Upload failed with error: {str(e)}"}), 500


if __name__ == "__main__":
    # Use the port assigned by Render or default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
