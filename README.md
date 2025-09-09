NL-SQL-Agent
NL-SQL-Agent is an interactive web application that serves as a natural language interface for data analysis. It allows users to upload a CSV file and query their data using plain English, which is then translated into executable SQL by a large language model.

✨ Features
Natural Language to SQL: Translate plain English questions into valid DuckDB SQL queries.

Dynamic Data Upload: Easily upload a CSV file to create a dynamic, in-memory database.

Real-time Results: Execute the generated SQL query and display the results in a clean, readable table.

Intuitive UI: A user-friendly, single-page web interface with a query history and a clean aesthetic.

Session Management: Maintain a persistent session for data and queries until the user clears it.

💻 Technologies
Backend:

Python 3.10+: The core programming language for the application.

Flask: A micro-framework for the backend web server.

DuckDB: A high-performance, in-process analytical database for fast data processing.

Groq API: Provides access to the large language model for natural language processing.

Gunicorn: A production-ready WSGI server for deployment.

Frontend:

HTML, CSS, JavaScript: A single, self-contained index.html file handles the entire user interface.

🚀 Getting Started
Follow these steps to get the project up and running on your local machine.

Prerequisites
Python 3.10 or higher

pip (Python package installer)

A Groq API key

1. Clone the Repository
git clone [https://github.com/your-username/NL-SQL-Agent.git](https://github.com/your-username/NL-SQL-Agent.git)
cd NL-SQL-Agent

2. Set Up the Environment
Create a virtual environment and install the required dependencies.

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
pip install -r requirements.txt

3. Configure API Key
Create a .env file in the root directory of your project and add your Groq API key:

GROQ_API_KEY="your-groq-api-key-here"

4. Run the Application
Start the Flask development server.

python app.py

The application will be accessible at http://127.0.0.1:8000.

🌐 Deployment
The application is configured for easy deployment to platforms like Render. The requirements.txt and gunicorn dependency are included to ensure a smooth deployment process.

Start Command: gunicorn app:app

This README provides a clear overview and all the instructions needed for anyone to understand, set up, and run your project.