import os
from dotenv import load_dotenv
import spacy
import re
import psycopg2
import requests
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from rapidfuzz import fuzz, process
import pandas as pd
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

load_dotenv()

app = Flask(__name__)
CORS(app)

# PostgreSQL Database Connection
DB_DSN = "postgresql://neondb_owner:npg_dkVFyg40rWmz@ep-solitary-fog-a5kwywge-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
logging.debug("spaCy model loaded.")

# PostgreSQL database connection using connection string URL
def get_db_connection():
    dsn = DB_DSN
    logging.debug(f"Connecting to DB with DSN: {dsn}")
    return psycopg2.connect(dsn)

def get_db_engine():
    dsn = DB_DSN
    logging.debug(f"Creating SQLAlchemy engine with DSN: {dsn}")
    engine = create_engine(dsn)
    return engine

def check_db_connection():
    """
    Attempt to connect to the PostgreSQL database.
    Returns a tuple (True, message) if successful, otherwise (False, error message).
    """
    try:
        conn = get_db_connection()
        conn.close()
        logging.debug("Database connection successful.")
        return True, "Database connection successful."
    except Exception as e:
        logging.error(f"Database connection failed: {str(e)}")
        return False, f"Database connection failed: {str(e)}"

# OpenRouter API configuration – update with your API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

# Define valid schema names (fields available in the employees table)
schema_name = [
    "dob", "DOJ", "salary", "phone number", "skills",
    "attendance", "last year projects", "past projects", "completed projects",
    "currently on", "total projects"
]

# Special keyword mappings (normalized for consistency)
SPECIAL_schema_name = {
    "date of birth": "dob",
    "dob": "dob",
    "phone": "phone number",
    "phone no": "phone number",
    "work history": "total projects",
    "on going": "currently on",
    "project status": "currently on",
    "currently working on": "currently on",
    "working on recently": "currently on",
    "ongoing ones": "currently on",
    "ongoing projects": "currently on",
    "join date": "DOJ",
    "hired": "DOJ",
    "earning": "salary",
    "paid ": "salary"
}

# Spelling corrections dictionary
SPELLING_CORRECTIONS = {
    "salry": "salary",
    "attndance": "attendance",
    "projetcs": "projects",
    "pastproject": "past projects",
    "past projetc": "past projects",
    "completed prject": "completed projects",
    "lst year project": "last year projects",
    "dateofjoining": "DOJ"
}

# Words to ignore before extracting names
IGNORED_WORDS = {"both", "me", "can", "is", "and", "the", "for", "to", "of", "on", "please", ",", "retrieve", "fetch", "tell", "show", "whats", "summarize"}
NON_PERSON_WORDS = {"phone", "dob", "date", "number", "details", "projects", "salary", "attendance", "skills", "history"}

def correct_spelling(word):
    corrected = SPELLING_CORRECTIONS.get(word.lower(), word)
    logging.debug(f"Correcting spelling: {word} -> {corrected}")
    return corrected

def extract_names(query):
    """Extract all person names from a query, handling multiple names."""
    query = query.strip()
    query = re.sub(r"(\w+)'s", r"\1", query)  # Remove possessive ('s)
    logging.debug(f"Query after stripping possessives: {query}")

    words = query.split()
    cleaned_words = [word for word in words if word.lower() not in IGNORED_WORDS]
    cleaned_query = " ".join(cleaned_words)
    logging.debug(f"Cleaned query for name extraction: {cleaned_query}")

    doc = nlp(cleaned_query)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    logging.debug(f"Names extracted by spaCy: {persons}")

    # Fallback: if spaCy doesn't detect a name, assume title-cased words (excluding known non-person words)
    if not persons:
        for word in cleaned_words:
            if word.istitle() and word.lower() not in NON_PERSON_WORDS:
                persons.append(word)
        logging.debug(f"Names extracted by fallback: {persons}")
    return list(set(persons)) if persons else None

def find_best_match(query, query_words):
    """Use fuzzy matching to find the closest valid keyword from the predefined list."""
    found_schema = []
    query_lower = query.lower()

    # Look for mapped keywords first
    for phrase, mapped_keyword in SPECIAL_schema_name.items():
        if phrase in query_lower:
            found_schema.append(mapped_keyword)
            logging.debug(f"Found mapped keyword: {phrase} -> {mapped_keyword}")

    corrected_words = [correct_spelling(word) for word in query_words]
    logging.debug(f"Corrected query words: {corrected_words}")

    for word in corrected_words:
        result = process.extractOne(word, schema_name, scorer=fuzz.partial_ratio)
        if result:
            match, score = result[0], result[1]
            if score > 80 and match in query_lower:
                found_schema.append(match)
                logging.debug(f"Fuzzy match: {word} -> {match} (score: {score})")
    return list(set(found_schema))

# Database field mapping for PostgreSQL columns (use correct casing as in your database)
DB_FIELD_MAPPINGS = {
    "dob": "Dob",                  # Assuming header is "Dob"
    "doj": "DOJ",                  # Assuming header is "DOJ"
    "salary": "Salary",            # Assuming header is "Salary"
    "phone number": "Phone Number",# Assuming header is "Phone Number"
    "skills": "Skills",
    "attendance": "Attendance",
    "last year projects": "Last Year Projects",
    "past projects": "Past Projects",
    "completed projects": "Completed Projects",
    "currently on": "Currently On",
    "total projects": "Total Projects"
}

def normalize_requested_fields(found_schema):
    """
    Normalize the user or AI returned field names to the exact database column names.
    It uses the DB_FIELD_MAPPINGS dictionary to perform the conversion.
    """
    normalized = []
    for field in found_schema:
        key = field.lower()
        if key in DB_FIELD_MAPPINGS:
            normalized.append(DB_FIELD_MAPPINGS[key])
            logging.debug(f"Normalized field: {field} -> {DB_FIELD_MAPPINGS[key]}")
        else:
            # Fallback: Capitalize each word (this might need adjustment based on your schema)
            fallback = " ".join(word.capitalize() for word in field.split())
            normalized.append(fallback)
            logging.debug(f"Fallback normalized field: {field} -> {fallback}")
    return normalized

def clean_sql_query(query):
    """
    Remove markdown formatting (e.g. triple backticks) from the generated SQL query.
    """
    if query.startswith("```"):
        lines = query.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        query = "\n".join(lines)
    logging.debug(f"Cleaned SQL query: {query}")
    return query.strip()

def generate_sql_query_via_ai(employee_name, requested_fields):
    """
    Use the provided AI model (via OpenRouter) to generate a parameterized PostgreSQL query.
    The prompt instructs the model to generate a query to fetch the specified fields from the test_table table.
    Here, 'name' is used in the WHERE clause.
    """
    prompt = (
        f"Generate a parameterized PostgreSQL query to retrieve the following fields "
        f"from the test_table table for an employee with the name '{employee_name}'. "
        f"The fields to retrieve are: {', '.join(requested_fields)}. "
        "Use %s as a placeholder for the name in the WHERE clause. "
        "Return only the SQL query without any markdown formatting."
    )
    logging.debug(f"Prompt for AI: {prompt}")
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_json = response.json()
        generated_query = response_json["choices"][0]["message"]["content"].strip()
        logging.debug(f"Raw AI generated query: {generated_query}")
        cleaned_query = clean_sql_query(generated_query)
        return cleaned_query
    except Exception as e:
        logging.error(f"Error generating SQL query: {e}")
        return None

def get_employee_data(employee_name, requested_fields):
    """
    Use the AI-generated SQL query to fetch employee data from PostgreSQL.
    Returns both the generated query (for display/debug) and the data.
    """
    ai_generated_query = generate_sql_query_via_ai(employee_name, requested_fields)
    if not ai_generated_query:
        return {"error": "Failed to generate SQL query using AI model."}
    
    logging.debug(f"Executing SQL query: {ai_generated_query} with parameter: {employee_name}")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(ai_generated_query, (employee_name,))
        row = cur.fetchone()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        cur.close()
        conn.close()
        logging.debug(f"Fetched row: {row}")
        employee_data = dict(zip(columns, row)) if row else {}
        return {"ai_generated_query": ai_generated_query, "data": employee_data}
    except Exception as e:
        logging.error(f"Failed to fetch data for {employee_name}. Error: {str(e)}")
        return {"error": f"Failed to fetch data for {employee_name}. Error: {str(e)}"}

def extract_context_and_schema_name(query):
    """
    Process the input query to extract the employee name (context) and relevant keywords.
    Then, use the AI model to generate a PostgreSQL query and fetch the employee data.
    Also returns the original query text.
    """
    query = query.strip()
    logging.debug(f"Original query: {query}")
    
    # Apply spelling corrections
    for wrong, correct in SPELLING_CORRECTIONS.items():
        query = query.replace(wrong, correct)
    
    context = extract_names(query)
    query_words = query.split()
    found_schema = find_best_match(query, query_words)
    logging.debug(f"Extracted context: {context}, schema: {found_schema}")
    
    response = {"query": query, "context": context, "schema_name": found_schema}
    if context:
        # Normalize the requested fields using the DB_FIELD_MAPPINGS
        normalized_fields = normalize_requested_fields(found_schema)
        logging.debug(f"Normalized requested fields: {normalized_fields}")
        employee_result = get_employee_data(context[0], normalized_fields)
        response["employee_data"] = employee_result
    else:
        response["error"] = "No valid employee name found in query."
    return response

@app.route("/api/check_connection", methods=["GET"])
def check_connection():
    """
    API endpoint to check if the database connection is working.
    """
    connected, message = check_db_connection()
    if connected:
        return jsonify({"connection": True, "message": message})
    else:
        return jsonify({"connection": False, "message": message}), 500

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    table_name = request.form.get('table_name')

    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if not table_name:
        return jsonify({"status": "error", "message": "Table name is required"}), 400

    try:
        # Read file using pandas
        df = pd.read_excel(file)

        # Get SQLAlchemy engine instead of psycopg2 connection
        engine = get_db_engine()

        # Use 'replace' to overwrite existing table if it exists
        df.to_sql(table_name, engine, if_exists='replace', index=False)

        return jsonify({"status": "success", "message": f"Data uploaded to table '{table_name}' successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to upload data: {str(e)}"}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    API endpoint that accepts a JSON payload with a list of queries.
    For each query, it returns:
      - The original query text (query)
      - Extracted context (employee names)
      - Detected keywords (fields)
      - The cleaned AI-generated SQL query
      - Data fetched from PostgreSQL based on the query
    """
    data = request.json
    queries = data.get("queries", [])
    results = {}
    for i, query in enumerate(queries, start=1):
        logging.debug(f"Processing query {i}: {query}")
        result = extract_context_and_schema_name(query)
        results[f"query{i}"] = result
    return jsonify(results)

@app.route("/api/tables", methods=["GET"])
def get_all_tables():
    """
    API endpoint to fetch all table names from a specified schema in the database.
    Provide the schema as a query parameter, e.g., /api/tables?schema=databse
    """
    try:
        schema = request.args.get("schema", "public")
        conn = get_db_connection()
        cur = conn.cursor()
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = %s
        """
        cur.execute(query, (schema,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        tables = [row[0] for row in rows]
        return jsonify({"tables": tables})
    except Exception as e:
        logging.error(f"Error fetching table names: {str(e)}")
        return jsonify({"error": f"Error fetching table names: {str(e)}"}), 500

import re

@app.route("/api/table", methods=["GET"])
def get_table_data():
    """
    API endpoint to fetch all data from a user-specified table.
    The table name is provided as a query parameter, e.g.,
      /api/table?name=your_table_name
    """
    table_name = request.args.get("name")
    if not table_name:
        return jsonify({"error": "Missing table name"}), 400

    # Validate the table name to prevent SQL injection.
    # Allow only alphanumeric characters and underscores.
    if not re.match(r"^[A-Za-z0-9_]+$", table_name):
        return jsonify({"error": "Invalid table name"}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Note: Table names cannot be parameterized using placeholders.
        # Ensure that the table name is validated before interpolating.
        query = f"SELECT * FROM {table_name}"
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()

        # Convert rows to a list of dictionaries
        data = [dict(zip(columns, row)) for row in rows]
        return jsonify({"table_name": table_name, "data": data})
    except Exception as e:
        logging.error(f"Failed to fetch data from table {table_name}: {str(e)}")
        return jsonify({"error": f"Failed to fetch data from table {table_name}: {str(e)}"}), 500

@app.route("/api/table_data", methods=["GET"])
def get_table_data_dynamic():
    """
    API endpoint to fetch all data from a specific table.
    Expects a query parameter 'name' (e.g. /api/table_data?name=your_table_name).
    It checks if the table exists in the public schema before executing the query.
    """
    table_name = request.args.get("name")
    if not table_name:
        return jsonify({"error": "No table name provided"}), 400
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Retrieve allowed table names from the public schema
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        allowed_tables = [row[0] for row in cur.fetchall()]
        if table_name not in allowed_tables:
            cur.close()
            conn.close()
            return jsonify({"error": f"Table '{table_name}' not found"}), 404

        # Fetch data from the specified table
        query = f"SELECT * FROM {table_name}"
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        data = [dict(zip(columns, row)) for row in rows]
        return jsonify({"table_name": table_name, "data": data})
    except Exception as e:
        logging.error(f"Failed to fetch data for table {table_name}: {str(e)}")
        return jsonify({"error": f"Failed to fetch table data: {str(e)}"}), 500

@app.route("/api/employees", methods=["GET"])
def get_employees():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM employees")
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()

        data = [dict(zip(columns, row)) for row in rows]
        return jsonify({"employees": data})
    except Exception as e:
        logging.error(f"Failed to fetch employees: {str(e)}")
        return jsonify({"error": f"Failed to fetch employees: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
