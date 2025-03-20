import os
import re
import json
import logging
import psycopg2
import requests
import spacy
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from rapidfuzz import fuzz, process
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file (for local testing)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

app = Flask(__name__)
CORS(app)

# Database connection string from environment variable (fallback to default if not set)
DSN = os.getenv("DATABASE_URL", "postgresql://admin:admin123@postgresql-194388-0.cloudclusters.net:19608/gsheet")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("spaCy model loaded.")
except Exception as e:
    logging.error(f"Error loading spaCy model: {e}")
    raise

# PostgreSQL database connection using DSN
def get_db_connection():
    return psycopg2.connect(DSN)

def get_db_engine():
    logging.info(f"Creating SQLAlchemy engine with DSN: {DSN}")
    return create_engine(DSN)

def check_db_connection():
    try:
        conn = get_db_connection()
        conn.close()
        logging.info("Database connection successful.")
        return True, "Database connection successful."
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return False, f"Database connection failed: {e}"

# OpenRouter API configuration from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

# Valid schema names and mapping dictionaries
schema_name = [
    "dob", "DOJ", "salary", "phone number", "skills",
    "attendance", "last year projects", "past projects", "completed projects",
    "currently on", "total projects"
]

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

IGNORED_WORDS = {"both", "me", "can", "is", "and", "the", "for", "to", "of", "on", "please", ",", "retrieve", "fetch", "tell", "show", "whats", "summarize"}
NON_PERSON_WORDS = {"phone", "dob", "date", "number", "details", "projects", "salary", "attendance", "skills", "history"}

def correct_spelling(word):
    corrected = SPELLING_CORRECTIONS.get(word.lower(), word)
    logging.debug(f"Correcting spelling: {word} -> {corrected}")
    return corrected

def extract_names(query):
    query = query.strip()
    query = re.sub(r"(\w+)'s", r"\1", query)
    logging.debug(f"Query after stripping possessives: {query}")

    words = query.split()
    cleaned_words = [word for word in words if word.lower() not in IGNORED_WORDS]
    cleaned_query = " ".join(cleaned_words)
    logging.debug(f"Cleaned query for name extraction: {cleaned_query}")

    doc = nlp(cleaned_query)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    logging.debug(f"Names extracted by spaCy: {persons}")

    if not persons:
        for word in cleaned_words:
            if word.istitle() and word.lower() not in NON_PERSON_WORDS:
                persons.append(word)
        logging.debug(f"Names extracted by fallback: {persons}")
    return list(set(persons)) if persons else None

def find_best_match(query, query_words):
    found_schema = []
    query_lower = query.lower()

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

DB_FIELD_MAPPINGS = {
    "dob": "dob",
    "DOJ": "doj",
    "salary": "salary",
    "phone number": "phone_number",
    "skills": "skills",
    "attendance": "attendance",
    "last year projects": "last_year_projects",
    "past projects": "past_projects",
    "completed projects": "completed_projects",
    "currently on": "currently_on",
    "total projects": "total_projects"
}

def clean_sql_query(query):
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
    ai_generated_query = generate_sql_query_via_ai(employee_name, requested_fields)
    if not ai_generated_query:
        return {"error": "Failed to generate SQL query using AI model."}
    
    logging.info(f"Executing SQL query: {ai_generated_query} with parameter: {employee_name}")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(ai_generated_query, (employee_name,))
        row = cur.fetchone()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        cur.close()
        conn.close()
        logging.info(f"Fetched row: {row}")
        employee_data = dict(zip(columns, row)) if row else {}
        return {"ai_generated_query": ai_generated_query, "data": employee_data}
    except Exception as e:
        logging.error(f"Failed to fetch data for {employee_name}. Error: {e}")
        return {"error": f"Failed to fetch data for {employee_name}. Error: {e}"}

def extract_context_and_schema_name(query):
    query = query.strip()
    logging.info(f"Original query: {query}")
    
    for wrong, correct in SPELLING_CORRECTIONS.items():
        query = query.replace(wrong, correct)
    
    context = extract_names(query)
    query_words = query.split()
    found_schema = find_best_match(query, query_words)
    logging.info(f"Extracted context: {context}, schema: {found_schema}")
    
    response = {"query": query, "context": context, "schema_name": found_schema}
    if context:
        employee_result = get_employee_data(context[0], found_schema)
        response["employee_data"] = employee_result
    else:
        response["error"] = "No valid employee name found in query."
    return response

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    queries = data.get("queries", [])
    results = {}
    for i, query in enumerate(queries, start=1):
        logging.info(f"Processing query {i}: {query}")
        result = extract_context_and_schema_name(query)
        results[f"query{i}"] = result
    return jsonify(results)

@app.route("/api/check_connection", methods=["GET"])
def check_connection():
    connected, message = check_db_connection()
    if connected:
        return jsonify({"connection": True, "message": message})
    else:
        return jsonify({"connection": False, "message": message}), 500

@app.route("/api/table", methods=["GET"])
def get_table_data():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM test_table")
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        data = [dict(zip(columns, row)) for row in rows]
        return jsonify({"table_name": "test_table", "data": data})
    except Exception as e:
        logging.error(f"Failed to fetch table data: {e}")
        return jsonify({"error": f"Failed to fetch table data: {e}"}), 500

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


if __name__ == "__main__":
    # Do not use debug mode in production.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
