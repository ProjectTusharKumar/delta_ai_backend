import os
import re
import ast
import json
import logging
import requests
import spacy
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from rapidfuzz import fuzz, process
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)
# Allow all origins (for development; adjust as needed for production)
CORS(app, resources={r"/*": {"origins": "*"}})

# MongoDB config. DO NOT instantiate client globally to avoid forking issues.
MONGO_URI = os.getenv(
    "MONGO_URI", 
    "mongodb+srv://itstusharkumar15:admin@cluster0.wnyhv.mongodb.net/mydatabase?retryWrites=true&w=majority&appName=Cluster0"
)
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "user_db")

def get_mongo_client():
    logging.debug(f"Connecting to MongoDB using URI: {MONGO_URI}")
    # Create a new MongoClient instance on demand (connection pooling is built-in)
    return MongoClient(MONGO_URI, connect=True)

def get_database():
    client = get_mongo_client()
    try:
        db = client.get_default_database()
    except Exception:
        db = client[MONGO_DB_NAME]
    return db

def check_db_connection():
    try:
        client = get_mongo_client()
        client.admin.command('ping')
        logging.debug("MongoDB connection successful.")
        return True, "MongoDB connection successful."
    except Exception as e:
        logging.error(f"MongoDB connection failed: {str(e)}")
        return False, f"MongoDB connection failed: {str(e)}"

# OpenRouter API config (if needed)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

# Schema and helper configuration
schema_name = [
    "dob", "doj", "salary", "phone number", "skills",
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
    "join date": "doj",
    "hired": "doj",
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
    "dateofjoining": "doj"
}

IGNORED_WORDS = {"both", "me", "can", "is", "and", "the", "for", "to", "of", "on", "please", ",", "retrieve", "fetch", "tell", "show", "whats", "summarize"}
NON_PERSON_WORDS = {"phone", "dob", "date", "number", "details", "projects", "salary", "attendance", "skills", "history"}

def correct_spelling(word):
    corrected = SPELLING_CORRECTIONS.get(word.lower(), word)
    logging.debug(f"Correcting spelling: {word} -> {corrected}")
    return corrected

def extract_names(query):
    query = re.sub(r"(\w+)'s", r"\1", query.strip())
    words = [w for w in query.split() if w.lower() not in IGNORED_WORDS]
    cleaned_query = " ".join(words)
    logging.debug(f"Cleaned query for name extraction: {cleaned_query}")
    doc = nlp(cleaned_query)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if not persons:
        persons = [w for w in words if w.istitle() and w.lower() not in NON_PERSON_WORDS]
    logging.debug(f"Names extracted by spaCy: {persons}")
    return list(set(persons)) if persons else None

def find_best_match(query, query_words):
    found_schema = []
    query_lower = query.lower()
    for phrase, mapped_keyword in SPECIAL_schema_name.items():
        if phrase in query_lower:
            found_schema.append(mapped_keyword)
            logging.debug(f"Found mapped keyword: {phrase} -> {mapped_keyword}")
    for word in map(correct_spelling, query_words):
        result = process.extractOne(word, schema_name, scorer=fuzz.partial_ratio)
        if result:
            match, score = result
            if score > 80 and match in query_lower:
                found_schema.append(match)
                logging.debug(f"Fuzzy match: {word} -> {match} (score: {score})")
    return list(set(found_schema))

# Predefined MongoDB query generator using dynamic projection.
def generate_mongo_query_via_ai(employee_name, requested_fields):
    projection = {field: 1 for field in requested_fields}
    projection["_id"] = 0
    mongo_query = {
        "database": "mydatabase",
        "find": "employees",
        "projection": projection,
        "query": {
            "name": employee_name
        }
    }
    logging.debug(f"Generated MongoDB query: {mongo_query}")
    return mongo_query

def get_employee_data(employee_name, requested_fields):
    mongo_query = generate_mongo_query_via_ai(employee_name, requested_fields)
    logging.debug(f"Executing MongoDB query: {mongo_query}")
    try:
        db = get_database()
        collection = db["employees"]
        employee_exists = collection.find_one({"name": employee_name})
        if not employee_exists:
            return {"error": f"No employee found with name {employee_name}"}
        projected_data = collection.find_one({"name": employee_name}, mongo_query["projection"])
        if not projected_data:
            return {"error": f"No requested fields found for {employee_name}"}
        filtered_data = {k: v for k, v in projected_data.items() if v is not None}
        if not filtered_data:
            return {"error": f"None of the requested fields were found for {employee_name}"}
        return {"mongo_query": mongo_query, "data": filtered_data}
    except Exception as e:
        logging.error(f"MongoDB query failed: {mongo_query}. Error: {str(e)}")
        return {"error": f"Failed to fetch data for {employee_name} from MongoDB."}

def extract_context_and_schema_name(query):
    query = query.strip()
    logging.debug(f"Original query: {query}")
    for wrong, correct in SPELLING_CORRECTIONS.items():
        query = query.replace(wrong, correct)
    context = extract_names(query)
    query_words = query.split()
    found_schema = find_best_match(query, query_words)
    logging.debug(f"Extracted context: {context}, schema: {found_schema}")
    response = {"query": query, "context": context, "schema_name": found_schema}
    if context:
        employee_result = get_employee_data(context[0], [s.lower() for s in found_schema])
        response["employee_data"] = employee_result
    else:
        response["error"] = "No valid employee name found in query."
    return response

# -------------------- API Endpoints --------------------

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    email = data.get("email")
    if not username or not password or not email:
        return jsonify({"error": "All fields are required"}), 400
    db = get_database()
    users_collection = db["users"]
    if users_collection.find_one({"username": username}):
        return jsonify({"error": "Username already exists"}), 400
    hashed_password = generate_password_hash(password)
    users_collection.insert_one({"username": username, "password": hashed_password, "email": email})
    logging.info(f"User {username} registered successfully.")
    return jsonify({"message": "User registered successfully"}), 201

@app.route('/login', methods=['GET'])
def login():
    username = request.args.get("username")
    password = request.args.get("password")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    db = get_database()
    users_collection = db["users"]
    user = users_collection.find_one({"username": username})
    if user and check_password_hash(user["password"], password):
        return jsonify({"message": "Login successful", "email": user["email"]}), 200
    else:
        return jsonify({"error": "Invalid username or password"}), 401

@app.route("/api/check_connection", methods=["GET"])
def api_check_connection():
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
    collection_name = request.form.get('collection_name')
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    if not collection_name:
        return jsonify({"status": "error", "message": "Collection name is required"}), 400
    try:
        df = pd.read_excel(file)
        df.columns = [col.lower() for col in df.columns]
        logging.debug(f"DataFrame columns after lowercasing: {df.columns.tolist()}")
        records = df.to_dict(orient="records")
        db = get_database()
        collection = db[collection_name]
        collection.insert_many(records)
        return jsonify({"status": "success", "message": f"Data uploaded to collection '{collection_name}' successfully!"})
    except Exception as e:
        logging.error(f"Failed to upload data: {str(e)}")
        return jsonify({"status": "error", "message": f"Failed to upload data: {str(e)}"}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    queries = data.get("queries", [])
    results = {}
    for i, query in enumerate(queries, start=1):
        logging.debug(f"Processing query {i}: {query}")
        result = extract_context_and_schema_name(query)
        results[f"query{i}"] = result
    return jsonify(results)

@app.route("/api/collections", methods=["GET"])
def get_all_collections():
    try:
        db = get_database()
        collections = db.list_collection_names()
        return jsonify({"collections": collections})
    except Exception as e:
        logging.error(f"Error fetching collection names: {str(e)}")
        return jsonify({"error": f"Error fetching collection names: {str(e)}"}), 500

@app.route("/api/collection", methods=["GET"])
def get_collection_data():
    collection_name = request.args.get("name")
    if not collection_name:
        return jsonify({"error": "Missing collection name"}), 400
    try:
        db = get_database()
        collection = db[collection_name]
        data = list(collection.find({}, {"_id": 0}))
        return jsonify({"collection_name": collection_name, "data": data})
    except Exception as e:
        logging.error(f"Failed to fetch data from collection {collection_name}: {str(e)}")
        return jsonify({"error": f"Failed to fetch data from collection {collection_name}: {str(e)}"}), 500

@app.route("/api/employees", methods=["GET"])
def get_employees():
    try:
        db = get_database()
        collection = db["employees"]
        data = list(collection.find({}, {"_id": 0}))
        return jsonify({"employees": data})
    except Exception as e:
        logging.error(f"Failed to fetch employees: {str(e)}")
        return jsonify({"error": f"Failed to fetch employees: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
