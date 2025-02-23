from flask import Flask, jsonify, render_template, request
import http.client
import json
import urllib.parse
import os
import pickle
import re
import docx
import pdfplumber
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import mysql.connector

# ---------------------- NLP & Resume Setup ---------------------- #

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Define common skills
common_skills = {
    skill.lower()
    for skill in {
        "Python", "Java", "C++", "C", "JavaScript", "HTML", "CSS",
        "TypeScript", "Swift", "Kotlin", "Go", "Ruby", "PHP", "R", "MATLAB",
        "Perl", "Rust", "Dart", "Scala", "Shell Scripting", "React", "Angular",
        "Vue.js", "Node.js", "Django", "Flask", "Spring Boot", "Express.js",
        "Laravel", "Bootstrap", "TensorFlow", "PyTorch", "Keras",
        "Scikit-learn", "NLTK", "Pandas", "NumPy", "SQL", "MySQL",
        "PostgreSQL", "MongoDB", "Firebase", "Cassandra", "Oracle", "Redis",
        "MariaDB", "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
        "Terraform", "CI/CD", "Jenkins", "Git", "GitHub", "Cybersecurity",
        "Penetration Testing", "Ubuntu", "Ethical Hacking", "Firewalls",
        "Cryptography", "IDS", "Network Security", "Machine Learning",
        "Deep Learning", "Numpy", "Pandas", "Matplotlib", "Computer Vision",
        "NLP", "Big Data", "Hadoop", "Spark", "Data Analytics", "Power BI",
        "Tableau", "Data Visualization", "Reinforcement Learning",
        "Advanced DSA", "DSA", "Data Structures and Algorithm", "DevOps", "ML",
        "DL", "Image Processing", "JIRA", "Postman", "Excel", "Leadership",
        "Problem-Solving", "Communication", "Time Management", "Adaptability",
        "Teamwork", "Presentation Skills", "Critical Thinking",
        "Decision Making", "Public Speaking", "Project Management"
    }
}

# ---------------------- Database Connection (for Resume Storage) ---------------------- #

def get_db_connection(db_name="resume_screening_db"):
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # your mysql password
        database=db_name,
        auth_plugin="mysql_native_password"
    )

# ---------------------- Resume Processing Functions ---------------------- #

def extract_text_from_file(file):
    text = ""
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip() if text.strip() else None

def extract_skills(text):
    extracted_skills = set()
    doc = nlp(text)
    for token in doc:
        if token.text.lower() in common_skills:
            extracted_skills.add(token.text.lower())
    return list(extracted_skills)

def extract_name(text):
    lines = text.split('\n')
    return lines[0].strip() if lines else None

def load_model_and_vectorizer():
    try:
        with open("model.pkl", "rb") as model_file:
            rf = pickle.load(model_file)
        with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
            tfidf = pickle.load(vectorizer_file)
        return rf, tfidf
    except Exception as e:
        print(f"[ERROR] Failed to load model/vectorizer: {e}")
        return None, None

def process_resume(file):
    rf, tfidf = load_model_and_vectorizer()
    if not rf or not tfidf:
        return "[ERROR] ML model is missing!", None, None, None
    
    text = extract_text_from_file(file)
    if not text:
        return "[ERROR] Invalid or unsupported file format!", None, None, None
    
    user_name = extract_name(text)
    extracted_skills = extract_skills(text)
    
    try:
        text_vectorized = tfidf.transform([text])
        predicted_job = rf.predict(text_vectorized)[0]
        return None, predicted_job, extracted_skills, user_name
    except Exception as e:
        return f"[ERROR] Prediction failed: {e}", None, extracted_skills, user_name

def compare_skills(predicted_job, extracted_skills, user_name):
    try:
        conn = get_db_connection("resume_screening_db")
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT skills FROM jobrolesskills WHERE job_role = %s", (predicted_job,))
        job_data = cursor.fetchone()
        
        if not job_data:
            return []
        
        required_skills = set(job_data["skills"].lower().split(", "))
        extracted_skills_set = set(skill.lower() for skill in extracted_skills)
        missing_skills = required_skills - extracted_skills_set
        
        if missing_skills:
            cursor.execute(
                "INSERT INTO recommendskills (name, job_role, missing_skills) VALUES (%s, %s, %s)",
                (user_name, predicted_job, ", ".join(missing_skills))
            )
            conn.commit()
        
        cursor.close()
        conn.close()
        return list(missing_skills)
    except Exception as e:
        print(f"[ERROR] Skill comparison failed: {e}")
        return []

# ---------------------- Job Listings via API ---------------------- #

def fetch_job_listings_from_api(query="developer"):
    """
    Calls the API /search endpoint using the provided query and returns the JSON response as a string.
    """
    conn = http.client.HTTPSConnection("jsearch.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "8f92a3f809msh5eedc068f4e98e7p196ca9jsn325f2d6ae7bf",
        'x-rapidapi-host': "jsearch.p.rapidapi.com"
    }
    # URL-encode the query
    query_param = urllib.parse.quote(query)
    endpoint = f"/search?query={query_param}&country=us"
    conn.request("GET", endpoint, headers=headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")

# ---------------------- Flask Application Setup ---------------------- #

app = Flask(__name__, template_folder="templates")
CORS(app)  # Enable CORS for all routes

@app.route("/api/job-details", methods=["GET"])
def job_details_api():
    # This endpoint is available if you want to fetch job details directly.
    try:
        job_data = fetch_job_listings_from_api()
        return jsonify({"success": True, "data": job_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_job = None
    error_message = None
    extracted_skills = []
    missing_skills = []
    user_name = ""
    job_list = []
    
    if request.method == "POST":
        if "resume" not in request.files:
            error_message = "No file uploaded!"
        else:
            file = request.files["resume"]
            if file.filename == "":
                error_message = "No selected file!"
            else:
                error_message, predicted_job, extracted_skills, user_name = process_resume(file)
                if not error_message:
                    missing_skills = compare_skills(predicted_job, extracted_skills, user_name)
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO resumes (name, skills) VALUES (%s, %s)",
                                       (user_name, ", ".join(extracted_skills)))
                        conn.commit()
                        cursor.close()
                        conn.close()
                    except Exception as db_error:
                        error_message = f"[ERROR] Database error: {db_error}"
    
    # Build the search query based on the extracted skills or predicted job
    if predicted_job:
        search_query = predicted_job
    elif extracted_skills:
        # Join skills into a space-separated string
        search_query = " ".join(extracted_skills)
    else:
        search_query = "developer"
    
    try:
        job_listings_json = fetch_job_listings_from_api(search_query)
        job_listings_data = json.loads(job_listings_json)
        # Adjust the key based on the API response structure; here we assume it returns a list under "data".
        job_list = job_listings_data.get("data", [])
    except Exception as e:
        print(f"[ERROR] Failed to fetch or parse job listings: {e}")
        job_list = []

    return render_template("index.html", 
                           predicted_job=predicted_job or "", 
                           error_message=error_message or "", 
                           extracted_skills=extracted_skills,
                           missing_skills=missing_skills,
                           job_list=job_list)

if __name__ == "__main__":
    app.run(debug=True)
