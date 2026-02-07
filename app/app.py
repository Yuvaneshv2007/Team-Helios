import os
import sys
import requests
import json
import re
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import easyocr
from functools import wraps
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai

# --- DB Import ---
# Importing the new Firestore 'Student' class from your models.py
from models import Student, db

# -------------------------------
# LLaMA / OpenRouter Config
# -------------------------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3-8b-instruct"
GITHUB_API_BASE = "https://api.github.com"

if not API_KEY:
    print("WARNING: OPENROUTER_API_KEY not set in .env")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.secret_key = 'super_secret_key'

# --- Configuration ---
basedir = os.path.abspath(os.path.dirname(__file__))
# SQLALCHEMY CONFIG REMOVED

# UPLOAD_FOLDER REMOVED (No longer needed since we aren't saving files)

# --- Lazy Loading OCR Reader ---
reader_instance = None

def get_reader():
    global reader_instance
    if reader_instance is None:
        print("Initializing OCR engine... (This happens only once)")
        reader_instance = easyocr.Reader(['en'], gpu=False)
    return reader_instance

# --- Helper: LeetCode Stats ---
def get_leetcode_topic_stats(username):
    url = "https://leetcode.com/graphql"
    query = """
    query skillStats($username: String!) {
      matchedUser(username: $username) {
        tagProblemCounts {
          advanced { tagName problemsSolved }
          intermediate { tagName problemsSolved }
          fundamental { tagName problemsSolved }
        }
      }
    }
    """
    variables = {"username": username}
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.post(url, json={"query": query, "variables": variables}, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "errors" in data or not data.get("data", {}).get("matchedUser"):
                return None
            return data["data"]["matchedUser"]["tagProblemCounts"]
    except Exception as e:
        print(f"LeetCode Error: {e}")
    return None

# --- Helper: Send Question to LLaMA ---
def ask_llama(context_text, question, max_tokens=1000):
    """Universal helper to communicate with LLaMA via OpenRouter."""
    if not API_KEY:
        return "API Key missing."
        
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a specialized Career & Tech Lead AI. Return strictly valid JSON when requested. No conversational filler or markdown code blocks."
            },
            {
                "role": "user",
                "content": f"Context/Resume:\n{context_text}\n\nQuestion/Task:\n{question}"
            }
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        return f"Error: {response.text}"
    except Exception as e:
        return f"Request Failed: {str(e)}"

# -------------------------------
# GITHUB UTILITIES
# -------------------------------
def fetch_github_data(username):
    """Fetch repos and their details from GitHub API."""
    try:
        res = requests.get(f"{GITHUB_API_BASE}/users/{username}/repos", timeout=5)
        if res.status_code != 200: return []
        
        repos = []
        for repo in res.json():
            # Get languages for each repo
            lang_res = requests.get(repo["languages_url"], timeout=5)
            langs = list(lang_res.json().keys()) if lang_res.status_code == 200 else []
            
            repos.append({
                "name": repo["name"],
                "description": repo["description"] or "No description provided.",
                "languages": langs,
                "url": repo["html_url"]
            })
        return repos
    except Exception as e:
        print(f"GitHub Fetch Error: {e}")
        return []

# -------------------------------
# ROLE → PROJECT EXPECTATION MAP
# -------------------------------
ROLE_PROJECT_REQUIREMENTS = {
    "software": {
        "critical": {
            "REST API": ["api", "flask", "fastapi", "node", "express", "django"],
            "Database Integration": ["sql", "mongodb", "firebase", "postgres", "sqlite"],
            "Authentication System": ["jwt", "auth", "oauth", "login"]
        },
        "optional": {
            "Cloud Deployment": ["docker", "aws", "kubernetes", "vercel"],
            "Microservices": ["microservice", "grpc"]
        }
    },
    "machine learning": {
        "critical": {
            "ML Model Project": ["classification", "regression", "scikit", "sklearn", "ml model"],
            "Deep Learning Project": ["cnn", "rnn", "lstm", "transformer", "pytorch", "tensorflow", "keras"],
            "Data Pipeline / EDA": ["pandas", "numpy", "eda", "data analysis", "cleaning"]
        },
        "optional": {
            "Deployment": ["flask", "fastapi", "streamlit", "gradio"],
            "Computer Vision": ["opencv", "yolo", "vision"]
        }
    },
    "web": {
        "critical": {
            "Frontend App": ["react", "vue", "angular", "html", "css", "tailwind", "bootstrap"],
            "Backend API": ["node", "express", "flask", "django", "php"],
            "Database": ["sql", "mongo", "firebase"]
        },
         "optional": {
            "Full Stack": ["mern", "mean", "fullstack"],
            "Authentication": ["auth", "jwt"]
        }
    },
    "embedded": {
        "critical": {
             "Microcontroller": ["arduino", "esp32", "stm32", "raspberry", "avr"],
             "Sensors & Actuators": ["sensor", "motor", "iot", "circuit"],
             "Communication Protocols": ["i2c", "spi", "uart", "mqtt"]
        },
        "optional": {
             "RTOS": ["freertos", "rtos"],
             "PCB Design": ["kicad", "eagle", "pcb"]
        }
    }
}

# -------------------------------
# NORMALIZE REPO TEXT
# -------------------------------
def normalize_repo_text(repo):
    """
    Convert repo content into searchable lowercase text
    """
    return (
        repo["name"] + " " +
        repo["description"] + " " +
        " ".join(repo["languages"])
    ).lower()

# -------------------------------
# CORE ANALYSIS FUNCTION (RULE BASED)
# -------------------------------
def analyze_github_rule_based(repos, target_role_key):
    """
    Analyze GitHub repositories based on target career role using strict rules.
    """
    # Normalize key
    target_role_key = target_role_key.lower() if target_role_key else "software"
    
    if 'data' in target_role_key or 'ai' in target_role_key or 'ml' in target_role_key: target_role_key = 'machine learning'
    elif 'web' in target_role_key: target_role_key = 'web'
    elif 'embedded' in target_role_key or 'hardware' in target_role_key: target_role_key = 'embedded'
    elif 'software' in target_role_key: target_role_key = 'software'
    else: target_role_key = 'software' # Default

    role_config = ROLE_PROJECT_REQUIREMENTS.get(target_role_key, ROLE_PROJECT_REQUIREMENTS['software'])

    critical = role_config["critical"]
    optional = role_config["optional"]

    covered_critical = set()
    covered_optional = set()
    repo_analysis_map = {} # Map repo name to matched domains

    detected_domains = set()

    for repo in repos:
        text = normalize_repo_text(repo)
        matched_projects = []
        
        # Simple domain detection for classification
        is_web = any(x in text for x in ['html', 'css', 'react', 'node'])
        is_ml = any(x in text for x in ['pandas', 'torch', 'sklearn', 'model'])
        is_embedded = any(x in text for x in ['arduino', 'iot', 'esp'])
        
        domain_label = "General"
        if is_web: domain_label = "Web Dev"
        elif is_ml: domain_label = "AI/ML"
        elif is_embedded: domain_label = "Embedded/IoT"
        
        if domain_label != "General":
            detected_domains.add(domain_label)

        # Check critical requirements
        for project, keywords in critical.items():
            if any(keyword in text for keyword in keywords):
                covered_critical.add(project)
                matched_projects.append(project)

        # Check optional boosters
        for project, keywords in optional.items():
            if any(keyword in text for keyword in keywords):
                covered_optional.add(project)
                matched_projects.append(project)
        
        repo_analysis_map[repo['name']] = {
            'domain': domain_label,
            'matched': matched_projects
        }

    missing_critical = [
        project for project in critical.keys()
        if project not in covered_critical
    ]

    # Readiness scoring logic
    if not missing_critical:
        readiness = "Job Ready"
    elif len(missing_critical) <= 1:
        readiness = "Good Progress"
    else:
        readiness = "Needs Improvement"

    return {
        "repo_map": repo_analysis_map,
        "detected_domains": list(detected_domains),
        "missing_projects": missing_critical,
        "career_readiness": readiness,
        "optional_suggestions": [k for k in optional.keys() if k not in covered_optional]
    }

# --- Helper: Connect to LLaMA for Chat ---
def chat_llama(messages):
    """
    Sends a full conversation history to LLaMA.
    Used for the interactive chatbot.
    """
    if not API_KEY:
        return "API Key missing."
        
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7, # Higher temperature for conversation
        "max_tokens": 800
    }

    try:
        response = requests.post(URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        return f"Error: {response.text}"
    except Exception as e:
        return f"Request Failed: {str(e)}"

# --- Authentication Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # FIRESTORE CHANGE: Use get_by_email
        student = Student.get_by_email(email)
        
        if student and student.password_hash and check_password_hash(student.password_hash, password):
            session['user_id'] = student.id
            session['user_name'] = student.name
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password')
            
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        department = request.form.get('department')
        enrollment_year = request.form.get('enrollment_year')
        
        # FIRESTORE CHANGE: Check existence
        if Student.get_by_email(email):
            flash('Email already registered')
            return redirect(url_for('signup'))
        
        hashed_password = generate_password_hash(password)
        
        # FIRESTORE CHANGE: Create new student doc
        new_student = Student.create(
            name=name, 
            email=email, 
            password_hash=hashed_password,
            department=department,
            enrollment_year=int(enrollment_year)
        )
        
        # ID is now the Firestore Document ID
        session['user_id'] = new_student.id
        session['user_name'] = new_student.name
        
        return redirect(url_for('create_resume'))
        
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # FIRESTORE CHANGE: get_by_id
    student = Student.get_by_id(session['user_id'])
    
    if not student:
        session.clear()
        flash('Session expired. Please login again.')
        return redirect(url_for('login'))
    
    # Use dynamic top_roles if available, otherwise empty
    top_roles = student.top_roles or []
    
    # Sort just in case LLaMA returned unsorted
    try:
        if isinstance(top_roles, list):
            top_roles.sort(key=lambda x: x.get('score', 0), reverse=True)
    except Exception as e:
        print(f"Sorting error: {e}") 

    # Prepare data for frontend
    best_role = top_roles[0] if top_roles else None
    
    # FIRESTORE CHANGE: Skills are now in the student object
    verified_skills = student.verified_skills
    
    # --- Daily Bounty Logic ---
    bounty = None
    today = datetime.utcnow().date()
    
    # Check if already solved today (Handle potential string conversion from DB)
    is_solved_today = False
    
    # Convert DB date (which might be a full datetime or string) to date obj for comparison
    last_bounty_val = student.last_bounty_date
    if last_bounty_val:
        if isinstance(last_bounty_val, datetime):
            if last_bounty_val.date() == today:
                is_solved_today = True
        elif isinstance(last_bounty_val, str):
            # If stored as ISO string
            try:
                if datetime.fromisoformat(last_bounty_val).date() == today:
                    is_solved_today = True
            except: pass

    if is_solved_today:
        pass # Already solved
    elif verified_skills:
        # Check session for existing bounty (Must be a list now)
        skill_names = [s['skill_name'] for s in verified_skills if s.get('verified')]
        
        # Check if session has valid list-based bounty
        existing_bounty = session.get('bounty_data')
        if existing_bounty and isinstance(existing_bounty, list) and len(existing_bounty) > 0:
            bounty = existing_bounty
        else:
            # Generate new 5-question bounty
            import random
            if skill_names:
                # Select up to 5 skills (repeat if necessary)
                target_skills = []
                for _ in range(5):
                    target_skills.append(random.choice(skill_names))
                
                skills_str = ", ".join(set(target_skills)) # Unique for prompt context
                print(f"Generating 5 Daily Bounties for: {skills_str}")
                
                bounty_prompt = (
                    f"Create 5 different advanced multiple-choice questions for a developer skilled in: {skills_str}. "
                    "Distribute questions across these topics if possible. "
                    "Each question must have 4 options and one correct answer index (0-3). "
                    "Return strictly a JSON ARRAY of objects. "
                    "Format: [{\"question\": \"...\", \"options\": [\"A\", \"B\", \"C\", \"D\"], \"answer\": 0, \"skill\": \"...\"}, ...]"
                    "No markdown."
                )
                
                bounty_resp = ask_llama("", bounty_prompt)
                
                # Parse
                import json
                import re
                try:
                    clean_bounty = re.sub(r'```json\s*|\s*```', '', bounty_resp).strip()
                    # Find outer brackets
                    s_idx = clean_bounty.find('[')
                    e_idx = clean_bounty.rfind(']')
                    
                    if s_idx != -1 and e_idx != -1:
                        bounty_list = json.loads(clean_bounty[s_idx:e_idx+1])
                        
                        # Validate structure
                        if isinstance(bounty_list, list) and len(bounty_list) > 0:
                            session['bounty_data'] = bounty_list
                            bounty = bounty_list
                        else:
                            print("Bounty Generation: Not a valid list.")
                    else:
                        print(f"Bounty Parse Fail. Response: {bounty_resp}")
                        
                except Exception as be:
                    print(f"Bounty Generation Error: {be}")
    else:
        session.pop('bounty_data', None)

    scores = {
        'roles': top_roles,
        'best_fit': best_role['role'] if best_role else 'N/A',
        'best_score': best_role['score'] if best_role else 0
    }

    return render_template('dashboard.html', student=student, scores=scores, verified_skills=verified_skills, bounty=bounty, is_solved_today=is_solved_today)

@app.route('/solve_bounty', methods=['POST'])
@login_required
def solve_bounty():
    # Legacy form-based submission (Keep if needed, or redirect to game)
    return redirect(url_for('play_game'))

@app.route('/play_game')
@login_required
def play_game():
    student = Student.get_by_id(session['user_id'])
    
    # Ensure generated bounty exists
    bounty = session.get('bounty_data')
    if not bounty:
        flash("No active bounty found. Please refresh dashboard.")
        return redirect(url_for('dashboard'))
        
    return render_template('game.html', bounty=bounty, student=student)

@app.route('/solve_game_bounty', methods=['POST'])
@login_required
def solve_game_bounty():
    total_xp = int(request.form.get('total_xp', 0))
    student = Student.get_by_id(session['user_id'])
    
    # Security: Cap max XP to avoid cheating via direct POST
    if total_xp > 50: total_xp = 50 
    
    updates = {}
    updates['last_bounty_date'] = datetime.utcnow()
    updates['xp'] = student.xp + total_xp
    
    if total_xp > 0:
        flash(f"Game Complete! You earned +{total_xp} XP.")
    else:
        flash("Game Complete. Better aim next time!")
        
    student.update(updates)
    session.pop('bounty_data', None) # Clear bounty
    
    return redirect(url_for('dashboard'))

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    # 1. Check File Presence
    if 'resume' not in request.files:
        flash('No file part')
        return redirect(url_for('dashboard'))
    
    file = request.files['resume']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('dashboard'))
    
    if file:
        filename = secure_filename(file.filename)
        # --- CHANGED: Read file content directly into memory ---
        file_bytes = file.read()

        # 2. OCR and Analysis Processing
        try:
            reader = get_reader()
            # Pass bytes directly to readtext
            results = reader.readtext(file_bytes, detail=0) 
            extracted_text = " ".join(results)
            
            # --- LLaMA Analysis Start ---
            print("\n📊 Calling LLaMA for Top 3 Role Suggestions...")
            
            prompt = (
                "Analyze the resume content below. Identify the top 3 most suitable specific job roles for this candidate. "
                "For each role, provide a suitability score (0-100) based on skills and experience. "
                "Return strictly a JSON array of objects, where each object has 'role' (string) and 'score' (integer). "
                "Example: [{\"role\": \"Backend Developer\", \"score\": 90}, {\"role\": \"Data Analyst\", \"score\": 85}]. "
                "Do NOT include markdown formatting or extra text.\n\n"
                f"Resume Text: {extracted_text[:1000]}..." 
            )
            
            response_text = ask_llama(extracted_text, prompt)
            print(f"DEBUG: LLaMA Raw Response: {response_text}")
            
            top_roles_data = []
            try:
                # cleans markdown code blocks if present
                clean_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
                # find first [ and last ]
                start = clean_text.find('[')
                end = clean_text.rfind(']')
                if start != -1 and end != -1:
                    json_str = clean_text[start:end+1]
                    top_roles_data = json.loads(json_str)
                    print(f"DEBUG: Parsed Roles: {top_roles_data}")
                else:
                    print("DEBUG: Could not find JSON array in response.")
            except Exception as js_e:
                 print(f"DEBUG: JSON Parsing Failed: {js_e}")

            # Fallback if empty
            if not top_roles_data:
                top_roles_data = [
                    {'role': 'SDE (Fallback)', 'score': 0}, 
                    {'role': 'Full Stack (Fallback)', 'score': 0}, 
                    {'role': 'AI/ML (Fallback)', 'score': 0}
                ]

            # --- LLaMA Analysis End ---

            # 3. Store in DB
            student = None
            if 'user_id' in session:
                 student_id = session['user_id']
                 student = Student.get_by_id(student_id)
                 if student:
                     print(f"DEBUG: Found logged-in student [ID: {student_id}]")
            
            if not student:
                flash('Please login to save resume data.')
                return redirect(url_for('login'))

            # FIRESTORE CHANGE: Update Fields via .update()
            updates = {}
            updates['top_roles'] = top_roles_data
            updates['market_analysis'] = None # Force regeneration
            
            # --- ROADMAP GENERATION ---
            try:
                # Determine best role (fallback to software engineer)
                best_role = top_roles_data[0]['role'] if top_roles_data else "Software Engineer"
                print(f"Generating personalized roadmap for: {best_role}")
                
                roadmap_prompt = (
                    f"Create a step-by-step learning roadmap for a student aspiring to be a '{best_role}'. "
                    "Based on the resume content provided earlier, mark steps as 'Completed' if they clearly have the skill. "
                    "Mark ALL remaining steps as 'Focus'. Do not use 'Locked' or 'In Progress'. "
                    "CRITICAL LOGIC RULE: If a higher-level step (e.g., Step 4) is marked 'Completed', ALL previous steps (Step 1, 2, 3) MUST also be marked 'Completed', regardless of explicit mention in the resume. "
                    "Return strictly a JSON list of objects. Each object must have: "
                    "'title' (string), 'description' (short string), 'status' ('Completed', 'Focus'). "
                    "Example: [{\"title\": \"Python\", \"description\": \"Data Structures\", \"status\": \"Completed\"}]. "
                    "Generate exactly 5-6 major steps. Do not use Markdown.\n\n"
                    f"Resume Context: {extracted_text[:1000]}"
                )
                
                roadmap_response = ask_llama("", roadmap_prompt)
                
                # Parse Roadmap JSON
                clean_rmap = re.sub(r'```json\s*|\s*```', '', roadmap_response).strip()
                # Find JSON array
                idx_start = clean_rmap.find('[')
                idx_end = clean_rmap.rfind(']')
                if idx_start != -1 and idx_end != -1:
                    roadmap_data = json.loads(clean_rmap[idx_start:idx_end+1])
                    updates['roadmap'] = roadmap_data
                    print(f"DEBUG: Roadmap saved with {len(roadmap_data)} steps.")
                else:
                    print("DEBUG: Roadmap JSON not found in response.")
            except Exception as r_e:
                print(f"Error generating roadmap: {r_e}")

            # APPLY UPDATES
            student.update(updates)

            # Create Resume Record (Subcollection)
            # Note: We are just saving the OCR text, not the file itself
            student.add_resume(filename=filename, ocr_content=extracted_text)
            
            print("DEBUG: Database update successful.")
            flash('Resume analyzed! Top suitable roles updated.')

        except Exception as e:
            flash(f'Error processing resume: {str(e)}')
            print(f"Error: {e}")
            
        return redirect(url_for('dashboard'))
    
    return redirect(url_for('dashboard'))

@app.route('/upload_certificate', methods=['POST'])
@login_required
def upload_certificate():
    if 'certificate' not in request.files:
        flash('No file part')
        return redirect(url_for('dashboard'))
    file = request.files['certificate']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('dashboard'))
    if file:
        filename = secure_filename(file.filename)
        # --- CHANGED: Read file content directly into memory ---
        file_bytes = file.read()
        
        # 1. OCR (Reuse get_reader)
        try:
            reader = get_reader()
            # Pass bytes directly to readtext
            results = reader.readtext(file_bytes, detail=0) 
            extracted_text = " ".join(results)
            print(f"DEBUG: Certificate OCR Text: {extracted_text[:100]}...")

            # 2. LLaMA Verification
            prompt = (
                "Analyze the following text from a file uploaded as a certificate. "
                "Step 1: Determine if this is a valid certificate of completion, achievement, or skill verification. "
                "Step 2: If valid (1), identify the primary skill or subject (e.g., 'Python', 'Machine Learning'). "
                "Step 3: If invalid (0), return empty skill. "
                "Return JSON ONLY: {\"valid\": 1, \"skill\": \"SkillName\"} or {\"valid\": 0}. "
                "No markdown.\n\n"
                f"Text: {extracted_text[:1000]}"
            )
            
            response_text = ask_llama(extracted_text, prompt)
            print(f"DEBUG: LLaMA Cert Response: {response_text}")
            
            # Parse JSON
            import re
            clean_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
            # Simple fallback regex parser if strict JSON fails
            match_valid = re.search(r'"valid":\s*(\d)', clean_text)
            is_valid = int(match_valid.group(1)) if match_valid else 0
            
            skill_name = "Unknown Skill"
            match_skill = re.search(r'"skill":\s*"([^"]+)"', clean_text)
            if match_skill:
                # Normalize: Strip whitespace and Title Case (e.g., "python " -> "Python")
                skill_name = match_skill.group(1).strip().title()
            
            # 3. DB Update
            if is_valid == 1:
                student = Student.get_by_id(session['user_id'])
                
                # Check for existing skill is handled inside add_skill
                success, msg = student.add_skill(skill_name=skill_name, proficiency=5, verified=True)
                
                if success:
                    flash(f"Skill '{skill_name}' added and verified!")
                else:
                    flash(f"Skill '{skill_name}' was already verified.")
            else:
                flash("Could not verify certificate. Please upload a clear image of a valid certificate.")
                
        except Exception as e:
             print(f"Error processing certificate: {e}")
             flash("Error processing certificate.")

    return redirect(url_for('dashboard'))

@app.route('/leetcode')
@login_required
def leetcode():
    student = Student.get_by_id(session['user_id'])
    return render_template('leetcode.html', student=student)

@app.route('/leetcode_analysis', methods=['POST'])
@login_required
def leetcode_analysis():
    username = request.form.get('username')
    if not username:
        flash('Please enter a username.')
        return redirect(url_for('dashboard'))

    stats = get_leetcode_topic_stats(username)
    if not stats:
        flash(f"User '{username}' not found on LeetCode.")
        if request.form.get('source_page') == 'leetcode':
            return redirect(url_for('leetcode'))
        return redirect(url_for('dashboard'))

    # Store username (FIRESTORE CHANGE)
    student = Student.get_by_id(session['user_id'])
    if student:
        student.update({'leetcode_username': username})

    # Sort and Flatten stats for LLaMA
    all_tags = []
    categories = ['advanced', 'intermediate', 'fundamental']
    for category in categories:
        if category in stats:
            for tag in stats[category]:
                if tag['problemsSolved'] > 0:
                    all_tags.append({
                        "topic": tag['tagName'],
                        "solved": tag['problemsSolved'],
                        "category": category.capitalize()
                    })

    # Sort descending by solved count
    sorted_tags = sorted(all_tags, key=lambda x: x['solved'], reverse=True)

    # Create summary text for prompt
    stats_text = "Topic | Solved | Category\n"
    stats_text += "---|---|---\n"
    for item in sorted_tags:
        stats_text += f"{item['topic']} | {item['solved']} | {item['category']}\n"
    
    prompt = (
    "You are a Senior Technical Interview Coach. Your goal is to analyze LeetCode topic "
    "statistics and output a data-driven JSON study plan.\n\n"
    
    "### INPUT DATA:\n"
    f"{stats_text}\n\n"
    
    "### CONSTRAINTS:\n"
    "- Output must be a SINGLE, valid JSON string.\n"
    "- NO markdown blocks, backticks (```), or extra text. Return ONLY the JSON.\n"
    "- Use HTML tags (<ul>, <li>, <p>) for internal formatting.\n"
    "- For the 'plan' section, you MUST provide exactly 3 problems. Each <li> must include "
    "a clickable LeetCode URL using the format: [https://leetcode.com/problems/](https://leetcode.com/problems/)[problem-slug]/\n\n"
    
    "### REQUIRED JSON STRUCTURE:\n"
    "{\n"
    '  "strengths": "<ul><li>[Metric-based strength]</li><li>[Advanced concept to study]</li></ul>",\n'
    '  "focus": "<p>[Identify a critical weak area based on high submission-to-acceptance ratios or low problem count and explain why it matters for interviews.]</p>",\n'
    '  "plan": "<ul><li>[Problem Name]: <a href=\'URL\'>URL</a></li><li>[Problem Name]: <a href=\'URL\'>URL</a></li><li>[Problem Name]: <a href=\'URL\'>URL</a></li></ul>"\n'
    "}\n\n"
    
    "### ANALYSIS LOGIC:\n"
    "1. Sort topics by completion rate and frequency.\n"
    "2. Identify the 'bottleneck' topic (e.g., high attempts but no success).\n"
    "3. Select 3 highly-rated problems from that bottleneck topic that are common in FAANG interviews.\n\n"
    "Return only the raw JSON."
)
    
    response_text = ask_llama("", prompt)
    
    # Parse JSON
    import json
    import re
    cleaned_json = re.sub(r'```json\s*|\s*```', '', response_text).strip()
    
    try:
        suggestion_data = json.loads(cleaned_json)
        # Ensure keys exist
        if not isinstance(suggestion_data, dict):
            raise ValueError("Not a dictionary")
            
        session['leetcode_suggestion_json'] = suggestion_data
        session['leetcode_suggestion'] = None # Clear legacy non-json if any
    except Exception as e:
        print(f"JSON Parse Error for LeetCode: {e}")
        # Fallback to string if JSON fails
        session['leetcode_suggestion'] = response_text
        session['leetcode_suggestion_json'] = None
    
    session['leetcode_stats'] = stats
    
    flash("LeetCode profile analyzed successfully!")
    
    if request.form.get('source_page') == 'leetcode':
        return redirect(url_for('leetcode'))
        
    return redirect(url_for('dashboard'))

# --- Resume Builder Routes ---
@app.route('/resume_builder')
# @login_required 
def resume_builder():
    resume_data = session.get('resume_data', {})

    # Fetch from Database if logged in
    if 'user_id' in session:
        student = Student.get_by_id(session['user_id'])
        if student and student.resume_profile:
            resume_data = student.resume_profile

    return render_template('resume_builder.html', resume_data=resume_data)

@app.route('/create-resume')
# @login_required
def create_resume():
    """Separate route for the specific 'Create Your Resume' page."""
    resume_data = session.get('resume_data', {})

    # Fetch from Database if logged in
    if 'user_id' in session:
        student = Student.get_by_id(session['user_id'])
        if student and student.resume_profile:
            # Load saved profile from DB
            resume_data = student.resume_profile

    return render_template('create_resume.html', resume_data=resume_data, hide_sidebar=True)

@app.route('/generate_resume', methods=['POST'])
def generate_resume():
    full_name = request.form.get('full_name')
    institute_name = request.form.get('institute_name')
    degree = request.form.get('degree')
    github_id = request.form.get('github_id')
    leetcode_id = request.form.get('leetcode_id')
    skills_str = request.form.get('skills')
    specialization = request.form.get('specialization')
    summary = request.form.get('summary')
    
    skills_list = [s.strip() for s in skills_str.split(',') if s.strip()]
    
    # Process Projects
    project_titles = request.form.getlist('project_title')
    project_descs = request.form.getlist('project_desc')
    project_links = request.form.getlist('project_link')
    
    projects = []
    if project_titles:
        for i in range(len(project_titles)):
            if project_titles[i].strip():
                projects.append({
                    'title': project_titles[i],
                    'desc': project_descs[i] if i < len(project_descs) else "",
                    'link': project_links[i] if i < len(project_links) else ""
                })

    # Process Achievements
    achievements_str = request.form.get('achievements')
    achievements_list = []
    if achievements_str:
        # Split by new line
        achievements_list = [a.strip() for a in achievements_str.split('\n') if a.strip()]

    # Construct Links
    github_url = f"https://github.com/{github_id}" if github_id else "#"
    leetcode_url = f"https://leetcode.com/{leetcode_id}" if leetcode_id else "#"
    
    # Get Email if logged in
    email = session.get('user_email', '') 
    if 'user_id' in session:
        student = Student.get_by_id(session['user_id'])
        if student:
            email = student.email

    data = {
        'full_name': full_name,
        'institute_name': institute_name,
        'degree': degree,
        'github_id': github_id,
        'leetcode_id': leetcode_id,
        'github_url': github_url,
        'leetcode_url': leetcode_url,
        'skills_list': skills_list,
        'specialization': specialization,
        'summary': summary, # Add summary
        'projects': projects,
        'achievements': achievements_list,
        'email': email
    }

    # --- SESSION STORAGE FOR EDITING ---
    # --- SESSION STORAGE FOR EDITING ---
    session['resume_data'] = {
        'full_name': full_name,
        'institute_name': institute_name,
        'degree': degree,
        'github_id': github_id,
        'leetcode_id': leetcode_id,
        'skills': skills_str,
        'specialization': specialization,
        'summary': summary, # Add summary
        'projects': projects,
        'achievements': achievements_str
    }

    # --- SAVE TO DB (PERSISTENCE) ---
    if 'user_id' in session:
        student = Student.get_by_id(session['user_id'])
        if student:
            # Save the raw form data for reloading later
            student.update({'resume_profile': session['resume_data']})
            print("✅ Resume Profile Details Saved to Firebase.")

    # --- ONE-TIME CAREER PATHWAY DERIVATION (FIREBASE) ---
    try:
        if 'user_id' in session:
            student = Student.get_by_id(session['user_id'])
            
            # RUN ONCE RULE: Only if not already derived
            if student and not student.data.get('profileDerived'):
                print("🧠 Deriving Career Pathway for the first time...")
                
                # Construct strict prompt for LLaMA
                pathway_prompt = (
                    "You are a strict Career Pathway Engine. Analyze this student profile and derive metadata.\n"
                    "RULES:\n"
                    "1. Anchor to Course/Degree/Branch (60% weight).\n"
                    "   - If 'Specialization' field is provided, it modifies/refines the Course anchor.\n"
                    "   - software: cse, it, ai, or software specialization\n"
                    "   - hardware: ece, eee, biomedical, or hardware specialization\n"
                    "   - core: mech, civil\n"
                    "2. Resume Skills are secondary (40% weight).\n"
                    "3. If Specialization strongly conflicts with Course (e.g. Civil + Data Science), treat Course as Primary foundation and Specialization as dominant Secondary Track.\n"
                    "4. Output strict JSON with fields: 'course' (inferred), 'primaryDomain', 'secondaryDomains' (list), 'pathwayType', 'pathwayWeights'.\n"
                    "\n"
                    f"Degree: {degree}\n"
                    f"Institute: {institute_name}\n"
                    f"Specialization: {specialization}\n"
                    f"Skills: {', '.join(skills_list)}\n"
                    f"Projects: {str(projects)}\n"
                    "\n"
                    "Return JSON only."
                )
                
                # Call AI
                ai_resp = ask_llama("", pathway_prompt)      
                
                # Parse and Save
                import json
                import re
                try:
                    clean_json = re.sub(r'```json\s*|\s*```', '', ai_resp).strip()
                    pathway_data = json.loads(clean_json)
                    
                    # Add system fields
                    pathway_data['profileDerived'] = True
                    pathway_data['derivedAt'] = datetime.utcnow().isoformat()
                    pathway_data['derivationTrigger'] = "profile_creation"
                    
                    # Update Firestore
                    student.update(pathway_data)
                    print("✅ Career Pathway Saved to Firebase.")
                    
                except Exception as parse_err:
                    print(f"❌ Pathway Derivation Failed: {parse_err}")

        # --- ROADMAP GENERATION (ALWAYS UPDATE) ---
        try:
            print("🗺️ Generating Roadmap...")
            roadmap_prompt = (
                "Create a 5-step detailed learning roadmap for this student.\n"
                "JSON format list of objects: [{'title', 'description', 'status'}].\n"
                "Status options: 'Completed' (if they have skills), 'Focus' (next step), 'Locked' (future).\n"
                "\n"
                f"Course: {degree}\n"
                f"Specialization: {specialization}\n"
                f"Current Skills: {', '.join(skills_list)}\n"
                f"Goal Role: {specialization if specialization else 'Software Engineer'}\n"
                "\n"
                "Return JSON only."
            )
            
            roadmap_resp = ask_llama("", roadmap_prompt)
            import re
            import json
            clean_roadmap_json = re.sub(r'```json\s*|\s*```', '', roadmap_resp).strip()
            
            # Handle potential wrapping in quotes or dict
            try:
                roadmap_data = json.loads(clean_roadmap_json)
                
                # Check if wrapped in a key
                if isinstance(roadmap_data, dict):
                    # Try to find a list value or use the first key
                    for k, v in roadmap_data.items():
                        if isinstance(v, list):
                            roadmap_data = v
                            break
                
                if isinstance(roadmap_data, list):
                    student.update({'roadmap': roadmap_data})
                    print("✅ Roadmap Updated.")
                else:
                    print("⚠️ Roadmap format invalid (not a list).")
                    
            except json.JSONDecodeError:
                print("⚠️ Roadmap JSON Decode Error.")

        except Exception as r_err:
             print(f"❌ Roadmap Generation Error: {r_err}")

    except Exception as e:
        print(f"⚠️ Pathway Logic Error: {e}")

    return render_template('generated_resume.html', data=data)

# --- Pages ---

@app.route('/career')
def career():
    return render_template('career.html')

@app.route('/roadmap')
@login_required
def roadmap():
    student = Student.get_by_id(session['user_id'])
    return render_template('roadmap.html', student=student)

@app.route('/market')
@login_required
def market():
    student = Student.get_by_id(session['user_id'])
    
    if not student:
        flash('User not found.')
        return redirect(url_for('login'))
    
    # Check if analysis exists, if so render it to save API calls
    if student.market_analysis:
        return render_template('market.html', student=student, analysis=student.market_analysis)

    # Generate Analysis
    # FIRESTORE CHANGE: Get latest resume from subcollection
    resume = student.get_latest_resume()
    resume_text = resume['ocr_content'] if resume else "Student with basic Computer Science skills."

    print("Generating Market Analysis...")
    prompt = (
        "Analyze the current tech job market and this candidate's resume.\n"
        "1. Identify 3 High Paying 'Booming' Roles. For each, provide Avg Package (e.g. '$150k') and 3 specific skills they lack.\n"
        "2. Identify 2 Target Roles suitable for them. Estimate progress (0-100%) and provide strategic advice + 2 action items.\n"
        "Return strict JSON with keys: 'market_roles' and 'optimization'.\n"
        "Example:\n"
        "{\n"
        "  \"market_roles\": [{\"role\": \"AI Architect\", \"package\": \"$160k\", \"skills\": [\"LLMs\", \"System Design\"]}],\n"
        "  \"optimization\": [{\"role\": \"Backend Dev\", \"progress\": 60, \"advice\": \"Good Python, weak DB.\", \"actions\": [\"Learn Redis\", \"Build API\"]}]\n"
        "}\n\n"
        f"Resume Content: {resume_text[:2000]}"
    )

    response = ask_llama("", prompt)
    
    # Parse JSON
    import json
    import re
    clean_json = re.sub(r'```json\s*|\s*```', '', response).strip()
    try:
        # Find start/end brackets to be safe
        s = clean_json.find('{')
        e = clean_json.rfind('}')
        if s != -1 and e != -1:
            analysis_data = json.loads(clean_json[s:e+1])
            # FIRESTORE CHANGE: Update logic
            student.update({'market_analysis': analysis_data})
            return render_template('market.html', student=student, analysis=analysis_data)
    except Exception as e:
        print(f"Market Analysis Error: {e}")
    
    # Fallback empty
    return render_template('market.html', student=student, analysis=None)

@app.route('/mentors')
def mentors():
    return render_template('mentors.html')

@app.route('/stories')
def stories():
    return render_template('stories.html')

@app.route('/institution')
def institution():
    return render_template('institution.html')

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/profile') 
def profile():
    return render_template('profile.html')

@app.route('/skills') 
def skills():
    return render_template('skills.html')

# Initialize DB
# REMOVED: db.create_all() (Not needed for Firestore)

# -------------------------------
# ANALYSIS CONFIG (GITHUB/LEETCODE)
# -------------------------------
GITHUB_API_BASE = "https://api.github.com"

COURSE_DOMAIN_MAP = {
    "cse": "software",
    "it": "software",
    "ai": "software",
    "aiml": "software",
    "ece": "hardware",
    "eee": "hardware",
    "biomedical": "hardware",
    "mechanical": "core",
    "mech": "core",
    "civil": "core"
}

PROJECT_DOMAIN_RULES = {
    "software": ["python", "java", "javascript", "react", "node", "flask", "django", "c++", "golang"],
    "data": ["machine learning", "ml", "pandas", "numpy", "pytorch", "tensorflow", "data analysis"],
    "hardware": ["arduino", "esp32", "embedded", "iot", "circuit", "pcb", "verilog", "fpga"],
    "web": ["html", "css", "react", "frontend", "backend", "full stack"],
    "core": ["matlab", "ansys", "solidworks", "cad", "simulation", "thermodynamics", "mechanics"]
}

REQUIRED_PROJECTS = {
    "software": ["backend api", "database integration", "full stack app"],
    "hardware": ["embedded system", "sensor interfacing", "iot dashboard"],
    "data": ["ml model deployment", "exploratory data analysis"],
    "core": ["simulation project", "design prototype"],
    "web": ["responsive website", "full stack app"]
}

# -------------------------------
# GITHUB HELPER FUNCTIONS
# -------------------------------
def get_github_repos(username):
    try:
        url = f"{GITHUB_API_BASE}/users/{username}/repos"
        res = requests.get(url, timeout=5)
        if res.status_code != 200:
            return []
        repos = []
        for repo in res.json():
            repos.append({
                "name": repo["name"],
                "description": repo["description"] or "",
                "languages_url": repo["languages_url"],
                "html_url": repo["html_url"],
                "commits": repo["size"]
            })
        return repos
    except:
        return []

def get_repo_languages(languages_url):
    try:
        res = requests.get(languages_url, timeout=5)
        if res.status_code != 200:
            return []
        return list(res.json().keys())
    except:
        return []

def classify_project(description, languages):
    text = (description + " " + " ".join(languages)).lower()
    scores = {d: 0 for d in PROJECT_DOMAIN_RULES}
    for domain, keywords in PROJECT_DOMAIN_RULES.items():
        for kw in keywords:
            if kw in text:
                scores[domain] += 1
    # If all scores 0, return unknown or default
    if max(scores.values()) == 0:
        return "unknown"
    return max(scores, key=scores.get)

def analyze_github(username):
    repos = get_github_repos(username)
    analysis = []
    domains_found = []
    for repo in repos:
        languages = get_repo_languages(repo["languages_url"])
        domain = classify_project(repo["description"], languages)
        if domain != 'unknown':
            domains_found.append(domain)
        analysis.append({
            "repo_name": repo["name"],
            "domain": domain,
            "languages": languages,
            "url": repo["html_url"]
        })
    return analysis, list(set(domains_found))

def detect_primary_domain(course):
    course = course.lower() if course else ""
    for key in COURSE_DOMAIN_MAP:
        if key in course:
            return COURSE_DOMAIN_MAP[key]
    return "software"

# -------------------------------
# ANALYSIS ROUTES
# -------------------------------
@app.route("/analyze", methods=["POST"])
@login_required 
def analyze_profile():
    data = request.json
    github_username = data.get("github_username")
    
    if not github_username:
        return jsonify({"error": "github_username is required"}), 400

    student = Student.get_by_id(session['user_id'])
    
    # 1. Determine Target Domain
    primary_domain = student.data.get('primaryDomain')
    if not primary_domain:
         course_name = data.get("course") or student.data.get('degree', "")
         primary_domain = detect_primary_domain(course_name)
    
    # 2. Fetch raw data from GitHub
    repos = fetch_github_data(github_username)
    if not repos:
        return jsonify({
            "error": "No data found",
            "projects": [], 
            "missing_projects": ["Could not fetch GitHub data"], 
            "github_domains_detected": [],
            "ai_suggestions": [],
            "career_readiness": "Unknown",
            "primary_domain": primary_domain
        })

    # 3. RULE-BASED Analysis
    analysis_result = analyze_github_rule_based(repos, primary_domain)
    
    repo_map = analysis_result['repo_map']
    
    # 4. Construct Frontend Response
    formatted_projects = []
    for r in repos:
        repo_info = repo_map.get(r['name'], {})
        formatted_projects.append({
            "repo_name": r["name"],
            "domain": repo_info.get('domain', 'General'), 
            "languages": r["languages"],
            "url": r["url"]
        })

    # 5. Generate Suggestions (Static based on missing)
    suggestions = []
    for gap in analysis_result['missing_projects']:
        suggestions.append({
            "title": f"Build a {gap}",
            "description": f"You are missing a critical {gap} project. This is essential for {primary_domain} roles.",
            "tech": "See standard tech stack"
        })
    
    # If no gaps, suggest optional
    if not suggestions:
        for opt in analysis_result['optional_suggestions'][:3]:
             suggestions.append({
                "title": f"Explore {opt}",
                "description": f"Advanced project to boost your profile.",
                "tech": "Advanced Tech"
            })

    response = {
        "github_user": github_username,
        "primary_domain": primary_domain,
        "secondary_domains": [],
        "repo_count": len(repos),
        "projects": formatted_projects, 
        "github_domains_detected": analysis_result['detected_domains'],
        "missing_projects": analysis_result['missing_projects'],
        "ai_suggestions": suggestions, 
        "career_readiness": analysis_result['career_readiness']
    }
    
    student.update({'last_github_analysis': response})

    return jsonify(response)

@app.route('/analysis')
@login_required
def analysis():
    student = Student.get_by_id(session['user_id'])
    return render_template('analysis.html', student=student)

# -------------------------------
# AI CHATBOT ROUTE
# -------------------------------
@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    data = request.json
    user_message = data.get('message')
    history = data.get('history', []) # Expects list of {role, content}
    
    if not user_message:
        return jsonify({'error': 'Message required'}), 400
        
    student = Student.get_by_id(session['user_id'])
    
    # 1. Construct Context System Prompt
    # Extract relevant student info
    
    # Skills
    skills_list = [s['skill_name'] for s in student.verified_skills if s.get('verified')]
    skills_str = ", ".join(skills_list) if skills_list else "None verified yet"
    
    # Domain
    domain = student.data.get('primaryDomain', 'General Engineering')
    secondary = ", ".join(student.data.get('secondaryDomains', []))
    
    # Roadmap context (optional, simple summary)
    current_roadmap = "Not generated"
    if student.roadmap:
        # Just take the first incomplete step
        for step in student.roadmap:
            if step['status'] == 'Focus':
                current_roadmap = f"Focusing on: {step['title']} - {step['description']}"
                break
    
    system_prompt = (
        f"You are EduBot, an AI mentor for {student.name}.\n"
        f"Student Profile:\n"
        f"- Course: {student.department} (Year {student.enrollment_year})\n"
        f"- Target Domain: {domain} {f'({secondary})' if secondary else ''}\n"
        f"- Verified Skills: {skills_str}\n"
        f"- Current Goal: {current_roadmap}\n\n"
        "Instructions:\n"
        "1. Answer questions based on their specific skills and domain.\n"
        "2. If they ask about learning, refer to their gaps or next roadmap steps.\n"
        "3. Be encouraging, professional, and concise.\n"
        "4. Do NOT explicitly mention 'I checked your database record'. Just know the facts.\n"
    )
    
    # 2. Build Message Chain
    # We shouldn't trust client history blindly for system prompt, so we prepend system prompt here.
    # History from client should be just user/assistant exchange.
    
    # Limit history to last 6 messages to save tokens
    trimmed_history = history[-6:] 
    
    messages = [{"role": "system", "content": system_prompt}] + trimmed_history + [{"role": "user", "content": user_message}]
    
    # 3. Call AI
    ai_response = chat_llama(messages)
    
    return jsonify({
        "reply": ai_response
    })

@app.route('/update_exam_scores', methods=['POST'])
@login_required
def update_exam_scores():
    student = Student.get_by_id(session['user_id'])
    if not student:
        return redirect(url_for('login'))
        
    scores = {
        'GRE': request.form.get('gre'),
        'GATE': request.form.get('gate'),
        'IELTS': request.form.get('ielts'),
        'TOEFL': request.form.get('toefl')
    }
    
    # Filter out empty strings
    clean_scores = {k: v for k, v in scores.items() if v and v.strip()}
    
    student.update({'exam_scores': clean_scores})
    
    # Clear cache since input changed
    # We might want to clear all higher_studies caches, or just the current context.
    # For simplicity, let's clear the specific cache we are about to view or broadly clear.
    # But since cache keys are dynamic (by country), we rely on the user to re-select or we force a refresh.
    # Ideally, we should invalidate caches. For now, we'll implement a simple suffix to the cache key or just clear the main one.
    
    # Let's force a refresh by clearing the last generic key used if possible, or just rely on the user flow.
    # A better way: The prompt depends on these scores. If we change scores, we want new recs.
    # Let's purge all higher_studies keys from student data? That's expensive.
    # Instead, we will append the scores hash to the cache key in the GET route, 
    # OR simpler: just update the 'higher_studies_advanced' cache for the current/global country if we can.
    
    flash("Exam scores updated! Analyzer will now use these for better probability estimates.")
    return redirect(url_for('higher_studies', country=request.form.get('current_country', 'Global')))

@app.route('/higher_studies')
@login_required
def higher_studies():
    student = Student.get_by_id(session['user_id'])
    if not student:
        flash("Please log in.")
        return redirect(url_for('login'))
        
    # Get selected country from query param, default to Global
    selected_country = request.args.get('country', 'Global')

    # Get Top Role
    top_roles = student.top_roles or []
    target_role = None
    if top_roles:
         try:
             sorted_roles = sorted(top_roles, key=lambda x: x.get('score', 0), reverse=True)
             target_role = sorted_roles[0]['role']
         except:
             target_role = top_roles[0].get('role', "Software Engineer")
    
    if not target_role:
        return render_template('higher_studies.html', universities=[], target_role=None, selected_country=selected_country, exam_scores={})

    # Retrieve Resume Content
    resume = student.get_latest_resume()
    resume_text = resume['ocr_content'] if resume else ""
    
    # Retrieve Exam Scores
    exam_scores = student.data.get('exam_scores', {})
    
    # Create a cache key that includes a hash of the exam scores to invalidate on change
    # or simply suffix if they exist. simpler: cache_key depends on country. 
    # note: if user updates scores, we essentially want to ignore the old cache. 
    # We can detect if the cached data 'exam_readiness' matches the current input?
    # No, simpler to just accept that if we want "Fresh" analysis, the user can change country or wait.
    # BUT, to make it responsive, let's include 'v2' or something if exams exist.
    
    # Let's try to load cache.
    cache_key = f'higher_studies_advanced_{selected_country}'
    existing_recs = student.data.get(cache_key)
    
    # If we have recs... check if they were generated with the CURRENT exam scores.
    # We can store 'used_scores' in the saved data to compare.
    need_refresh = True
    if existing_recs and isinstance(existing_recs, dict) and existing_recs.get('role') == target_role:
        saved_scores = existing_recs.get('used_scores', {})
        # Compare saved_scores with current exam_scores
        # Treat None and empty string as same
        current_clean = {k: v for k, v in exam_scores.items() if v}
        saved_clean = {k: v for k, v in saved_scores.items() if v}
        
        if current_clean == saved_clean:
            need_refresh = False
            
    if not need_refresh:
         return render_template('higher_studies.html', 
                                universities=existing_recs.get('universities', []), 
                                target_role=target_role, 
                                exam_readiness=existing_recs.get('exam_readiness'), 
                                selected_country=selected_country,
                                exam_scores=exam_scores)

    # Generate Recommendations via AI
    print(f"Generating Advanced University Recommendations for: {target_role} in {selected_country} (Scores: {exam_scores})")
    
    country_focus = f"in {selected_country}" if selected_country != 'Global' else "Worldwide (Top Global)"
    
    # Inject Scores into Prompt
    scores_context = "No official exam scores provided. Estimate based on resume only."
    if exam_scores:
        scores_list = ", ".join([f"{k}: {v}" for k, v in exam_scores.items() if v])
        if scores_list:
            scores_context = f"STUDENT OFFICIAL SCORES: {scores_list}. USE THESE for probability calculation."
    
    prompt = (
        f"Act as an expert Overseas & Domestic Education Counselor. Analyze this student's profile (resume below) and the target role '{target_role}'.\n"
        f"{scores_context}\n"
        "Task 1: Evaluate Exam Readiness. If official scores are provided above, use them to grade readiness (e.g. 'GRE 320 -> High'). If not, estimate based on resume.\n"
        f"Task 2: Recommend 5 top universities ({country_focus}, best fit for profile) for this role.\n"
        "\n"
        "For each university, provide:\n"
        "1. Name\n"
        "2. Location (City, Country)\n"
        "3. Global Ranking (Use ARWU, THE, or QS format, e.g. 'QS #45')\n"
        "4. Tuition Fee (Yearly in local currency + approx INR)\n"
        "5. Exams Required (e.g. 'GRE: 315+, IELTS: 7.0')\n"
        "6. Admit Probability (Estimate % based on scores/resume vs uni standards)\n"
        "7. Scholarship Probability (Estimate %)\n"
        "8. Suitability Score (0-100)\n"
        "9. Reason (Short, mentioning if their score is good enough)\n"
        "\n"
        "Return strictly a JSON Object with two keys:\n"
        "1. 'exam_readiness': { 'GRE': '...', 'GATE': '...', 'IELTS': '...', 'TOEFL': '...' }\n"
        "2. 'universities': [ Array of objects with keys: 'name', 'location', 'ranking', 'fee', 'exams', 'admit_prob', 'scholarship_prob', 'suitability', 'reason' ]\n"
        "No Markdown."
        f"\n\nResume Context: {resume_text[:2500]}"
    )
    
    response_text = ask_llama("", prompt)
    
    data = {'universities': [], 'exam_readiness': {}}
    import json
    import re
    
    try:
        clean_json = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        # Parsing handling for potentially wrapped or raw json
        s = clean_json.find('{')
        e = clean_json.rfind('}')
        if s != -1 and e != -1:
            data = json.loads(clean_json[s:e+1])
            
            # Save to Student DB
            save_data = {
                'role': target_role,
                'universities': data.get('universities', []),
                'exam_readiness': data.get('exam_readiness', {}),
                'used_scores': exam_scores, # Store which scores were used
                'updated_at': datetime.utcnow().isoformat()
            }
            student.update({cache_key: save_data})
            
    except Exception as e:
        print(f"University Recs Error: {e}")

    return render_template('higher_studies.html', universities=data.get('universities', []), target_role=target_role, exam_readiness=data.get('exam_readiness', {}), selected_country=selected_country, exam_scores=exam_scores)

# --- Internship Section ---

INTERNSHIP_DATA = [
    {
        "title": "Python Developer Intern",
        "company": "Infrabyte Consulting",
        "link": "https://www.infrabyteconsulting.com/jobs",
        "type": "Internship",
        "location": "Remote",
        "stipend": "₹15,700 / month",
        "duration": "2–3 months",
        "tags": ["Python", "Remote", "Backend", "Flask"],
        "description": "Assist in application development and automation projects. Work with Flask/Django, SQL, Git."
    },
    {
        "title": "Machine Learning Intern",
        "company": "Innovexis",
        "link": "https://careerspage.io/innovexis/machine-learning-intern-in121?src=37",
        "type": "Internship",
        "location": "Remote",
        "stipend": "₹20,000 - ₹25,000 / month",
        "duration": "N/A",
        "tags": ["Machine Learning", "AI", "Remote", "Data Science"],
        "description": "Build, train, and evaluate ML models. Work in Education Management industry."
    },
    {
        "title": "Web Developer Intern",
        "company": "Inficore Soft",
        "link": "https://inficoresoft.com/jobs",
        "type": "Internship",
        "location": "Remote",
        "stipend": "₹14,500 / month",
        "duration": "1-3 months",
        "tags": ["Web Development", "Remote", "Frontend"],
        "description": "Full-time internship for web development."
    },
    {
        "title": "Cyber Security Intern",
        "company": "Various",
        "link": "https://docs.google.com/forms/d/1pk9ZN-620VbRliDY7YeVKs4JA149rmVnM96ki354Z2I/viewform?edit_requested=true",
        "type": "Internship",
        "location": "Remote",
        "stipend": "Unpaid",
        "duration": "Flexible",
        "tags": ["Cyber Security", "Remote"],
        "description": "Virtual/Remote internship opportunity."
    },
    {
        "title": "GenAI & Agent Research Scientist",
        "company": "Sententia Research",
        "link": "https://www.sourcingxpress.com/jobs/TEz4O1B70rZpoY?utm_source=linkedin&utm_medium=job_board&utm_campaign=careers_2026",
        "type": "Job",
        "location": "Unknown",
        "stipend": "₹ 1-2 Lacs PA",
        "duration": "Full Time",
        "tags": ["GenAI", "Data Science", "Startup", "AI"],
        "description": "Developing intelligent systems using AI agents and generative AI."
    },
    {
        "title": "Embedded Controller Designer",
        "company": "Leading Automotive Client",
        "link": "https://www.linkedin.com/jobs/view/4367206528/",
        "type": "Job",
        "location": "India",
        "stipend": "Competitive",
        "duration": "Full Time",
        "tags": ["Embedded Systems", "Electronics", "Automotive"],
        "description": "Hardware design for controller and power PCBs, motor control systems."
    },
    {
        "title": "IOT Product Development Engineer",
        "company": "BONbLOC TECHNOLOGIES",
        "link": "https://jobs.best-jobs-online.com/c/job/380c7973696605bc379b0999e3b27109?q=iot&l=chennai%2C+state+of+tamil+nadu&jlc=k8J83yAJMG5gM062b0KJMBVWXOvm&jt=IOT+Product+Development+Engineer&c=BONbLOC",
        "type": "Job",
        "location": "Chennai",
        "stipend": "Competitive",
        "duration": "Full Time",
        "tags": ["IoT", "Python", "Embedded", "Senior"],
        "description": "End-to-end ownership of IoT products—from prototyping to deployment."
    },
    {
        "title": "VLSI Design Internship",
        "company": "Maven Silicon",
        "link": "https://www.maven-silicon.com/vlsi-design-internship/",
        "type": "Internship",
        "location": "Bengaluru",
        "stipend": "Training",
        "duration": "N/A",
        "tags": ["VLSI", "Electronics", "Chip Design"],
        "description": "For Pre-final and final year electronics/electrical engineering students."
    },
    {
        "title": "Robotics Trainer/Intern",
        "company": "EdTech Client",
        "link": "https://www.linkedin.com/jobs/view/4356454963/",
        "type": "Job",
        "location": "Remote",
        "stipend": "Salary",
        "duration": "Full Time (Night Shift)",
        "tags": ["Robotics", "Python", "Remote", "Teaching"],
        "description": "Robotics, Python, PCB Design. Delivering sessions to students."
    },
    {
        "title": "Electrical/Electronics Eng. Intern",
        "company": "Speedways Electric",
        "link": "https://secure.indeed.com/auth?hl=en_IN&co=IN&continue=https%3A%2F%2Fin.indeed.com%2Fthirdpartysignin%3Fjk%3Dba4da650bf1336c1",
        "type": "Internship",
        "location": "Jalandhar",
        "stipend": "Unpaid/Stipend",
        "duration": "Internship",
        "tags": ["Electrical", "Electronics", "EV", "Internship"],
        "description": "Assist in development and testing of electrical systems for electric vehicles."
    },
    {
        "title": "PCB Design Engineer",
        "company": "Robotics Firm",
        "link": "https://www.linkedin.com/jobs/view/4355935191/",
        "type": "Internship/Job",
        "location": "Pune",
        "stipend": "Salary",
        "duration": "Full Time",
        "tags": ["PCB Design", "Robotics", "IoT"],
        "description": "Design and refine multi-layer PCB layouts, create footprints."
    },
    {
        "title": "Senior Mechatronics Engineer",
        "company": "Industrial Automation Firm",
        "link": "https://www.linkedin.com/jobs/view/4367555617/",
        "type": "Job",
        "location": "Bengaluru",
        "stipend": "₹12-18 LPA",
        "duration": "Full Time",
        "tags": ["Mechatronics", "Automation", "Senior"],
        "description": "Mechanical + Electrical Design. Industrial Automation / Machine Vision."
    },
    {
        "title": "Java Backend Developer",
        "company": "Tech Firm",
        "link": "https://www.linkedin.com/jobs/search/?currentJobId=4369508760",
        "type": "Job",
        "location": "Unknown",
        "stipend": "₹14-18 LPA",
        "duration": "Full Time",
        "tags": ["Java", "Backend", "Spring Boot"],
        "description": "Design and develop server-side applications using Java."
    },
    {
        "title": "SQL Developer Intern",
        "company": "Infrabyte Consulting",
        "link": "https://www.infrabyteconsulting.com/jobs",
        "type": "Internship",
        "location": "Remote",
        "stipend": "₹15,700 / month",
        "duration": "2–3 months",
        "tags": ["SQL", "Database", "Remote"],
        "description": "Support database management and query optimization."
    }
]

def seed_internships():
    """Seeds the internships collection if empty."""
    try:
        internships_ref = db.collection('internships')
        # Check if already seeded (limit 1 to save reads)
        docs = list(internships_ref.limit(1).stream())
        if not docs:
            print("Seeding Internships...")
            batch = db.batch()
            for item in INTERNSHIP_DATA:
                doc_ref = internships_ref.document()
                batch.set(doc_ref, item)
            batch.commit()
            print("Internships Seeded Successfully.")
    except Exception as e:
        print(f"Error seeding internships: {e}")

@app.route('/internships')
def internships():
    seed_internships() # Ensure data exists
    
    try:
        internships_ref = db.collection('internships')
        docs = internships_ref.stream()
        
        all_internships = []
        all_tags = set()
        
        for doc in docs:
            data = doc.to_dict()
            all_internships.append(data)
            if 'tags' in data:
                for tag in data['tags']:
                    all_tags.add(tag)
        
        # Sort by title for consistency
        all_internships.sort(key=lambda x: x.get('title', ''))
        
        sorted_tags = sorted(list(all_tags))
        
        return render_template('internships.html', internships=all_internships, tags=sorted_tags)
    except Exception as e:
        print(f"Error fetching internships: {e}")
        return render_template('internships.html', internships=[], tags=[])


@app.route('/quiz/<role_name>')
@login_required
def take_quiz(role_name):
    # Prompt AI to generate questions
    prompt = (
        f"Create a technical skill assessment quiz for the role: '{role_name}'.\n"
        "Generate 10 Multiple Choice Questions (MCQs) testing key skills required for this role.\n"
        "Format strictly as a JSON object with this structure:\n"
        "{\n"
        "  \"questions\": [\n"
        "    {\n"
        "      \"id\": 1,\n"
        "      \"text\": \"Question text here?\",\n"
        "      \"options\": [\"Option A\", \"Option B\", \"Option C\", \"Option D\"],\n"
        "      \"correct_index\": 0\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "No Markdown. Return only valid JSON. Do not use trailing commas. Do NOT use 'All of the above' or 'None of the above' options."
    )
    
    response_text = ask_llama("", prompt)
    
    questions = []
    try:
        clean_json = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        data = json.loads(clean_json)
        questions = data.get('questions', [])
    except Exception as e:
        print(f"Quiz Gen Error: {e}")
        flash("Could not generate quiz. Please try again.")
        return redirect(url_for('market'))
        
    return render_template('quiz.html', role_name=role_name, questions=questions)

@app.route('/quiz/submit', methods=['POST'])
@login_required
def submit_quiz():
    role_name = request.form.get('role_name')
    # Reconstruct user answers and correct answers
    # We need the original questions to evaluate properly or typically we store them in session.
    # For a robust solution without DB persistence for quizzes, we can embed the full question data 
    # (including correct answer) in hidden fields or rely on the AI to grade "User selected X vs Correct Y".
    # BUT, passing correct answers to the client is insecure (though okay for a learning app prototype).
    # Better approach: The form submits question text + selected option text.
    
    quiz_data_str = request.form.get('quiz_data_json') # Passed as hidden
    if not quiz_data_str:
        return "Error: Missing quiz data", 400
        
    import json
    questions = json.loads(quiz_data_str)
    
    user_results = []
    score = 0
    total = len(questions)
    
    transcript = f"Quiz Assessment for {role_name}:\n\n"
    
    for q in questions:
        qid = str(q['id'])
        selected_option = request.form.get(f"q_{qid}")
        correct_option = q['options'][q['correct_index']]
        
        is_correct = (selected_option == correct_option)
        if is_correct: score += 1
        
        user_results.append({
            'question': q['text'],
            'selected': selected_option,
            'correct': correct_option,
            'is_correct': is_correct
        })
        
        transcript += f"Q: {q['text']}\nUser Answer: {selected_option}\nCorrect Answer: {correct_option}\nResult: {'Correct' if is_correct else 'Incorrect'}\n\n"

    # AI Analysis
    analysis_prompt = (
        f"Analyze this quiz performance for the role '{role_name}'.\n"
        f"Score: {score}/{total}\n"
        f"Transcript:\n{transcript}\n"
        "Task:\n"
        "1. Provide an 'Effectiveness Score' (0-100) and a short verdict (e.g. 'Ready for Junior Role').\n"
        "2. Identify specific 'Weak Areas' based on incorrect answers.\n"
        "3. Recommend 3 concrete 'Focus Areas' to study.\n"
        "\n"
        "Return JSON:\n"
        "{\n"
        "  \"effectiveness_score\": 85,\n"
        "  \"verdict\": \"...\",\n"
        "  \"weak_areas\": [\"...\", \"...\"],\n"
        "  \"focus_recommendations\": [\"...\", \"...\"]\n"
        "}"
    )
    
    ai_resp = ask_llama("", analysis_prompt)
    analysis = {}
    try:
        clean_resp = re.sub(r'```json\s*|\s*```', '', ai_resp).strip()
        analysis = json.loads(clean_resp)
    except:
        analysis = {
            "effectiveness_score": int((score/total)*100),
            "verdict": "Completion based assessment.",
            "weak_areas": [],
            "focus_recommendations": []
        }

    return render_template('quiz_result.html', role_name=role_name, score=score, total=total, results=user_results, analysis=analysis)

@app.route('/roadmap/quiz/<int:step_index>')
@login_required
def take_roadmap_quiz(step_index):
    student = Student.get_by_id(session['user_id'])
    
    if not student or not student.roadmap or step_index >= len(student.roadmap):
        flash("Invalid roadmap step.")
        return redirect(url_for('roadmap'))
    
    step = student.roadmap[step_index]
    step_title = step.get('title', 'Unknown Topic')

    prompt = (
        f"Create a technical assessment quiz for the topic: '{step_title}'.\n"
        "Generate exactly 25 Multiple Choice Questions (MCQs) testing in-depth knowledge.\n"
        "Format strictly as a JSON object with this structure:\n"
        "{\n"
        "  \"questions\": [\n"
        "    {\n"
        "      \"id\": 1,\n"
        "      \"text\": \"Question text here?\",\n"
        "      \"options\": [\"Option A\", \"Option B\", \"Option C\", \"Option D\"],\n"
        "      \"correct_index\": 0\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "No Markdown. Return only valid JSON. Do not use trailing commas."
    )
    
    response_text = ask_llama("", prompt, max_tokens=3000)
    
    questions = []
    try:
        clean_json = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        data = json.loads(clean_json)
        questions = data.get('questions', [])
    except Exception as e:
        print(f"Roadmap Quiz Gen Error: {e}")
        flash("Could not generate quiz. Please try again.")
        return redirect(url_for('roadmap'))
        
    return render_template('quiz.html', 
                           role_name=f"Roadmap Step: {step_title}", 
                           questions=questions,
                           step_index=step_index,
                           submit_action=url_for('submit_roadmap_quiz'))

@app.route('/roadmap/quiz/submit', methods=['POST'])
@login_required
def submit_roadmap_quiz():
    step_index_str = request.form.get('step_index')
    if not step_index_str:
        flash("Error: Missing step index.")
        return redirect(url_for('roadmap'))
    
    step_index = int(step_index_str)
    student = Student.get_by_id(session['user_id'])
    
    quiz_data_str = request.form.get('quiz_data_json')
    if not quiz_data_str:
        return "Error: Missing quiz data", 400
        
    import json
    questions = json.loads(quiz_data_str)
    
    score = 0
    total = len(questions)
    
    # Grading
    for q in questions:
        qid = str(q['id'])
        selected_option = request.form.get(f"q_{qid}")
        correct_option = q['options'][q['correct_index']]
        if selected_option == correct_option:
            score += 1

    # Logic: Need 18 to pass
    PASSING_SCORE = 18
    
    if score >= PASSING_SCORE:
        # Update Roadmap
        if student.roadmap and step_index < len(student.roadmap):
            # Mark current as Completed
            student.roadmap[step_index]['status'] = 'Completed'
            
            # Unlock next if exists
            if step_index + 1 < len(student.roadmap):
                 student.roadmap[step_index + 1]['status'] = 'Focus'
            
            # Save to DB (update entire roadmap)
            student.update({'roadmap': student.roadmap})
            
            flash(f"Congratulations! You passed with {score}/{total}. Next step unlocked!")
    else:
        flash(f"You scored {score}/{total}. You need {PASSING_SCORE} to proceed. Review the material and try again.")

    return redirect(url_for('roadmap'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)