import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter # Added import for warning fix
from datetime import datetime
import os
from dotenv import load_dotenv

# Load env variables (looks for .env in current or parent dirs)
load_dotenv()

# 1. Initialize Firebase (Singleton Check)
db = None
try:
    if not firebase_admin._apps:
        private_key = os.getenv("FIREBASE_PRIVATE_KEY")
        if private_key:
            private_key = private_key.replace("\\n", "\n")
        
        cred = credentials.Certificate({
            "type": "service_account",
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key": private_key,
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "token_uri": "https://oauth2.googleapis.com/token"
        })
        firebase_admin.initialize_app(cred)

    db = firestore.client()
except Exception as e:
    print(f"Warning: Firebase initialization failed. Database features will be unavailable. Error: {e}")
    db = None


db = firestore.client()

class Student:
    """
    Acts as the interface for the 'students' collection in Firestore.
    Replaces the SQLAlchemy 'Student' model.
    """
    collection_name = 'students'

    def __init__(self, uid, data):
        self.id = uid
        self.data = data
        
        # Map common fields for easy access (dot notation)
        # This mimics SQLAlchemy object access (e.g., student.name)
        self.name = data.get('name')
        self.email = data.get('email')
        self.password_hash = data.get('password_hash')
        self.enrollment_year = data.get('enrollment_year')
        self.department = data.get('department')
        self.leetcode_username = data.get('leetcode_username')
        
        # Scores & Game Stats
        self.xp = data.get('xp', 0)
        self.sde_score = data.get('sde_score', 0)
        self.fsd_score = data.get('fsd_score', 0)
        self.ai_score = data.get('ai_score', 0)
        self.last_bounty_date = data.get('last_bounty_date')

        # JSON/Dict Fields
        self.top_roles = data.get('top_roles', [])
        self.roadmap = data.get('roadmap', [])
        self.market_analysis = data.get('market_analysis', {})
        self.verified_skills = data.get('verified_skills', []) # Replaces Skills Relationship
        self.resume_profile = data.get('resume_profile', {}) # Stores structured resume details

    @staticmethod
    def create(name, email, password_hash, department, enrollment_year):
        """Creates a new student document."""
        if db is None: return None
        new_data = {
            'name': name,
            'email': email,
            'password_hash': password_hash,
            'department': department,
            'enrollment_year': enrollment_year,
            'created_at': datetime.utcnow(),
            'xp': 0,
            'verified_skills': [],
            'top_roles': [],
            'roadmap': [],
            'resume_profile': {},
            'market_analysis': None
        }
        # Add to Firestore
        update_time, doc_ref = db.collection(Student.collection_name).add(new_data)
        return Student(doc_ref.id, new_data)

    @staticmethod
    def get_by_id(uid):
        """Fetches a student by Document ID."""
        if db is None: return None
        doc_ref = db.collection(Student.collection_name).document(uid)
        doc = doc_ref.get()
        if doc.exists:
            return Student(doc.id, doc.to_dict())
        return None

    @staticmethod
    def get_by_email(email):
        """Fetches a student by Email (Query)."""
        if db is None: return None
        # Updated to use FieldFilter to resolve warning
        docs = db.collection(Student.collection_name).where(filter=FieldFilter('email', '==', email)).stream()
        for doc in docs:
            return Student(doc.id, doc.to_dict())
        return None

    def update(self, updates):
        """
        Updates specific fields in Firestore and the local object.
        Example: student.update({'xp': 150})
        """
        if db is None: return
        doc_ref = db.collection(self.collection_name).document(self.id)
        doc_ref.update(updates)
        # Update local attributes
        self.data.update(updates)
        for key, value in updates.items():
            setattr(self, key, value)

    # --- Handling "Relationships" (Subcollections vs Arrays) ---

    def add_skill(self, skill_name, proficiency=5, verified=False):
        """
        Adds a skill to the 'verified_skills' ARRAY inside the student doc.
        Replaces: db.session.add(Skill(...))
        """
        if db is None: return False, "Database unavailable"
        new_skill = {
            'skill_name': skill_name,
            'proficiency': proficiency,
            'verified': verified,
            'added_at': datetime.utcnow()
        }
        
        # Check for duplicates locally
        for s in self.verified_skills:
            if s['skill_name'].lower() == skill_name.lower():
                return False, "Skill already exists"

        # Firestore ArrayUnion ensures uniqueness of the exact object, 
        # but we handle logic manually for safety.
        current_skills = self.verified_skills
        current_skills.append(new_skill)
        
        self.update({'verified_skills': current_skills})
        return True, "Skill added"

    def add_resume(self, filename, ocr_content):
        """
        Adds a resume to the 'resumes' SUBCOLLECTION.
        Resumes are heavy (OCR text), so we don't keep them in the main doc.
        """
        if db is None: return
        resume_data = {
            'filename': filename,
            'ocr_content': ocr_content,
            'uploaded_at': datetime.utcnow()
        }
        db.collection(self.collection_name).document(self.id).collection('resumes').add(resume_data)

    def get_latest_resume(self):
        """Fetches the most recent resume from the subcollection."""
        if db is None: return None
        docs = db.collection(self.collection_name).document(self.id)\
                 .collection('resumes')\
                 .order_by('uploaded_at', direction=firestore.Query.DESCENDING)\
                 .limit(1).stream()
        
        for doc in docs:
            return doc.to_dict()
        return None

    def add_academic_record(self, semester, gpa, courses=None):
        """Adds record to 'academic_records' SUBCOLLECTION."""
        if db is None: return
        record_data = {
            'semester': semester,
            'gpa': gpa,
            'courses': courses or [],
            'added_at': datetime.utcnow()
        }
        db.collection(self.collection_name).document(self.id).collection('academic_records').add(record_data)

    def add_career_goal(self, role, industry):
        """Adds goal to 'career_goals' ARRAY in main doc (lightweight)."""
        if db is None: return
        goal = {'role': role, 'industry': industry}
        # Use ArrayUnion to append
        doc_ref = db.collection(self.collection_name).document(self.id)
        doc_ref.update({'career_goals': firestore.ArrayUnion([goal])})

# --- Helper Classes (Optional) ---
# Since we don't have strictly defined schemas, we don't *need* classes for 
# Resume/Skill, but we use the methods inside 'Student' to manage them.