from flask import Flask, render_template, jsonify, request, session
import random

app = Flask(__name__)
app.secret_key = "dev_secret_key"

# --------------------
# MOCK DATA
# --------------------

MOCK_STUDENT = {
    'xp': 1250,
    'leetcode_username': 'mock_user_123',
    'data': {
        'github_id': 'octocat',
        'degree': 'B.Tech',
        'specialization': 'Computer Science'
    }
}

MOCK_SCORES = {
    'roles': [
        {'role': 'Full Stack Developer', 'score': 88},
        {'role': 'AI Engineer', 'score': 75},
        {'role': 'Data Scientist', 'score': 65}
    ],
    'best_fit': 'Full Stack Developer',
    'best_score': 88,
    'sde': 40,
    'fsd': 88,
    'ai': 75
}

MOCK_VERIFIED_SKILLS = [
    {'skill_name': 'Python'},
    {'skill_name': 'JavaScript'},
    {'skill_name': 'React'},
    {'skill_name': 'SQL'}
]

@app.context_processor
def inject_mock_data():
    return dict(
        student=MOCK_STUDENT,
        scores=MOCK_SCORES,
        verified_skills=MOCK_VERIFIED_SKILLS,
        bounty=MOCK_BOUNTY_QUESTIONS
    )

# --------------------
# HOME
# --------------------

@app.route("/")
def index():
    return render_template("dashboard.html")

# --------------------
# DASHBOARD
# --------------------

@app.route("/dashboard")
def dashboard():
    # Context processor handles the data injection
    return render_template("dashboard.html", 
                           is_solved_today=False,
                           bounty=True)

@app.route("/upload_resume", methods=['POST'])
def upload_resume():
    # Dummy route to handle resume upload form
    return render_template("dashboard.html", 
                           student=MOCK_STUDENT, 
                           scores=MOCK_SCORES, 
                           verified_skills=MOCK_VERIFIED_SKILLS, 
                           messages=["Resume uploaded successfully! (Mock)"])

@app.route("/upload_certificate", methods=['POST'])
def upload_certificate():
    # Dummy route to handle certificate upload form
    return render_template("dashboard.html", 
                           student=MOCK_STUDENT, 
                           scores=MOCK_SCORES, 
                           verified_skills=MOCK_VERIFIED_SKILLS, 
                           messages=["Certificate verified! (Mock)"])

@app.route("/leetcode_analysis", methods=['POST'])
def leetcode_analysis():
    # Dummy route
    session['leetcode_suggestion'] = "Focus on Dynamic Programming and Graph problems to improve your rating."
    return render_template("dashboard.html",
                           student=MOCK_STUDENT,
                           scores=MOCK_SCORES, 
                           verified_skills=MOCK_VERIFIED_SKILLS)

# --------------------
# RESUME
# --------------------

@app.route("/create-resume")
def create_resume():
    mock_resume_data = {
        'full_name': 'Alex Student',
        'institute_name': 'Tech University',
        'degree': 'B.Tech',
        'github_id': 'octocat',
        'leetcode_id': 'octocoder',
        'skills': 'Python, Flask, JavaScript, SQL',
        'specialization': 'Full Stack Development',
        'summary': 'Passionate developer with 3 years of coding experience.',
        'projects': [
            {'title': 'Portfolio Site', 'desc': 'Personal portfolio using React', 'link': 'https://github.com'}
        ],
        'achievements': 'Hackathon Winner 2024\nDean\'s List'
    }
    return render_template("create_resume.html", hide_sidebar=True, resume_data=mock_resume_data)

@app.route("/resume_builder")
def resume_builder():
    return render_template("resume_builder.html")

@app.route("/generate_resume", methods=['POST'])
def generate_resume():
    # In a static demo, we can just use the form data directly 
    # or fallback to our mock data if accessed directly
    if request.method == 'POST':
        data = {
            'full_name': request.form.get('full_name'),
            'institute_name': request.form.get('institute_name'),
            'specialization': request.form.get('specialization'),
            'email': session.get('user_email', 'student@example.com'),
            'github_url': f"https://github.com/{request.form.get('github_id', '')}",
            'leetcode_url': f"https://leetcode.com/{request.form.get('leetcode_id', '')}",
            'skills_list': [s.strip() for s in request.form.get('skills', '').split(',')],
            'degree': request.form.get('degree'),
            'achievements': [a.strip() for a in request.form.get('achievements', '').split('\n') if a.strip()],
            # Handle projects
        }
        
        # Simple project extraction from form lists (assuming array inputs or manual parsing)
        titles = request.form.getlist('project_title')
        descs = request.form.getlist('project_desc')
        links = request.form.getlist('project_link')
        
        projects = []
        for i in range(len(titles)):
            if titles[i]:
                projects.append({
                    'title': titles[i],
                    'desc': descs[i] if i < len(descs) else '',
                    'link': links[i] if i < len(links) else ''
                })
        data['projects'] = projects

        return render_template("generated_resume.html", data=data)
    
    return render_template("generated_resume.html", data={})

@app.route("/generated_resume")
def generated_resume_view():
    # If viewed directly without POST
    mock_data = {
        'full_name': 'Alex Student',
        'specialization': 'Software Engineer',
        'email': 'student@example.com',
        'institute_name': 'Tech University',
        'degree': 'B.Tech',
        'skills_list': ['Python', 'Flask', 'HTML/CSS'],
        'projects': [{'title': 'Demo Project', 'desc': 'A cool project', 'link': '#'}],
        'achievements': ['Won a prize']
    }
    return render_template("generated_resume.html", data=mock_data)

MOCK_BOUNTY_QUESTIONS = [
    {
        'skill': 'Python',
        'question': 'What is the output of print(2 ** 3)?',
        'options': ['6', '8', '9', '12'],
        'answer': 1
    },
    {
        'skill': 'Web',
        'question': 'Which tag is used for the largest heading?',
        'options': ['<h6>', '<head>', '<h1>', '<header>'],
        'answer': 2
    },
    {
        'skill': 'SQL',
        'question': 'Which command retrieves data?',
        'options': ['SELECT', 'UPDATE', 'DELETE', 'INSERT'],
        'answer': 0
    }
]

# --------------------
# GAME / BOUNTY
# --------------------

@app.route("/play_game")
def play_game():
    return render_template("game.html", bounty=MOCK_BOUNTY_QUESTIONS)

@app.route("/solve_game_bounty", methods=['POST'])
def solve_game_bounty():
    # Mock handling of game completion
    try:
        xp_gained = int(request.form.get('total_xp', 0))
        MOCK_STUDENT['xp'] += xp_gained
    except:
        pass
    return render_template("dashboard.html", 
                           student=MOCK_STUDENT, 
                           scores=MOCK_SCORES, 
                           verified_skills=MOCK_VERIFIED_SKILLS,
                           messages=[f"Great job! You earned {xp_gained} XP."])

# --------------------
# LEETCODE
# --------------------

@app.route("/leetcode")
def leetcode():
    return render_template("leetcode.html")

# --------------------
# ANALYSIS PAGES
# --------------------

@app.route("/analysis")
def analysis():
    return render_template("analysis.html", student=MOCK_STUDENT)

@app.route("/analyze", methods=['POST'])
def analyze_profile():
    # Return mock analysis data
    return jsonify({
        'primary_domain': 'Full Stack Web Development',
        'career_readiness': 'High',
        'projects': [
            {'repo_name': 'portfolio-v2', 'domain': 'Web', 'languages': ['HTML', 'CSS', 'JS']},
            {'repo_name': 'ecommerce-api', 'domain': 'Backend', 'languages': ['Python', 'SQL']}
        ],
        'missing_projects': ['Real-time Chat App', 'Cloud Deployment CI/CD Pipeline'],
        'github_domains_detected': ['Web Development', 'API Design'],
        'ai_suggestions': [
            {'title': 'Build a Real-time Chat App', 'description': 'Demonstrate WebSocket knowledge.', 'tech': 'Socket.IO / WebSockets'},
            {'title': 'Deploy to AWS', 'description': 'Learn cloud infrastructure.', 'tech': 'AWS EC2 / Lambda'}
        ]
    })

@app.route("/roadmap")
def roadmap():
    return render_template("roadmap.html")

@app.route("/market")
def market():
    return render_template("market.html")

# --------------------
# STATIC CONTENT PAGES
# --------------------

@app.route("/career")
def career():
    return render_template("career.html")

@app.route("/mentors")
def mentors():
    return render_template("mentors.html")

@app.route("/stories")
def stories():
    return render_template("stories.html")

@app.route("/institution")
def institution():
    return render_template("institution.html")

@app.route("/support")
def support():
    return render_template("support.html")

@app.route("/profile")
def profile():
    return render_template("profile.html")

@app.route("/skills")
def skills():
    return render_template("skills.html")

# --------------------
# API ROUTES
# --------------------

@app.route("/api/chat", methods=['POST'])
def chat_api():
    data = request.json
    message = data.get('message', '')
    # Simple mock response
    reply = f"That's a great question about '{message}'! As an AI mentor, I'd suggest focusing on building consistent habits."
    return jsonify({'reply': reply})

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Mock login success
        session['user_id'] = "mock_user_id"
        session['user_name'] = "Alex Student"
        # In a real app we would verify password here
        return dashboard() # Or redirect(url_for('dashboard')) but keeping it simple as per previous pattern
    return render_template("login.html")

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Mock signup success
        session['user_id'] = "mock_user_id"
        session['user_name'] = "New Student"
        return dashboard()
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return render_template("index.html")

# --------------------
# ENTRY POINT
# --------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
