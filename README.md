# 🚀 CareerHub AI

**Navigate Your Future with Intelligence.**

CareerHub (formerly EduTrack) is an advanced, AI-powered career navigation platform designed to guide students from learning to leadership. By analyzing profiles, academic records, and technical footprints (GitHub/LeetCode), CareerHub generates personalized roadmaps, internships opportunities, and actionable insights to accelerate career growth.

![CareerHub Landing](static/workshop.jpg) 
*(Note: Replace with actual screenshots if available)*

---

## 🌟 Key Features

### 🧠 **AI-Driven Logic**
- **Dynamic Profiling**: Continuously builds a live student profile based on skills, interests, and course progress.
- **Career GPS**: Uses Generative AI (Google Gemini) to suggest bespoke career paths matched to industry trends.
- **Resume Intelligence**: Upload your resume for AI analysis, parsing, and improvement suggestions.

### 🔗 **Integrations**
- **GitHub Analysis**: Connect your GitHub to inspect repositories, detect tech stacks (Software, ML, Web), and identify project gaps.
- **LeetCode Stats**: Visualize your algorithmic problem-solving progress directly on the dashboard.

### 🎮 **Gamification**
- **Daily Bounties**: A "Bounty Hunter" game mode where you shoot targets to answer technical quizzes and earn XP.
- **XP & Leaderboard**: Track your growth and compete with others.

### 🗺️ **Roadmap Builder**
- **Step-by-Step Guidance**: tailored milestones from beginner to expert.
- **Resource Recommendation**: Curated courses and projects for every stage.

### 🎨 **Modern UI/UX**
- **Immersive Landing**: Features GSAP animations and scroll-triggered storytelling.
- **Glassmorphism Design**: sleek, dark-themed dashboard with responsive elements.

---

## 🛠️ Technology Stack

- **Backend**: Python (Flask)
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript
- **Animations**: GSAP, Lenis Smooth Scroll
- **AI Models**: Google Gemini Pro (GenAI), LLaMA (via API)
- **Database**: Firebase / Firestore
- **Deployment**: Vercel / Heroku Support (`Procfile` included)

---

## ⚡ Getting Started

### Prerequisites
- Python 3.9+
- Git

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Pranatheesh-S/AI-IGNITE.git
   cd AI-IGNITE
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   SECRET_KEY=your_flask_secret
   GOOGLE_API_KEY=your_gemini_api_key
   FIREBASE_CREDENTIALS=path/to/firebase_key.json
   GITHUB_TOKEN=your_github_token (optional)
   ```

4. **Run the Application**
   ```bash
   python app/app.py
   ```
   Access the app at `http://127.0.0.1:5000`

---

## 📂 Project Structure

```
AI-IGNITE/
│
├── app/
│   └── app.py            # Main application entry point
│
├── static/               # CSS, JS, Images, Videos
│   ├── style.css
│   ├── bg_video.mp4
│   └── ...
│
├── templates/            # HTML Templates (Jinja2)
│   ├── index.html        # Landing Page
│   ├── dashboard.html    # User Dashboard
│   ├── game.html         # Bounty Hunter Game
│   └── ...
│
├── requirements.txt      # Python dependencies
├── Procfile              # Deployment configuration
└── README.md             # Documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and create a pull request for any feature enhancements or bug fixes.

---

## 📜 License

This project is licensed under the MIT License.

---

**Built with ❤️ by the CareerHub Team.**