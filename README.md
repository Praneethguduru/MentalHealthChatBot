# 🧠 MentalHealthChatBot

An AI-powered mental health support chatbot that talks with you, listens to your voice, and adapts its tone based on how you're feeling. It uses real clinical conversation data to give grounded, empathetic responses — not generic AI filler.

> ⚠️ **Supportive tool only. Not a clinical diagnosis system. Always consult a qualified mental health professional for medical advice.**

---

## 📌 Table of Contents

1. [What It Does](#-what-it-does)
2. [How It Works](#-how-it-works)
3. [Tech Stack](#-tech-stack)
4. [Requirements](#-requirements)
5. [Installation](#-installation)
6. [Using the App](#-using-the-app)
7. [API Reference](#-api-reference)
8. [Project Structure](#-project-structure)
9. [Training the Acoustic Model](#-training-the-acoustic-model-optional)
10. [Common Issues & Fixes](#-common-issues--fixes)
11. [FAQ](#-faq)
12. [Screenshots](#-screenshots)
13. [Crisis Support](#-crisis-support)
14. [Contributing](#-contributing)
15. [Team](#-team)

---

## ✨ What It Does

| Feature | Description |
|---------|-------------|
| 💬 **Grounded Chat** | Every response is anchored in real therapist–patient conversations from clinical interviews — so replies feel genuinely helpful, not like generic AI filler |
| 🎤 **Voice Input** | Speak naturally in your browser — no extra app needed. Includes a fix for a browser bug that silently dropped 40–60% of short voice messages |
| 📋 **PHQ-8 Assessment** | A clinically validated 8-question mental health check-in. Your score quietly shapes how the chatbot talks to you — from gentle exploration at low scores to professional referral language at high ones |
| 🔊 **Tone Adaptation** | The app listens to acoustic qualities in your voice (pitch, energy, rhythm) and privately adjusts its warmth level — more supportive when you seem distressed |
| 💾 **Conversation Memory** | Every session is saved. Come back the next day and pick up where you left off |
| 🔐 **Private Accounts** | Each user has their own encrypted account. Conversations live only on your server — nothing sent to third parties except the LLM API call |
| 🚨 **Crisis Safety** | Crisis keywords bypass the AI entirely and immediately show verified helpline numbers — always safe, always deterministic |

---

## ⚙️ How It Works

```
You type or speak a message
           │
           ▼
  ┌─────────────────────┐
  │  Crisis keyword?    │──── YES ──▶  Show verified helplines (AI skipped)
  └─────────────────────┘
           │ NO
           ▼
  Find the most relevant therapist dialogue
  from DAIC-WOZ clinical transcripts (ChromaDB)
           │
           ▼
  Assemble a 5-part prompt:
    1. Your PHQ-8 severity instruction
    2. Your voice tone level (if voice was used)
    3. Retrieved therapist example
    4. Last 15 messages of your conversation
    5. Your current message
           │
           ▼
  LLaMA 3.1 8B generates a response via Groq
  (typically 1–2 seconds)
           │
           ▼
  Response shown in chat + saved to your history
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend API | FastAPI + Uvicorn | Fast async Python server |
| Language Model | LLaMA 3.1 8B via Groq | Free, ~1–2s responses, open-weight |
| RAG Pipeline | LangChain + ChromaDB | Grounds responses in real clinical data |
| Embeddings | all-MiniLM-L6-v2 | Fast semantic search (384-dim vectors) |
| Voice (primary) | Web Speech API | On-device transcription, zero server cost |
| Voice (fallback) | Whisper tiny.en | Used if browser transcription fails |
| Acoustic Analysis | Librosa + scikit-learn | Extracts voice features for tone detection |
| Database | SQLite WAL + SQLAlchemy | Zero-config, concurrent-safe, fast |
| Auth | JWT + bcrypt | Secure login, no external auth service |
| Frontend | HTML + CSS + JavaScript | Single file, no build tools needed |

---

## 📋 Requirements

### System Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Check with `python --version` |
| RAM | 2 GB free minimum | 4 GB recommended |
| Disk space | 2 GB free | For models + database |
| Browser | Chrome 80+ or Edge 80+ | Required for voice input — Firefox not supported |
| ffmpeg | Any recent version | For audio conversion |

### Accounts You Need

| Account | Required? | Link |
|---------|-----------|------|
| Groq API key | ✅ Yes — free | [console.groq.com](https://console.groq.com) |
| DAIC-WOZ dataset | ❌ Optional | [dcapswoz.ict.usc.edu](https://dcapswoz.ict.usc.edu) — only needed to retrain the model |

---

## 🚀 Installation

### Step 1 — Clone the repo

```bash
git clone https://github.com/Praneethguduru/MentalHealthChatBot.git
cd MentalHealthChatBot
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows — Command Prompt
venv\Scripts\activate.bat

# Windows — PowerShell
venv\Scripts\Activate.ps1
```

You'll see `(venv)` in your terminal when it's active. Type `deactivate` to exit it later.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Takes 3–8 minutes on first run. Installs FastAPI, LangChain, ChromaDB, Librosa, Whisper, scikit-learn, and everything else.

### Step 4 — Install ffmpeg

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg -y

# macOS
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg

# Verify it worked
ffmpeg -version
```

### Step 5 — Get a Groq API key

1. Go to [console.groq.com](https://console.groq.com) and sign up (free, no credit card)
2. Click **API Keys** → **Create API Key**
3. Copy the key — you can only see it once

### Step 6 — Create your `.env` file

Create a file named `.env` in the project root:

```env
GROQ_API_KEY=gsk_paste_your_key_here
JWT_SECRET=any_long_random_string_at_least_32_chars
```

> Never commit `.env` to GitHub — it's already in `.gitignore`.
> You can also copy the template: `cp .env.example .env` then edit it.

### Step 7 — Build the knowledge base *(optional but recommended)*

If you have the DAIC-WOZ transcripts, place them in `dataset/transcripts/` then run:

```bash
python setup_chromadb.py
```

This indexes the transcripts into ChromaDB for RAG retrieval. Skip this if you don't have the dataset — the chatbot still works without it.

### Step 8 — Start the app

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and go to:

```
http://localhost:8000
```

✅ **The app is running!**

> **Note:** First startup takes 20–30 seconds while Whisper (~151MB) and the embedding model (~90MB) load. All starts after that are fast.

---

## 📱 Using the App

### Register & Log In
1. Open `http://localhost:8000` in Chrome or Edge
2. Click **Register**, enter a username and password
3. You're logged in automatically — sessions last 24 hours

### Text Chat
1. Type your message in the input box at the bottom
2. Press **Enter** or click **Send**
3. The AI responds in 1–3 seconds
4. Your conversation saves automatically

**Tips for better responses:**
- Be specific — *"I feel anxious before work presentations"* gets a better response than *"I'm anxious"*
- Ask follow-ups — the AI remembers your last 15 messages

### Voice Chat
1. Click the 🎤 **microphone button**
2. Allow microphone access when the browser asks
3. Speak naturally — click the button again when done
4. Your speech is transcribed and sent automatically
5. The app also quietly analyses your voice tone to adjust its warmth

> Works best in Chrome on desktop. Speak in a quiet place for best accuracy.

### PHQ-8 Assessment
1. Click **Take Assessment** in the menu
2. Answer 8 questions about the past 2 weeks (0–3 scale)
3. Submit — the chatbot adjusts its tone based on your score immediately

| Score | Severity | What changes |
|-------|----------|-------------|
| 0–4 | Minimal | Warm, focuses on wellbeing |
| 5–9 | Mild | Gentle exploration, self-care tips |
| 10–14 | Moderate | Acknowledges difficulty, nudges toward help |
| 15–19 | Moderately Severe | Strongly encourages professional support |
| 20–24 | Severe | Urgent referral language + crisis resources |

### Conversation History
- Click **History** in the navigation
- All past sessions listed by date
- Click any session to read or continue it

---

## 🌐 API Reference

Full interactive docs available at `http://localhost:8000/docs` when the server is running.

All endpoints except `/register` and `/login` require:
```
Authorization: Bearer your_jwt_token
```

### `POST /register` — Create account
```json
// Request
{ "username": "yourname", "password": "yourpassword" }

// Response
{ "access_token": "eyJhbG...", "token_type": "bearer" }
```

### `POST /login` — Log in
```json
// Request
{ "username": "yourname", "password": "yourpassword" }

// Response
{ "access_token": "eyJhbG...", "token_type": "bearer" }
```

### `POST /chat` — Send a message
```json
// Request
{
  "message": "I've been feeling overwhelmed at work lately",
  "conversation_id": "optional-uuid-to-continue-a-session"
}

// Response (immediate)
{ "task_id": "abc123-uuid" }
```

### `POST /voice` — Send voice message
```json
// Request
{
  "audio_data": "base64-encoded-webm-audio",
  "transcript": "text from Web Speech API",
  "conversation_id": "optional-uuid"
}

// Response (immediate)
{ "task_id": "abc123-uuid" }
```

### `POST /phq8` — Submit assessment
```json
// Request — exactly 8 integers, each 0–3
{ "answers": [0, 1, 2, 1, 0, 3, 1, 2], "conversation_id": "optional-uuid" }

// Response (HTTP 200 immediately — AI generates in background)
{ "task_id": "abc123-uuid", "score": 10, "severity": "Moderate" }
```

### `GET /task/{task_id}` — Poll for AI response
```json
// Still generating
{ "status": "pending" }

// Ready
{ "status": "complete", "response": "I hear you — that sounds really difficult..." }

// Error
{ "status": "error", "error": "description" }
```

> Poll every 2 seconds. Responses are usually ready in 1–3 seconds.

---

## 📁 Project Structure

```
MentalHealthChatBot/
│
├── main.py                       # FastAPI app — all 6 routes, startup events, task store
├── rag.py                        # RAG pipeline, ChromaDB, LLM call, crisis bypass
├── database.py                   # SQLite (WAL mode), SQLAlchemy ORM, 3 tables
├── auth.py                       # JWT tokens, bcrypt hashing, 60s user cache
│
├── audio_feature_extraction.py   # 9 acoustic feature groups via Librosa
├── voice_processor.py            # Whisper tiny.en wrapper (fallback STT)
├── voice_depression_detector.py  # Loads model, runs tone prediction on audio
│
├── improved_depression_model.py  # Training script — DAIC-WOZ → Random Forest
├── test_model.py                 # Test predictions on held-out participants
│
├── phq8.py                       # 8 questions, 0–3 scoring, 5-tier severity logic
├── phq8_therapeutic_responses.py # Severity tier → LLM prompt instruction map
├── setup_chromadb.py             # One-time script: index transcripts into ChromaDB
│
├── index.html                    # Entire frontend (HTML + CSS + JS) in one file
│
├── dataset/                      # DAIC-WOZ files go here (not included in repo)
│   ├── labels.csv                # Participant_ID and PHQ8_Total columns
│   ├── *_COVAREP.csv             # One per participant
│   ├── *_Formant.csv             # One per participant
│   └── transcripts/              # Interview transcript .txt files
│
├── chroma_db/                    # Auto-created by setup_chromadb.py
├── models/
│   └── depression_model_improved.pkl   # Auto-created by training script
├── mental_health.db              # Auto-created on first server start
│
├── screenshots/                  # Add your app screenshots here
├── requirements.txt
├── .env                          # Your secrets (never commit this)
├── .env.example                  # Template for .env
└── README.md
```

---

## 🧪 Training the Acoustic Model *(Optional)*

The repo ships with a pre-trained model. Use this only if you want to train from scratch with the DAIC-WOZ dataset.

### Prepare your data

```
dataset/
├── labels.csv              # Columns: Participant_ID, PHQ8_Total
├── 300_COVAREP.csv         # One COVAREP file per participant
├── 300_Formant.csv         # One Formant file per participant
└── ...
```

### Train

```bash
python improved_depression_model.py
```

What it does:
1. Loads COVAREP + Formant files for all participants
2. Computes 5 statistics per column → ~27,975 features per person
3. 80/20 stratified train/test split
4. Selects top 100 features (ANOVA F-test)
5. Trains Random Forest (100 trees, depth 5, balanced weights)
6. Runs 5-fold cross-validation
7. Saves full pipeline to `models/depression_model_improved.pkl`

### Test

```bash
python test_model.py
```

### About the model output

The model outputs a probability score mapped to three warmth levels:
- **Below 0.30** → Low warmth
- **0.30–0.60** → Moderate warmth
- **Above 0.60** → High warmth

This is injected privately into the LLM prompt — the user never sees it. It's a tone signal, not a diagnosis.

---

## 🐛 Common Issues & Fixes

### Installation

**`pip install` fails on Windows with "Microsoft Visual C++ required"**
Install Visual C++ Build Tools from [visualstudio.microsoft.com/visual-cpp-build-tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) then retry.

**Packages installing globally instead of into venv**
Make sure you see `(venv)` in your terminal. If not, activate it first:
```bash
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

### Startup

**`GROQ_API_KEY not set`**
- `.env` file must be in the same folder as `main.py`
- No spaces around `=`: `GROQ_API_KEY=gsk_...` ✅ not `GROQ_API_KEY = gsk_...` ❌
- No quotes: `GROQ_API_KEY=gsk_...` ✅ not `GROQ_API_KEY="gsk_..."` ❌
- Restart the server after editing `.env`

**Port 8000 already in use**
```bash
uvicorn main:app --port 8001 --reload
# Then open http://localhost:8001
```

**First startup takes 30+ seconds**
Normal — Whisper (151MB) and the embedding model (90MB) are loading into memory.

### Voice Input

**Voice records but sends nothing**
- Switch to Chrome or Edge — Firefox doesn't support Web Speech API
- Allow microphone access when the browser asks
- Only works on `http://localhost` or HTTPS — not plain HTTP on a remote IP

**Slow or inaccurate transcription**
- Speak in a quiet environment
- Use sentences of 5+ words — single words are less reliable
- Make sure no other app is using the microphone

### Runtime

**Groq API 401 Unauthorized**
Your key is wrong or expired. Generate a new one at [console.groq.com](https://console.groq.com).

**Groq API 429 Too Many Requests**
You've hit the free tier rate limit. Wait a minute — the limit resets quickly.

**`chromadb.errors.InvalidCollectionException`**
Run `python setup_chromadb.py` first, or skip if you don't have transcripts.

**Responses taking more than 5 seconds**
Check your internet connection — the Groq API call requires internet. The very first response after a cold start is always slower.

---

## ❓ FAQ

**Does this send my conversations to any external server?**
Messages are sent to the Groq API for LLM inference. Everything else — your account, history, PHQ-8 scores — stays on your own machine in the local SQLite database.

**Can I use this without the DAIC-WOZ dataset?**
Yes. Skip Steps 6–7 in installation. The chatbot works using the LLM alone, just without clinical conversation grounding.

**Can I use a different LLM?**
Yes. In `rag.py`, replace the `ChatGroq(...)` call with any LangChain-compatible LLM:
```python
# Example: OpenAI GPT-4o
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0.35, max_tokens=350)
```

**How do I reset everything and start fresh?**
Delete `mental_health.db` from the project root and restart the server. This clears all users and conversations.

**Can multiple users use this at the same time?**
Yes — FastAPI is async and handles concurrent users. Each user has their own separate account and conversation history.

**Can I deploy this online?**
Yes, but you need HTTPS for voice input (browsers require HTTPS for microphone access on non-localhost). Use nginx + Let's Encrypt, or deploy to Railway, Render, or any VPS with SSL.

---

## 🚨 Crisis Support

If the chatbot detects crisis-related language, the AI is bypassed entirely and these numbers appear immediately — hardcoded, always safe.

| Helpline | Number | Hours |
|----------|--------|-------|
| iCall | 9152987821 | Mon–Sat, 8am–10pm |
| Vandrevala Foundation | 1860-2662-345 | 24/7 |
| NIMHANS Helpline | 080-46110007 | 24/7 |
| Emergency | 100 | 24/7 |

---

## 🤝 Contributing

All contributions welcome — code, docs, bug reports, or ideas.

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "Add: describe your change"
git push origin feature/your-feature-name
# Open a Pull Request on GitHub
```

**Ideas for contributions:**
- [ ] Docker + docker-compose setup
- [ ] Hindi / Telugu / Tamil language support
- [ ] Streaming responses (SSE)
- [ ] PHQ-8 score history chart
- [ ] Dark mode
- [ ] Unit tests
- [ ] Mobile-friendly UI

---

## 👥 Team

| Name | 
|------|
| **Guduru Praneeth** | 
| **I Sai Prabhas Reddy** |
| **JA Kruthin** | 
| **K Vamshi Krishna Reddy** | 

---

*If this project helped you, please give it a ⭐ — it really helps!*
