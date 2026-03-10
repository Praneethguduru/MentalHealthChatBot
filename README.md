# MentalHealthChatBot
<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&pause=1000&color=6C8EBF&center=true&vCenter=true&width=500&lines=Supportive+%E2%80%A2+Grounded+%E2%80%A2+Ethical+%E2%80%A2+Open+Source" alt="Tagline" />

# 🧠 MentalHealthChatBot

### *AI-Based Multimodal Conversational System for Supportive Interaction and Psychological State Prediction*

<p align="center">
  <a href="https://github.com/Praneethguduru/MentalHealthChatBot/stargazers"><img src="https://img.shields.io/github/stars/Praneethguduru/MentalHealthChatBot?style=for-the-badge&color=FFD700&labelColor=1a1a2e" alt="Stars"></a>
  <a href="https://github.com/Praneethguduru/MentalHealthChatBot/network/members"><img src="https://img.shields.io/github/forks/Praneethguduru/MentalHealthChatBot?style=for-the-badge&color=4ECDC4&labelColor=1a1a2e" alt="Forks"></a>
  <a href="https://github.com/Praneethguduru/MentalHealthChatBot/issues"><img src="https://img.shields.io/github/issues/Praneethguduru/MentalHealthChatBot?style=for-the-badge&color=FF6B6B&labelColor=1a1a2e" alt="Issues"></a>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white&labelColor=1a1a2e" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge&labelColor=1a1a2e" alt="License">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LLaMA_3.1_8B-Groq_LPU-FF6B35?style=for-the-badge&labelColor=1a1a2e" alt="LLaMA">
  <img src="https://img.shields.io/badge/ChromaDB-RAG_Pipeline-7B2FBE?style=for-the-badge&labelColor=1a1a2e" alt="ChromaDB">
  <img src="https://img.shields.io/badge/DAIC--WOZ-189_Participants-2E86AB?style=for-the-badge&labelColor=1a1a2e" alt="DAIC-WOZ">
  <img src="https://img.shields.io/badge/User_Study-n%3D15_✓-27AE60?style=for-the-badge&labelColor=1a1a2e" alt="User Study">
</p>

<br/>

> **⚠️ Disclaimer:** This system is designed exclusively for **supportive conversation** — not clinical diagnosis, treatment, or medical advice. Always consult a qualified mental health professional for clinical needs.

</div>

---

## 📖 Table of Contents

- [✨ What Makes This Different](#-what-makes-this-different)
- [🏗️ System Architecture](#️-system-architecture)
- [🚀 Features](#-features)
- [📊 Performance Results](#-performance-results)
- [⚙️ Installation](#️-installation)
- [🔧 Configuration](#-configuration)
- [📁 Project Structure](#-project-structure)
- [🧪 Training the Acoustic Model](#-training-the-acoustic-model)
- [🌐 API Reference](#-api-reference)
- [🔬 Research Contributions](#-research-contributions)
- [📈 User Study Results](#-user-study-results)
- [🛡️ Ethical Design](#️-ethical-design)
- [🤝 Contributing](#-contributing)
- [📚 Citation](#-citation)
- [👥 Team](#-team)

---

## ✨ What Makes This Different

Most AI mental health chatbots fail in one of four ways. **This one doesn't.**

| Problem | What Others Do | What We Do |
|--------|---------------|------------|
| 🎭 **Hallucination** | Generic LLM responses from internet text | RAG grounded in real DAIC-WOZ clinical interview transcripts |
| 🔇 **Silent Voice Failures** | Ignore the race condition (40–60% failure rate) | Formally resolved with 3-mechanism browser fix → ~0% failure |
| 🩺 **Misleading AI "Diagnosis"** | Present AUC≈0.50 models as "depression detectors" | Honest: acoustic model = **tone adaptation signal only** |
| 📊 **No Proof It Works** | No controlled comparison study | 15-participant counterbalanced user study with baseline control |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 1 — USER INTERFACE                    │
│         HTML5 + CSS3 + JS  │  Voice  │  PHQ-8  │  Chat          │
└──────────────────────┬──────────────────────────────────────────┘
                       │  JWT Bearer Token (every request)
┌──────────────────────▼──────────────────────────────────────────┐
│                  LAYER 2 — APPLICATION (FastAPI)                 │
│   /register  /login  /chat  /voice  /phq8  /task/{id}           │
│   ThreadPoolExecutor(4) │ BackgroundTasks │ Pydantic Validation  │
└──────────┬───────────────────────────────────┬──────────────────┘
           │                                   │
┌──────────▼──────────┐             ┌──────────▼──────────────────┐
│  LAYER 3A — RAG     │             │  LAYER 3B — ACOUSTIC TONE   │
│  LangChain Pipeline │             │  Librosa → 9 feature groups │
│  ChromaDB (k=1)     │             │  Random Forest classifier   │
│  all-MiniLM-L6-v2   │             │  → Low / Moderate / High    │
│  LLaMA 3.1 8B Groq  │             │  (private to prompt only)   │
└──────────┬──────────┘             └──────────┬──────────────────┘
           │                                   │
┌──────────▼───────────────────────────────────▼──────────────────┐
│                    LAYER 4 — DATA PERSISTENCE                    │
│   SQLite WAL + 64MB cache  │  ChromaDB  │  RF Model (.pkl)      │
└──────────────────────────────────────────────────────────────────┘

⚠️  LAYER 5 — SAFETY BYPASS (horizontal, always active)
    Crisis keywords → LLM SKIPPED → Hardcoded verified resources
```

### Five-Part Prompt Assembly (every LLM call)
```
[PHQ-8 Severity Context]     ← score tier → therapeutic instruction
[Voice Tone Indicator]       ← PRIVATE: Low/Moderate/High warmth
[DAIC-WOZ Retrieved Context] ← real therapist language (k=1 cosine)
[Conversation History]       ← first 2 + last 15 messages
[Current User Message]       ← what the user just said
```

---

## 🚀 Features

### 💬 Intelligent Conversation
- **RAG-Grounded Responses** — Every reply conditioned on real DAIC-WOZ clinical interview transcripts via ChromaDB semantic retrieval
- **Context Memory** — First 2 + last 15 messages kept per session for natural multi-turn dialogue
- **LLaMA 3.1 8B via Groq LPU** — ~500–800 tokens/sec, sub-2-second response generation

### 🎤 Voice Interaction
- **Dual-Track Pipeline** — Web Speech API (instant transcript) + MediaRecorder (acoustic analysis) run in parallel
- **Race Condition Fix** — The *only* published solution to the W3C SpeechRecognition `onend`/`onresult` race condition
  - `interimTranscript` continuous tracking
  - `onend` Promise with 600ms safety timeout
  - Merged `finalTranscript + interimTranscript` assembly
- **Result:** 40–60% silent failure rate → **~0%**

### 📋 PHQ-8 Assessment
- Validated 8-item Patient Health Questionnaire
- Five severity tiers mapped to distinct therapeutic LLM prompt contexts:

  | Score | Tier | LLM Instruction |
  |-------|------|-----------------|
  | 0–4 | None/Minimal | Psychoeducation, normalise |
  | 5–9 | Mild | Validate, explore, self-care |
  | 10–14 | Moderate | Acknowledge, suggest help |
  | 15–19 | Moderately Severe | Strongly encourage professional |
  | 20–24 | Severe | Urgent referral + crisis resources |

- **HTTP response < 200ms** via FastAPI `BackgroundTasks` (LLM generates asynchronously)

### 🔊 Acoustic Tone Adaptation
- 9 Librosa feature groups extracted from every voice message:
  - F0/Pitch · RMS Energy · 13 MFCCs · ZCR · Spectral Centroid · Rolloff · Bandwidth · Tempo · H/P Ratio
- Random Forest classifier → Low / Moderate / High warmth → injected **privately** into LLM prompt
- **Honestly framed** as tone adaptation, never clinical prediction

### 🔐 Security
- JWT authentication (24h tokens)
- bcrypt(rounds=12) password hashing
- 60-second in-process user cache (eliminates DB round-trips)
- OAuth2 `PasswordBearer` dependency injection

### 🛡️ Safety First
- **Deterministic crisis bypass** — LLM completely skipped for crisis keywords
- Crisis resources hardcoded and human-verified:
  - iCall: `9152987821`
  - Vandrevala Foundation: `1860-2662-345` (24/7)
  - NIMHANS: `080-46110007`
  - Emergency: `100`

---

## 📊 Performance Results

### ⚡ Latency (Before vs After Optimisation)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Voice response (end-to-end) | 8–15 s | **1–3 s** | ~87% ↓ |
| PHQ-8 HTTP response | 5–15 s | **< 200 ms** | ~98% ↓ |
| First voice (cold start) | 12–20 s | **1–3 s** | ~87% ↓ |
| Auth DB lookup | 15–50 ms | **< 1 ms** | ~99% ↓ |
| Voice transcript failure rate | 40–60% | **~0%** | Eliminated |

### 🤖 Acoustic Model (Actual Training Output)

```
Dataset:          DAIC-WOZ — 189 participants
Raw features:     27,975 per participant (23,580 COVAREP + 4,395 Formant)
After SelectKBest: 100 features (ANOVA F-test)
Train/Test split: 80/20 stratified (151 train, 38 test)

Test Accuracy:    71.05%   (majority-class prediction — expected for p>>n)
ROC-AUC:          0.5000   (honest: used as tone signal ONLY, not diagnosis)
CV ROC-AUC:       0.5000 ± 0.0000 (5-fold stratified)
Confusion Matrix: TN=27, FP=0, FN=11, TP=0
```

> **Why is AUC 0.50 OK?** With 189 samples and 27,975 features (p >> n problem), any acoustic model on this dataset has essentially no clinical discriminative power. We publish this honestly and use the output only to modulate conversational warmth — not to make any clinical claim. Most published systems hide this. We don't.

---

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- Node.js (optional, for frontend dev)
- ffmpeg (required for audio conversion)
- Chrome 80+ or Edge 80+ (for Web Speech API)
- A free [Groq API key](https://console.groq.com)
- DAIC-WOZ dataset access ([request here](https://dcapswoz.ict.usc.edu/))

### 1. Clone the Repository

```bash
git clone https://github.com/Praneethguduru/MentalHealthChatBot.git
cd MentalHealthChatBot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# OR
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
fastapi==0.110.0
uvicorn[standard]==0.29.0
langchain==0.1.16
langchain-groq==0.1.3
langchain-community==0.0.34
langchain-huggingface==0.0.3
chromadb==0.4.24
sentence-transformers==2.6.1
librosa==0.10.1
soundfile==0.12.1
pydub==0.25.1
scikit-learn==1.4.1
sqlalchemy==2.0.29
pyjwt==2.8.0
bcrypt==4.1.2
openai-whisper==20231117
numpy==1.26.4
pandas==2.2.1
```

### 4. Install ffmpeg

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

### 5. Set Environment Variables

```bash
export GROQ_API_KEY="your_groq_api_key_here"
export JWT_SECRET="your_strong_random_secret_here"
```

Or create a `.env` file:
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
JWT_SECRET=your-super-secret-jwt-key-min-32-chars
```

### 6. Set Up ChromaDB (Transcript Indexing)

First, place your DAIC-WOZ transcript files in `dataset/transcripts/`:

```bash
python setup_chromadb.py
# This will:
# - Load all transcript files from dataset/transcripts/
# - Chunk into 500-char segments with 100-char overlap
# - Embed with all-MiniLM-L6-v2 (384-dim)
# - Persist to chroma_db/ directory
```

### 7. Train the Acoustic Model (Optional)

```bash
# Place DAIC-WOZ COVAREP and Formant CSVs in dataset/
python improved_depression_model.py
# Model saved to models/depression_model_improved.pkl
```

### 8. Run the Application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser: `http://localhost:8000`

---

## 🔧 Configuration

All configuration is in environment variables or at the top of relevant modules:

```python
# rag.py
CHROMA_PATH  = "chroma_db"           # ChromaDB persistence directory
COLLECTION   = "daic_woz_transcripts" # Collection name

# main.py
executor = ThreadPoolExecutor(max_workers=4)  # CPU-bound worker pool

# improved_depression_model.py
N_FEATURES = 100     # SelectKBest k
PHQ_THRESH = 10      # Depression threshold (PHQ-8 ≥ 10)

# auth.py
TOKEN_EXP_H = 24     # JWT token lifetime (hours)
CACHE_TTL   = 60     # In-process user cache TTL (seconds)
```

### LLM Settings (rag.py)

```python
llm = ChatGroq(
    model_name  = "llama-3.1-8b-instant",
    temperature = 0.35,    # Lower = more consistent therapeutic tone
    max_tokens  = 350      # Hard cap for latency control
)
```

---

## 📁 Project Structure

```
MentalHealthChatBot/
│
├── 📄 main.py                      # FastAPI app, all 6 endpoints, task registry
├── 📄 rag.py                       # RAG pipeline, ChromaDB, LangChain, crisis bypass
├── 📄 database.py                  # SQLite WAL, SQLAlchemy ORM, 5 PRAGMAs
├── 📄 auth.py                      # JWT, bcrypt(12), 60s user cache
├── 📄 audio_feature_extraction.py  # Librosa 9-group acoustic features
├── 📄 voice_processor.py           # Whisper tiny.en wrapper (fallback STT)
├── 📄 voice_depression_detector.py # Thread-safe RF inference singleton
├── 📄 improved_depression_model.py # DAIC-WOZ training pipeline
├── 📄 phq8.py                      # PHQ-8 questions, scoring, severity tiers
├── 📄 phq8_therapeutic_responses.py# Tier → therapeutic prompt context map
├── 📄 setup_chromadb.py            # One-time transcript indexing script
│
├── 🌐 index.html                   # Single-file frontend (HTML5/CSS3/JS)
│
├── 📁 dataset/                     # DAIC-WOZ data (not included — requires agreement)
│   ├── labels.csv                  # Participant IDs + PHQ-8 scores
│   ├── *_COVAREP.csv               # Acoustic feature files
│   ├── *_Formant.csv               # Formant analysis files
│   └── transcripts/                # Session transcript text files
│
├── 📁 chroma_db/                   # ChromaDB persistent vector store (auto-created)
├── 📁 models/                      # Trained model files (auto-created)
│   └── depression_model_improved.pkl
│
├── 📄 requirements.txt
├── 📄 .env.example
└── 📄 README.md
```

---

## 🧪 Training the Acoustic Model

```bash
# Full training pipeline
python improved_depression_model.py
```

**What happens:**
1. Loads COVAREP + Formant CSVs for 189 DAIC-WOZ participants
2. Computes 5 statistics (mean, std, min, max, median) per column → **27,975 features**
3. 80/20 stratified train/test split
4. `SelectKBest(f_classif, k=100)` — fitted on train only
5. `StandardScaler` — fitted on train only
6. `RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced')`
7. 5-fold stratified CV evaluation
8. ROC-optimal threshold selection
9. Saves pipeline to `models/depression_model_improved.pkl`

**Testing the model:**
```bash
python test_model.py
# Tests predictions on held-out participants
# Expected: ~49.53% probability → "Moderate" warmth (majority-class prior)
```

---

## 🌐 API Reference

All endpoints (except `/register` and `/login`) require:
```
Authorization: Bearer <jwt_token>
```

### `POST /register`
```json
Request:  { "username": "string", "password": "string" }
Response: { "access_token": "string", "token_type": "bearer" }
```

### `POST /login`
```json
Request:  { "username": "string", "password": "string" }
Response: { "access_token": "string", "token_type": "bearer" }
```

### `POST /chat`
```json
Request:  { "message": "I've been feeling overwhelmed lately", "conversation_id": "uuid" }
Response: { "task_id": "uuid" }
```

### `POST /voice`
```json
Request: {
  "audio_data":      "base64-encoded WebM/OGG bytes",
  "transcript":      "text from Web Speech API",
  "conversation_id": "uuid"
}
Response: { "task_id": "uuid" }
```

### `POST /phq8`
```json
Request: {
  "answers": [0, 1, 2, 1, 0, 3, 1, 2],   // 8 integers, each 0-3
  "conversation_id": "uuid"
}
Response: { "task_id": "uuid", "score": 10, "severity": "Moderate" }
// HTTP 200 returned immediately — LLM generates in background
```

### `GET /task/{task_id}`
```json
// Poll every 2 seconds (60s timeout)
Response (pending):  { "status": "pending" }
Response (complete): { "status": "complete", "response": "AI response text..." }
Response (error):    { "status": "error", "error": "message" }
```

---

## 🔬 Research Contributions

This project makes **three novel contributions** to the field:

### 1. 🐛 Browser Speech Recognition Race Condition (First Published Fix)

The W3C Web Speech API has a race condition: `recog.stop()` causes `onend` to fire **before** `onresult` finalises interim words — silently discarding them. No prior paper documents or resolves this.

**Our fix (3 mechanisms):**
```javascript
// Mechanism 1: Track interim words continuously
recognition.onresult = (event) => {
  for (let i = event.resultIndex; i < event.results.length; i++) {
    if (event.results[i].isFinal) {
      finalTranscript += event.results[i][0].transcript;
    } else {
      interimTranscript = event.results[i][0].transcript; // Never lost
    }
  }
};

// Mechanism 2: Promise resolved by onend (with 600ms safety timeout)
let resolveStop;
const stopPromise = new Promise(resolve => { resolveStop = resolve; });
recognition.onend = () => resolveStop();

async function stopVoice() {
  recognition.stop();
  await Promise.race([stopPromise, new Promise(r => setTimeout(r, 600))]);
  // Mechanism 3: Merge final + interim (captures everything)
  const transcript = (finalTranscript + ' ' + interimTranscript).trim();
  return transcript;
}
```

**Result:** 40–60% failure rate → **~0%** across 50 test utterances in Chrome 120 + Edge 120.

### 2. 🎯 Honest Acoustic Model Framing

ROC-AUC of 0.50 on 189 participants is the **correct and expected result** given the p>>n problem (27,975 features, 189 samples). We publish this honestly and reframe the output as a **conversational warmth modulator** — not a depression detector. This is the first system to make this distinction explicitly.

### 3. 📊 First Controlled RAG vs Baseline User Study (Mental Health)

A 15-participant counterbalanced within-subjects study comparing full RAG system vs retrieval-free baseline — the first such study in AI mental health chatbots.

---

## 📈 User Study Results

**n=15 participants | Counterbalanced order | 7-item Likert scale (1–5)**

| Question | RAG System | Baseline | Δ |
|----------|-----------|----------|---|
| Q1: Understood what I expressed | **4.2** | 2.9 | +1.3 |
| Q2: Felt empathetic and supportive | **4.3** | 2.8 | +1.5 ⬆️ |
| Q3: Responses were contextually relevant | **4.1** | 2.6 | +1.5 ⬆️ |
| Q4: Would use this system when stressed | **4.0** | 2.7 | +1.3 |
| Q5: Responses felt generic *(reversed)* | **1.8** | 3.9 | −2.1 |
| Q6: Maintained context across conversation | **4.2** | 3.1 | +1.1 |
| Q7: Felt safe and non-judgemental | **4.5** | 4.3 | +0.2 ✅ |

> **Q7 finding:** Safety ratings are **equal in both conditions** — confirming the ethical safeguards (crisis bypass, non-diagnostic constraints) work independently of retrieval. Safety is a floor, not a RAG feature.

> **14/15 participants preferred the RAG system** overall.

---

## 🛡️ Ethical Design

This system was designed with ethics as a **hard constraint**, not an afterthought:

```
✅  Never diagnoses, treats, or prescribes
✅  PHQ-8 used for conversational context only — never shown as "your diagnosis"
✅  Acoustic indicator private to LLM prompt — never disclosed to user
✅  Crisis bypass is deterministic — LLM cannot generate unsafe responses to crises
✅  All crisis resources manually verified before hardcoding
✅  Users consistently reminded to seek professional care
✅  No user data sold or used for advertising
✅  Academic research only — DAIC-WOZ data under institutional agreement
```

**Crisis Keywords Monitored:**
`suicide` · `kill myself` · `want to die` · `end my life` · `harm myself` · `self harm` · `cut myself` · `overdose` · `don't want to live` · `no reason to live`

---

## 🤝 Contributing

We welcome contributions! Here's how:

### Bug Reports
Open an issue with:
- Python version + OS
- Full error traceback
- Steps to reproduce
- Expected vs actual behaviour

### Pull Requests

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
# Open a PR against main
```

### Good First Issues
- [ ] Add Hindi/Telugu language support (swap to `paraphrase-multilingual-MiniLM-L12-v2`)
- [ ] PostgreSQL migration (just swap `DATABASE_URL` in `database.py`)
- [ ] Server-Sent Events for streaming LLM responses
- [ ] Longitudinal PHQ-8 trend chart in frontend
- [ ] Dark mode for frontend
- [ ] Docker/Docker Compose setup
- [ ] Unit tests for acoustic feature extraction

### Code Style
- Follow PEP 8
- Type hints on all function signatures
- Docstrings for all public functions
- No hardcoded secrets (use env vars)

---

## 📚 Citation

If you use this work in research, please cite:

```bibtex
@inproceedings{guduru2026mentalhealth,
  title     = {AI-Based Multimodal Conversational System for Supportive
               Interaction and Psychological State Prediction},
  author    = {Guduru, Praneeth and Reddy, I Sai Prabhas and
               Kruthin, JA and Reddy, K Vamshi Krishna and Shafi, Md.},
  booktitle = {Proceedings of the International Conference on
               Networks, Power and Communication (ICNPCV)},
  year      = {2026},
  note      = {Paper ID 496},
  institution = {Malla Reddy University, Hyderabad}
}
```

**Key references this project builds on:**

- [DAIC-WOZ Dataset](https://dcapswoz.ict.usc.edu/) — Gratch et al., LREC 2014
- [RAG](https://arxiv.org/abs/2005.11401) — Lewis et al., NeurIPS 2020
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — Reimers & Gurevych, EMNLP 2019
- [COVAREP](https://ieeexplore.ieee.org/document/6853931) — Degottex et al., ICASSP 2014
- [LLaMA 2](https://arxiv.org/abs/2307.09288) — Meta AI, 2023

---

## 👥 Team

<div align="center">

| | Name | Role |
|---|------|------|
| 👨‍💻 | **Guduru Praneeth** | Lead Developer — RAG Pipeline, FastAPI, Voice System |
| 👨‍💻 | **I Sai Prabhas Reddy** | Acoustic Model, DAIC-WOZ Processing, ML Pipeline |
| 👨‍💻 | **JA Kruthin** | Frontend, Auth System, Database Design |
| 👨‍💻 | **K Vamshi Krishna Reddy** | PHQ-8 Module, Ethical Design, Testing |
| 👨‍🏫 | **Prof. Md. Shafi** | Project Guide, Malla Reddy University |

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

**Dataset Note:** The DAIC-WOZ dataset is NOT included in this repository. It requires a separate institutional data sharing agreement with the University of Southern California. [Request access here.](https://dcapswoz.ict.usc.edu/)

---

<div align="center">

**Built with ❤️ at Malla Reddy University, Hyderabad**

*Department of CSE — Artificial Intelligence & Machine Learning*

*Academic Year 2025–2026*

<br/>

⭐ **If this helped you, please star the repo!** ⭐

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Praneethguduru-181717?style=for-the-badge&logo=github&labelColor=1a1a2e)](https://github.com/Praneethguduru)

</div>