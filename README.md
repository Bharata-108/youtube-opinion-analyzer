# 🎬 YouTube Sentiment & Dynamic Opinion Analysis System

A comprehensive Natural Language Processing (NLP) and opinion mining platform tailored for **2nd-Year CSE undergraduate study and demonstration**.

Unlike conventional sentiment analysis tools that merely report positive/negative numbers, this system extracts the **exact viewpoints** viewers express (e.g., *"Appreciation of Cinematography"*, *"Criticism of Storyline"*, *"Accusations of Political Propaganda"*, *"Praise for Background Music"*), explains **why** they hold those opinions, and uncovers coordinated commenting patterns.

---

## 🌟 Key Features

### 1. Dynamic Opinion Classification (Shared Analytical Engine)
- **Beyond 3-Class Sentiment:** Combines **TF-IDF Vectorization** (unigrams + bigrams) and **K-Means Clustering** to discover organic viewpoint categories directly from unlabelled text.
- **Explainable Opinion Labels:** Cluster centroids are mapped against semantic aspect taxonomies and paired with dominant sentiment polarity.
- **Minority Aggregation:** Minor opinions (< 5% volume) are consolidated into a clean *"Other Opinions"* category while remaining accessible for detailed inspection.

### 2. Dual Operating Modes
- **🎯 Narrow Mode (Single Video / Channel):**
  - In-depth inspection of a specific YouTube video discussion.
  - **Suspicious Account & Bot Analysis:** Heuristic scoring based on copy-paste text velocity, burst intervals (&le; 120s), and handle entropy.
- **🌐 Vast Mode (Multi-Topic / Hashtag Exploration):**
  - Aggregates comments across 2–3 hashtags or search terms.
  - **Cross-Video Audience Overlap Analysis:** Calculates pairwise **Jaccard Similarity** ($J(A, B) = \frac{|A \cap B|}{|A \cup B|}$) between videos and tracks recurring "super-commenters".

### 3. Respondent Deduplication
- Multiple comments from the same YouTube account are **merged into a single primary audience vote**, ensuring spammers and coordinated campaigns cannot skew public opinion consensus.

### 4. Linguistic N-Gram Tracking
- Analyzes unigrams (single words), bigrams (2-word collocations like *"waste money"*), and trigrams (3-word phrases like *"worth watching again"*), tracking their frequency and sentiment tendency.

### 5. Decoupled Full-Stack Architecture
- **Backend:** Python REST API (`server.py`) powered by standard library HTTP server + Scikit-learn + VADER.
- **Frontend:** Modern, responsive Single-Page Dashboard (`static/index.html`) using **HTML5, Tailwind CSS, Chart.js, and Lucide icons**.
- **Zero-Dependency Quickstart:** Pre-bundled with realistic offline datasets for instant demonstration without requiring a YouTube API key.

---

## 📁 Project Structure

```text
youtube_opinion_analyzer/
├── server.py                  # HTTP REST API server & static file host (Port 8000)
├── config.py                  # System hyperparameters and threshold settings
├── requirements.txt           # Core Python dependencies
├── core/
│   ├── __init__.py
│   ├── models.py              # Strongly-typed dataclasses for comments & clusters
│   ├── preprocessor.py         # Text cleaning, regex, and author deduplication
│   ├── sentiment_engine.py     # VADER sentiment scoring & polarity classification
│   ├── opinion_clustering.py   # TF-IDF vectorizer + K-Means clustering + dynamic labeling
│   ├── ngram_extractor.py      # Multi-word n-gram extractor & cross-video keyword tracker
│   ├── suspicious_accounts.py  # Narrow Mode: Bot & coordinated behavior heuristics
│   ├── audience_overlap.py     # Vast Mode: Cross-video Jaccard similarity matrix
│   ├── collector.py           # YouTube Data API v3 client + CSV dataset loader
│   └── pipeline.py            # Orchestrator linking ingestion to analytical outputs
├── static/
│   └── index.html             # Single-Page Web Dashboard (Tailwind CSS + Chart.js)
├── sample_data/
│   ├── movie_trailer_comments.csv   # Narrow Mode demo dataset (Sci-Fi Trailer)
│   └── tech_launch_comments.csv     # Vast Mode demo dataset (3 Tech Reviews)
└── tests/
    ├── __init__.py
    ├── test_preprocessor.py    # Unit tests for sanitization & deduplication
    ├── test_clustering.py      # Unit tests for clustering & opinion labels
    ├── test_ngram.py           # Unit tests for n-gram frequency extraction
    ├── test_heuristics.py      # Unit tests for bot heuristics & Jaccard overlap
    └── test_server.py          # Integration tests for REST API endpoints
```

---

## 🚀 Quickstart Guide

### 1. Launch the Application Server
Run the Python web server using your virtual environment:

```powershell
& "$HOME\.local\youtube_venv\Scripts\python.exe" server.py
```
*(Or with standard Python on PATH: `python server.py`)*

### 2. Open the Web Dashboard
Open your web browser and navigate to:
```text
http://localhost:8000/
```

- **Demo Mode:** Check the **"Use Pre-bundled Demo Dataset"** checkbox to immediately explore both Narrow Mode and Vast Mode without entering any API key.
- **Live Mode:** Uncheck demo mode and enter your Google Cloud YouTube Data API v3 Key to analyze any live YouTube video URL or hashtag in real time!

---

## 🧪 Running Automated Tests

Run all unit tests:
```powershell
$env:PYTHONPATH = "c:\Users\HP\Downloads\youtube_opinion_analyzer"
& "$HOME\.local\youtube_venv\Scripts\python.exe" -m unittest discover -s tests
```

---

## 🎓 Academic Defense / Viva Talking Points

1. **Why not just standard sentiment analysis?**
   *Coarse Positive/Negative analysis loses crucial semantic context. If 60% of reviews are negative, a creator needs to know whether the backlash is due to sound mixing, plot holes, or political messaging.*
2. **How does Author Deduplication mitigate bias?**
   *Spam bots and passionate users frequently repeat identical comments. Counting each identifiable account once per consensus distribution prevents volume astroturfing.*
3. **What is the mathematical definition of Audience Overlap?**
   *We use the Jaccard Similarity coefficient $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$, measuring the proportion of unique commenters who engaged with both videos.*
