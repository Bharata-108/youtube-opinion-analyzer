"""Configuration settings and hyperparameter defaults for YouTube Opinion Analysis System.

Designed for 2nd-Year CSE academic demonstration and production-grade extensibility.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

# YouTube Server-Side API Key (Configured via environment variable on cloud hosts)
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

# Sentiment Analysis Configuration (VADER Thresholds)
SENTIMENT_POSITIVE_THRESHOLD = 0.05
SENTIMENT_NEGATIVE_THRESHOLD = -0.05

# Dynamic Opinion Clustering Configuration
DEFAULT_MIN_CLUSTERS = 3
DEFAULT_MAX_CLUSTERS = 7
DEFAULT_NUM_CLUSTERS = 5
TFIDF_MAX_FEATURES = 1500
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.85
MINORITY_OPINION_THRESHOLD = 0.05  # Clusters with < 5% volume grouped under "Other Opinions"

# N-Gram Tracking Configuration
TOP_NGRAMS_LIMIT = 15
STOPWORDS_LANGUAGE = "english"

# Suspicious Account Heuristic Thresholds (Narrow Mode)
TEXT_SIMILARITY_DUPLICATION_THRESHOLD = 0.85  # Jaccard / Levenshtein similarity for copy-paste detection
BURST_TIME_WINDOW_SECONDS = 120              # Window to flag rapid-fire commenting
HIGH_COMMENT_FREQUENCY_PER_AUTHOR = 3        # Flag authors posting > 3 times in a single source
SUSPICION_SCORE_HIGH = 70.0
SUSPICION_SCORE_MEDIUM = 40.0

# Vast Mode Configuration
DEFAULT_VAST_VIDEOS_PER_QUERY = 3
MAX_VAST_VIDEOS_TOTAL = 10

# YouTube API Configuration Defaults
DEFAULT_MAX_COMMENTS = 300
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
