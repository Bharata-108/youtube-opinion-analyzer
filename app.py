"""YouTube Sentiment & Dynamic Opinion Analysis System.

An AI-assisted NLP application for undergraduate 2nd-year CSE study and demonstration.
Features Narrow Mode, Vast Mode, Dynamic Opinion Clustering, Heuristic Suspicious
Account Detection, Cross-Video Audience Overlap Analysis, and Interactive Visualizations.
"""

import json
import io
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd

from config import (
    SAMPLE_DATA_DIR,
    DEFAULT_NUM_CLUSTERS,
    MINORITY_OPINION_THRESHOLD,
    TOP_NGRAMS_LIMIT
)
from core.models import RawComment, CleanedComment, OpinionCluster, AnalysisResult
from core.collector import YouTubeDataCollector
from core.preprocessor import TextPreprocessor, Deduplicator
from core.sentiment_engine import SentimentEngine
from core.opinion_clustering import OpinionClusterer
from core.ngram_extractor import NGramExtractor
from core.suspicious_accounts import SuspiciousAccountAnalyzer
from core.audience_overlap import AudienceOverlapAnalyzer
from visualization.charts import (
    create_opinion_distribution_donut,
    create_sentiment_bar_chart,
    create_ngram_bar_chart,
    create_suspicious_accounts_gauge,
    create_suspicious_indicators_bar,
    create_audience_overlap_heatmap
)

# Page configuration
st.set_page_config(
    page_title="YouTube Opinion & Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for academic presentation
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2E86AB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .opinion-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# PIPELINE ORCHESTRATION (SHARED ANALYTICAL ENGINE)
# ==============================================================================

def run_analysis_pipeline(
    raw_comments: List[RawComment],
    mode: str,
    source_name: str,
    n_clusters: int = DEFAULT_NUM_CLUSTERS,
    enable_suspicious: bool = True,
    enable_overlap: bool = True
) -> AnalysisResult:
    """Executes the complete end-to-end analytical pipeline across raw comments."""
    if not raw_comments:
        raise ValueError("No comments provided for analysis.")

    # 1. Text Preprocessing & Respondent Deduplication
    preprocessor = TextPreprocessor()
    cleaned_comments, dedup_stats = Deduplicator.deduplicate_respondents(raw_comments, preprocessor)

    # 2. Sentiment Scoring (VADER)
    sentiment_engine = SentimentEngine()
    enriched_comments = sentiment_engine.enrich_comments(cleaned_comments)

    # 3. Dynamic Opinion Clustering (TF-IDF + K-Means)
    # Deduplicated comments form primary opinion consensus
    primary_comments = [c for c in enriched_comments if not c.is_author_duplicate]
    if not primary_comments:
        primary_comments = enriched_comments

    clusterer = OpinionClusterer(n_clusters=n_clusters)
    _, opinion_clusters = clusterer.cluster_opinions(
        primary_comments,
        min_cluster_ratio=MINORITY_OPINION_THRESHOLD
    )

    # Map labels to all comments for complete inspection
    label_dict = {c.cluster_id: c.label for c in opinion_clusters}
    for c in enriched_comments:
        if c.opinion_cluster_id in label_dict:
            c.opinion_label = label_dict[c.opinion_cluster_id]

    # 4. Multi-Word N-Gram & Keyword Extraction
    ngram_data = NGramExtractor.extract_all_ngrams(enriched_comments, top_k=TOP_NGRAMS_LIMIT)

    # 5. Overall Sentiment Distribution
    sent_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for c in primary_comments:
        sent_counts[c.sentiment_label] = sent_counts.get(c.sentiment_label, 0) + 1

    total_primary = len(primary_comments) if primary_comments else 1
    sent_dist = {
        k: round((v / total_primary) * 100, 1) for k, v in sent_counts.items()
    }

    # 6. Specialized Module: Suspicious Accounts (Narrow Mode)
    suspicious_report = None
    if enable_suspicious:
        analyzer = SuspiciousAccountAnalyzer()
        suspicious_report = analyzer.analyze(enriched_comments)

    # 7. Specialized Module: Audience Overlap (Vast Mode)
    overlap_report = None
    if enable_overlap and mode == "Vast":
        overlap_analyzer = AudienceOverlapAnalyzer()
        overlap_report = overlap_analyzer.analyze(enriched_comments)

    return AnalysisResult(
        mode=mode,
        source_description=source_name,
        total_comments_collected=len(raw_comments),
        total_valid_comments=len(enriched_comments),
        unique_respondents=dedup_stats["unique_authors"],
        cleaned_comments=enriched_comments,
        opinion_clusters=opinion_clusters,
        ngram_frequencies=ngram_data,
        overall_sentiment_distribution=sent_dist,
        overall_sentiment_counts=sent_counts,
        suspicious_report=suspicious_report,
        audience_overlap_report=overlap_report,
        metadata=dedup_stats
    )


# ==============================================================================
# SIDEBAR NAVIGATION & CONFIGURATION
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/youtube-play.png", width=64)
    st.title("YouTube Opinion Analyzer")
    st.caption("CSE 2nd Year Capstone | Antigravity AI")

    st.markdown("---")
    mode_selection = st.radio(
        "Choose Operating Mode:",
        ["🎯 Narrow Mode (Single Video)", "🌐 Vast Mode (Multi-Topic)", "📁 CSV Dataset Import", "📖 System Documentation"],
        index=0
    )

    st.markdown("---")
    st.subheader("Data Source Settings")
    use_demo_data = st.checkbox(
        "Use Pre-bundled Demo Dataset",
        value=True,
        help="Loads realistic offline datasets without needing a Google Cloud API Key."
    )

    api_key = ""
    if not use_demo_data:
        api_key = st.text_input(
            "YouTube Data API v3 Key:",
            type="password",
            help="Enter your Google Cloud Console API Key with YouTube Data API v3 enabled."
        )

    with st.expander("⚙️ Advanced Analytical Tuning"):
        cluster_k = st.slider("Number of Opinion Clusters (k)", min_value=3, max_value=8, value=DEFAULT_NUM_CLUSTERS)
        enable_suspicious_check = st.checkbox("Enable Suspicious Account Analysis", value=True)
        enable_overlap_check = st.checkbox("Enable Audience Overlap Analysis", value=True)


# ==============================================================================
# MAIN DISPLAY LOGIC
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. NARROW MODE (SINGLE VIDEO ANALYSIS)
# ------------------------------------------------------------------------------
if mode_selection.startswith("🎯 Narrow Mode"):
    st.title("🎯 Narrow Mode: Focused Video Opinion Analysis")
    st.markdown(
        "Deeply analyze discussions for a specific YouTube video to identify **what** viewers think, "
        "quantify viewpoints, examine sentiment, and detect coordinated bot activity."
    )

    raw_comments = []
    source_name = "Movie Trailer Discussion"

    if use_demo_data:
        st.info("💡 **Active Data Source:** Pre-bundled demo dataset (`movie_trailer_comments.csv` - Sci-Fi Action Trailer).")
        sample_path = SAMPLE_DATA_DIR / "movie_trailer_comments.csv"
        if sample_path.exists():
            raw_comments = YouTubeDataCollector.load_from_csv(sample_path)
            source_name = "Chronicles of Eldoria - Official Trailer"
        else:
            st.error("Demo dataset file not found in sample_data directory.")
    else:
        video_input = st.text_input(
            "Enter YouTube Video URL or Video ID:",
            placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        col_btn, _ = st.columns([1, 4])
        if col_btn.button("Fetch & Analyze Video", type="primary"):
            if not api_key:
                st.error("Please enter a valid YouTube Data API Key in the sidebar or toggle 'Use Demo Dataset'.")
            elif not video_input:
                st.warning("Please provide a valid YouTube URL or Video ID.")
            else:
                collector = YouTubeDataCollector(api_key=api_key)
                vid_id = collector.extract_video_id(video_input)
                if not vid_id:
                    st.error("Could not parse a valid YouTube Video ID from the provided URL.")
                else:
                    with st.spinner(f"Ingesting live comments for video {vid_id}..."):
                        try:
                            raw_comments = collector.fetch_narrow_comments(vid_id, max_comments=300)
                            source_name = f"Live Video ({vid_id})"
                        except Exception as e:
                            st.error(f"Error fetching comments: {str(e)}")

    if raw_comments:
        with st.spinner("Executing Shared Analytical Engine (Deduplication, Clustering, Sentiment, N-Grams)..."):
            result = run_analysis_pipeline(
                raw_comments=raw_comments,
                mode="Narrow",
                source_name=source_name,
                n_clusters=cluster_k,
                enable_suspicious=enable_suspicious_check,
                enable_overlap=False
            )

        # Overview Metrics
        st.markdown(f"### 📊 Analysis Summary: *{result.source_description}*")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Comments", result.total_comments_collected)
        m2.metric("Unique Viewers", result.unique_respondents)
        m3.metric("Spam / Duplicates Removed", result.metadata.get("duplicate_author_comments", 0))
        dominant_op = result.opinion_clusters[0].label if result.opinion_clusters else "N/A"
        m4.metric("Leading Viewpoint", dominant_op[:20] + "...")
        m5.metric("Net Sentiment Ratio", f"{result.overall_sentiment_distribution.get('Positive', 0)}% Pos")

        st.markdown("---")

        # Tabs for Analytical Deep-Dive
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💡 Dynamic Opinions",
            "📈 Sentiment Analysis",
            "🔤 Recurring N-Grams",
            "🛡️ Suspicious Accounts",
            "📋 Data & Export"
        ])

        with tab1:
            st.subheader("Dynamic Opinion Classification")
            st.markdown(
                "Unlike standard 3-class sentiment, the system groups semantically similar comments "
                "into **meaningful categories** based on what viewers are actually discussing."
            )

            col_donut, col_details = st.columns([1, 1])

            with col_donut:
                fig_donut = create_opinion_distribution_donut(result.opinion_clusters)
                st.plotly_chart(fig_donut, use_container_width=True)

            with col_details:
                st.markdown("#### Identified Viewpoints Breakdown")
                for cluster in result.opinion_clusters:
                    with st.expander(f"**{cluster.label}** — {cluster.percentage}% ({cluster.respondent_count} respondents)"):
                        st.markdown(f"**Dominant Sentiment:** `{cluster.dominant_sentiment}`")
                        st.markdown(f"**Top Distinguishing Keywords:** *{', '.join(cluster.top_keywords)}*")
                        st.markdown("**Representative Comments:**")
                        for quote in cluster.representative_samples:
                            st.caption(f"💬 \"{quote}\"")

        with tab2:
            st.subheader("Statistical Sentiment Distribution")
            col_bar, col_pie = st.columns([1, 1])
            with col_bar:
                fig_sent = create_sentiment_bar_chart(result.overall_sentiment_counts)
                st.plotly_chart(fig_sent, use_container_width=True)
            with col_pie:
                st.markdown("#### Sentiment vs. Viewpoint Insights")
                st.markdown(
                    f"- **Positive Discussions:** {result.overall_sentiment_distribution.get('Positive', 0)}%\n"
                    f"- **Neutral / Informational:** {result.overall_sentiment_distribution.get('Neutral', 0)}%\n"
                    f"- **Critical / Negative:** {result.overall_sentiment_distribution.get('Negative', 0)}%\n"
                )
                st.info(
                    "💡 **Respondent Deduplication Rule Applied:** Each identifiable YouTube account is counted "
                    "exactly once in this consensus distribution, preventing multi-posting spammers from skewing public perception."
                )

        with tab3:
            st.subheader("Multi-Word Phrase & Keyword Tracking")
            st.markdown("Analyzes n-grams to identify significant recurring single words and collocations.")

            ngram_type = st.radio("Select N-Gram Order:", ["Unigrams (Single Words)", "Bigrams (2-Word Phrases)", "Trigrams (3-Word Expressions)"], horizontal=True)

            key = "unigrams" if "Unigrams" in ngram_type else "bigrams" if "Bigrams" in ngram_type else "trigrams"
            ngram_list = result.ngram_frequencies.get(key, [])

            col_nchart, col_ntable = st.columns([3, 2])
            with col_nchart:
                fig_ng = create_ngram_bar_chart(ngram_list, title=f"Frequently Occurring {ngram_type}")
                st.plotly_chart(fig_ng, use_container_width=True)
            with col_ntable:
                st.markdown("#### Frequency Table")
                df_ng = pd.DataFrame(ngram_list)
                if not df_ng.empty:
                    st.dataframe(df_ng[["phrase", "count", "sentiment_category"]], use_container_width=True, height=350)

        with tab4:
            st.subheader("🛡️ Suspicious Account & Coordinated Activity Analysis")
            st.markdown(
                "Examines observable commenting behavior (text copy-paste velocity, burst intervals, "
                "and handle entropy) to identify characteristics associated with automated or coordinated campaigns."
            )
            if result.suspicious_report:
                rep = result.suspicious_report
                col_g, col_ind = st.columns([1, 1])
                with col_g:
                    fig_gauge = create_suspicious_accounts_gauge(rep.suspicious_percentage)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    st.caption(f"Flagged **{rep.suspicious_accounts_count}** out of **{rep.total_accounts}** accounts as potentially suspicious.")
                with col_ind:
                    fig_ind = create_suspicious_indicators_bar(rep.behavioral_indicators)
                    st.plotly_chart(fig_ind, use_container_width=True)

                if rep.suspicious_accounts:
                    st.markdown("#### Flagged Account Activity Details")
                    for s in rep.suspicious_accounts[:5]:
                        with st.expander(f"⚠️ Account: {s.author_name} (Suspicion Index: {s.suspicion_score}%)"):
                            st.write(f"**Reasons:** {', '.join(s.reasons)}")
                            st.write(f"**Comments Posted:** {s.comment_count}")
                            st.write(f"**Associated Viewpoints:** {', '.join(s.associated_opinions) if s.associated_opinions else 'Spam'}")
                            st.caption(f"Sample Comment: \"{s.sample_text}\"")
            else:
                st.info("Suspicious account module was disabled in the sidebar settings.")

        with tab5:
            st.subheader("Data Explorer & Export Center")
            table_data = []
            for c in result.cleaned_comments:
                table_data.append({
                    "Author": c.raw.author_name,
                    "Comment": c.cleaned_text,
                    "Assigned Opinion": c.opinion_label,
                    "Sentiment": c.sentiment_label,
                    "Score": c.sentiment_score,
                    "Likes": c.raw.like_count,
                    "Is Duplicate Author": c.is_author_duplicate
                })
            df_export = pd.DataFrame(table_data)
            st.dataframe(df_export, use_container_width=True, height=350)

            col_d1, col_d2 = st.columns(2)
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False)
            col_d1.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"opinion_analysis_{result.mode.lower()}.csv",
                mime="text/csv"
            )

            # JSON export
            json_payload = {
                "source": result.source_description,
                "total_comments": result.total_comments_collected,
                "unique_respondents": result.unique_respondents,
                "sentiment_distribution": result.overall_sentiment_distribution,
                "opinion_clusters": [
                    {
                        "label": cl.label,
                        "percentage": cl.percentage,
                        "respondent_count": cl.respondent_count,
                        "dominant_sentiment": cl.dominant_sentiment,
                        "top_keywords": cl.top_keywords
                    }
                    for cl in result.opinion_clusters
                ]
            }
            col_d2.download_button(
                label="📥 Download Analysis Report (JSON)",
                data=json.dumps(json_payload, indent=2),
                file_name="opinion_report.json",
                mime="application/json"
            )


# ------------------------------------------------------------------------------
# 2. VAST MODE (MULTI-TOPIC / HASHTAG EXPLORATION)
# ------------------------------------------------------------------------------
elif mode_selection.startswith("🌐 Vast Mode"):
    st.title("🌐 Vast Mode: Multi-Source & Hashtag Opinion Exploration")
    st.markdown(
        "Broadly analyze discussions across multiple trending videos surrounding 2–3 hashtags or search terms. "
        "Includes **Cross-Video Audience Overlap Analysis** to evaluate recurring commentators and audience similarity."
    )

    raw_vast_comments = []
    source_name = "Multi-Source Tech Discussion"

    if use_demo_data:
        st.info("💡 **Active Data Source:** Pre-bundled Vast Mode dataset (`tech_launch_comments.csv` covering 3 distinct videos).")
        sample_path = SAMPLE_DATA_DIR / "tech_launch_comments.csv"
        if sample_path.exists():
            raw_vast_comments = YouTubeDataCollector.load_from_csv(sample_path)
            source_name = "Tech Launch & Flagship Comparisons"
        else:
            st.error("Demo dataset file not found in sample_data directory.")
    else:
        tags_input = st.text_input(
            "Enter 2-3 Search Queries or Hashtags (comma-separated):",
            placeholder="#UltraPhonePro, #FlagshipReview, #Smartphone2026"
        )
        if st.button("Search & Analyze Multi-Video Discussions", type="primary"):
            if not api_key:
                st.error("Please enter a YouTube Data API Key in the sidebar or toggle 'Use Demo Dataset'.")
            elif not tags_input:
                st.warning("Please provide search queries.")
            else:
                queries = [q.strip() for q in tags_input.split(",") if q.strip()]
                collector = YouTubeDataCollector(api_key=api_key)
                with st.spinner("Discovering engaging videos and aggregating comments across topics..."):
                    try:
                        raw_vast_comments = collector.fetch_vast_comments(queries, videos_per_query=2, comments_per_video=75)
                        source_name = ", ".join(queries)
                    except Exception as e:
                        st.error(f"Error fetching vast comments: {str(e)}")

    if raw_vast_comments:
        with st.spinner("Aggregating multi-source dataset and computing cross-video matrices..."):
            vast_result = run_analysis_pipeline(
                raw_comments=raw_vast_comments,
                mode="Vast",
                source_name=source_name,
                n_clusters=cluster_k,
                enable_suspicious=enable_suspicious_check,
                enable_overlap=enable_overlap_check
            )

        # Overview Metrics
        st.markdown(f"### 🌐 Aggregated Discussion Analysis: *{vast_result.source_description}*")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Total Collected Comments", vast_result.total_comments_collected)
        v2.metric("Unique Identified Accounts", vast_result.unique_respondents)
        if vast_result.audience_overlap_report:
            rep = vast_result.audience_overlap_report
            v3.metric("Multi-Video Commenters", f"{rep.multi_video_accounts_count} ({rep.multi_video_percentage}%)")
            v4.metric("Analyzed Sources", len(rep.video_titles))

        st.markdown("---")

        vtab1, vtab2, vtab3, vtab4 = st.tabs([
            "💡 Consolidated Opinions",
            "🔄 Audience Overlap Matrix",
            "👥 Cross-Participating Commenters",
            "🔤 Cross-Topic Keywords"
        ])

        with vtab1:
            st.subheader("Aggregated Dynamic Opinion Distribution")
            c_donut, c_sent = st.columns([1, 1])
            with c_donut:
                fig_v_donut = create_opinion_distribution_donut(vast_result.opinion_clusters)
                st.plotly_chart(fig_v_donut, use_container_width=True)
            with c_sent:
                fig_v_sent = create_sentiment_bar_chart(vast_result.overall_sentiment_counts)
                st.plotly_chart(fig_v_sent, use_container_width=True)

        with vtab2:
            st.subheader("Cross-Video Audience Overlap (Jaccard Similarity)")
            st.markdown(
                "Calculates set intersection of unique commenter IDs between distinct YouTube videos: "
                r"$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$"
            )
            if vast_result.audience_overlap_report:
                fig_hm = create_audience_overlap_heatmap(vast_result.audience_overlap_report)
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Audience overlap was not computed.")

        with vtab3:
            st.subheader("Accounts Appearing Across Multiple Discussions")
            st.markdown("Identifies active respondents who participate in multiple videos across the searched topics.")
            if vast_result.audience_overlap_report and vast_result.audience_overlap_report.cross_participating_accounts:
                cross_accounts = vast_result.audience_overlap_report.cross_participating_accounts
                for acc in cross_accounts:
                    with st.expander(f"👤 {acc['author_name']} — Active in {acc['video_count']} Videos ({acc['total_comments']} Comments)"):
                        st.write(f"**Videos Participated:**")
                        for vid in acc["videos_participated"]:
                            st.markdown(f"- *{vid}*")
                        st.write(f"**Expressed Sentiments:** {acc['sentiments']}")
            else:
                st.info("No cross-video overlap detected among commenters in this dataset.")

        with vtab4:
            st.subheader("Keywords Spanning Multiple Videos")
            st.markdown("Identifies high-impact terms appearing across distinct video discussions.")
            cross_kw = NGramExtractor.extract_cross_video_keywords(vast_result.cleaned_comments)
            if cross_kw:
                df_ckw = pd.DataFrame(cross_kw)
                st.dataframe(df_ckw, use_container_width=True)
            else:
                st.info("No cross-video keywords discovered.")


# ------------------------------------------------------------------------------
# 3. CSV DATASET IMPORT (CUSTOM OFFLINE UPLOADS)
# ------------------------------------------------------------------------------
elif mode_selection.startswith("📁 CSV Dataset Import"):
    st.title("📁 Offline Dataset Import & Custom Analysis")
    st.markdown(
        "Upload your own CSV comment exports (e.g. Kaggle datasets, exported YouTube scraper files). "
        "The system auto-detects column names (`comment`, `text`, `author`, `likes`, `published_at`)."
    )

    uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])
    if uploaded_file:
        try:
            custom_comments = YouTubeDataCollector.load_from_csv(uploaded_file)
            st.success(f"Successfully loaded {len(custom_comments)} comments from `{uploaded_file.name}`.")

            analysis_mode = st.radio("Interpret dataset as:", ["Narrow Mode (Single Discussion)", "Vast Mode (Multi-Topic)"], horizontal=True)
            mode_tag = "Narrow" if "Narrow" in analysis_mode else "Vast"

            if st.button("Run Full NLP & Opinion Analysis", type="primary"):
                with st.spinner("Processing custom dataset..."):
                    custom_res = run_analysis_pipeline(
                        raw_comments=custom_comments,
                        mode=mode_tag,
                        source_name=uploaded_file.name,
                        n_clusters=cluster_k,
                        enable_suspicious=enable_suspicious_check,
                        enable_overlap=(mode_tag == "Vast")
                    )

                st.markdown("### Analysis Results")
                fig1 = create_opinion_distribution_donut(custom_res.opinion_clusters)
                st.plotly_chart(fig1, use_container_width=True)

                fig2 = create_sentiment_bar_chart(custom_res.overall_sentiment_counts)
                st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to process CSV file: {str(e)}")


# ------------------------------------------------------------------------------
# 4. SYSTEM DOCUMENTATION & ACADEMIC VIVA GUIDE
# ------------------------------------------------------------------------------
else:
    st.title("📖 System Architecture & Academic Guide")
    st.markdown("""
    ### Project Overview
    **System Title:** YouTube Sentiment & Dynamic Opinion Analysis System  
    **Academic Level:** Undergraduate, 2nd-Year Computer Science & Engineering (CSE)  
    **Core Objective:** Discover *what* viewers think and *why*, rather than stopping at raw Positive/Negative sentiment.

    ---

    ### 1. Mathematical & Algorithmic Foundations

    #### A. Dynamic Opinion Clustering
    1. **TF-IDF (Term Frequency - Inverse Document Frequency):**
       $$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\log\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1$$
       Transforms cleaned natural language into continuous vector space while down-weighting ubiquitous non-discriminative terms.

    2. **K-Means Objective:**
       $$\\arg\\min_{\\mathbf{S}} \\sum_{i=1}^{k} \\sum_{\\mathbf{x} \\in S_i} \\|\\mathbf{x} - \\boldsymbol{\\mu}_i\\|^2$$
       Partitions comments into $k$ spherical clusters by minimizing within-cluster variance.

    3. **Automated Opinion Labeling:**
       Centroids $\\boldsymbol{\\mu}_i$ are inspected to extract top discriminative keywords, matched against semantic aspect taxonomies (Cinematography, Storyline, Performances, Politics, Value, Audio), and combined with the cluster's dominant sentiment polarity to generate intuitive labels (e.g. *"Praise for Cinematography"*).

    #### B. Respondent Deduplication
    To avoid astroturfing and spam bias, the system applies **Author-Level Deduplication**:
    - Each unique `author_id` is assigned exactly one primary vote in the consensus distribution.
    - Prevents single highly-active commenters from skewing population statistics.

    #### C. Audience Overlap Analysis (Vast Mode)
    Calculates pairwise Jaccard set similarity across videos $A$ and $B$:
    $$J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}$$

    #### D. Suspicious Account Detection (Narrow Mode)
    Heuristics evaluate:
    - Text duplication velocity ($J_{\\text{text}} \\ge 0.85$).
    - Temporal burst frequency ($\\Delta t \\le 120$ seconds).
    - Handle entropy and repetitive username syntax.
    """)
