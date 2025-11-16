import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pipeline.feature_sentiment import analyze_sentiment

# -------------------------------
# Streamlit App Config
# -------------------------------
st.set_page_config(
    page_title="Feature-Specific Sentiment Analysis",
    layout="wide"
)

st.title("Feature-Specific Sentiment Analysis Tool")
st.write(
    """
Upload a cleaned reviews CSV (from previous sentiment step) or use the default processed file to analyze feature-specific sentiment.
"""
)

# -------------------------------
# Default CSV (optional)
# -------------------------------
DEFAULT_CSV = "data/processed/noom_google_clean.csv"

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Upload your cleaned reviews CSV", type=["csv"]
)

# Load the CSV: uploaded file > default file
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} reviews from uploaded CSV.")
elif st.button("Use default processed CSV"):
    try:
        df = pd.read_csv(DEFAULT_CSV)
        st.write(f"Loaded {len(df)} reviews from default CSV.")
    except FileNotFoundError:
        st.error(f"Default CSV not found at {DEFAULT_CSV}. Please upload a file.")
        df = None
else:
    df = None
    st.info("Please upload a CSV or click 'Use default processed CSV'.")

# -------------------------------
# Helper: Plot Confidence Histogram
# -------------------------------
def plot_confidence_histogram(sent_df):
    """Plots the confidence histogram in Streamlit."""
    if sent_df is None or sent_df.empty:
        st.warning("No sentiment data available to plot confidence histogram.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sent_df["confidence"].hist(bins=20, edgecolor='black', ax=ax)
    ax.set_title('Sentiment Model Confidence Distribution', fontsize=16)
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Number of Clauses', fontsize=12)
    ax.grid(axis='y')
    st.pyplot(fig)

# -------------------------------
# Run Analysis Button
# -------------------------------
if df is not None:
    if st.button("Run Feature Specific Analysis"):
        with st.spinner("Running sentiment analysis... This may take a few minutes..."):
            result_df = analyze_sentiment(df, text_column="cleaned_content")
        
        if result_df is not None and not result_df.empty:
            st.success("Sentiment analysis complete!")

            # --- Confidence Histogram ---
            st.subheader("Model Confidence Distribution")
            plot_confidence_histogram(result_df)

            # --- Feature-Level Summary Table ---
            st.subheader("Feature-Level Sentiment Summary")
            sentiment_summary = result_df.groupby('bucket')['label'].value_counts().unstack(fill_value=0)
            st.dataframe(sentiment_summary)

            # --- Horizontal Bar Chart for Sentiment Scores ---
            st.subheader("Feature Sentiment Scores")
            # Compute sentiment scores
            sentiment_summary['total'] = sentiment_summary.get('POSITIVE', 0) + sentiment_summary.get('NEGATIVE', 0)
            sentiment_summary['positive_rate'] = sentiment_summary.get('POSITIVE', 0) / sentiment_summary['total']
            sentiment_summary['negative_rate'] = sentiment_summary.get('NEGATIVE', 0) / sentiment_summary['total']
            sentiment_summary['sentiment_score'] = sentiment_summary['positive_rate'] - sentiment_summary['negative_rate']
            sentiment_summary = sentiment_summary.sort_values(by='sentiment_score', ascending=True)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#4CAF50' if x >= 0 else '#F44336' for x in sentiment_summary['sentiment_score']]
            sentiment_summary['sentiment_score'].plot(kind='barh', color=colors, ax=ax, width=0.8)
            ax.set_xlabel("Sentiment Score (Positive Rate - Negative Rate)")
            ax.set_ylabel("Feature Bucket")
            ax.axvline(0, color='grey', linestyle='--', linewidth=1)
            ax.grid(axis='x', linestyle=':', alpha=0.7)
            st.pyplot(fig)
        else:
            st.warning("No relevant clauses were found with the current keywords.")
