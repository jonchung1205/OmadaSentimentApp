import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pipeline.feature_sentiment import analyze_sentiment

import sys
import os

# Add project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from pipeline.feature_sentiment import analyze_sentiment


st.title("Feature-Specific Sentiment Analysis")

DEFAULT_CSV = "data/processed/noom_google_clean.csv"

uploaded_file = st.file_uploader("Upload cleaned reviews CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} reviews from uploaded CSV.")
elif st.button("Use default processed CSV"):
    try:
        df = pd.read_csv(DEFAULT_CSV)
        st.write(f"Loaded {len(df)} reviews from default CSV.")
    except FileNotFoundError:
        st.error(f"Default CSV not found: {DEFAULT_CSV}")
        df = None
else:
    df = None
    st.info("Upload a CSV or use the default file.")

def plot_confidence_histogram(sent_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sent_df["confidence"].hist(bins=20, edgecolor='black', ax=ax)
    ax.set_title("Model Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    st.pyplot(fig)

if df is not None:
    if st.button("Run Feature Specific Analysis"):
        with st.spinner("Running sentiment analysis..."):
            result_df, summary_df = analyze_sentiment(df, text_column="cleaned_content")

        st.success("Analysis complete!")

        st.subheader("Confidence Distribution")
        plot_confidence_histogram(result_df)

        st.subheader("Feature-Level Sentiment Summary")

        summary = result_df.groupby("feature")["sentiment"].value_counts().unstack(fill_value=0)
        st.dataframe(summary)

        st.subheader("Sentiment Scores by Feature")
        summary["score"] = (
            summary.get("POSITIVE", 0) - summary.get("NEGATIVE", 0)
        ) / (summary.get("POSITIVE", 0) + summary.get("NEGATIVE", 0))

        summary = summary.sort_values("score")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#4CAF50" if x >= 0 else "#F44336" for x in summary["score"]]
        summary["score"].plot(kind="barh", ax=ax, color=colors)
        ax.set_xlabel("Sentiment Score")
        st.pyplot(fig)
