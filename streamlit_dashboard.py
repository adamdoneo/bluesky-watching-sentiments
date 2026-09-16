# basic streamlit dashboard to showcase output of  movie/TV bluesky sentiment pipeline
# this reads the pipeline output from the latest .CSVs produced every 3 hours
# code adapted from https://docs.streamlit.io/get-started/tutorials/create-an-app and developed using Claude Code AI

import streamlit as st
import pandas as pd
import os
import plotly.express as px
from azure.storage.blob import BlobServiceClient
import io
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="3-hourly Bluesky Sentiment Dashboard", layout="wide")
st.title("3-hourly Bluesky Movie & TV-Show Sentiment Dashboard")

#define filepaths to pipeline outputs
SUMMARY_FILE = "data/summary_latest.csv"
SIGHTINGS_FILE = "data/sightings_latest.csv"

#using @st.cache_data decorator to cache the CSV once and force streamlit not to re-load the dataset after each interaction
@st.cache_data(ttl=900)  # cache for 15 minutes, meaning the data will be reloaded from source every 15 minutes
def load_data():
    conn_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn_string:
        blob_service = BlobServiceClient.from_connection_string(conn_string)
        container = blob_service.get_container_client("sightings")
        try:
            summary_blob = container.download_blob("summary_latest.csv").readall()
            sightings_blob = container.download_blob("sightings_latest.csv").readall()
            summary = pd.read_csv(io.StringIO(summary_blob.decode("utf-8")), index_col=0)
            sightings = pd.read_csv(io.StringIO(sightings_blob.decode("utf-8")))
            return summary, sightings, "Azure Blob storage"
        except Exception as e:
            st.error(f"Failed to load from Azure: {e}")
            return None, None, None
    else:
        # fall back to local files
        if not os.path.exists(SUMMARY_FILE) or not os.path.exists(SIGHTINGS_FILE):
            return None, None
        summary = pd.read_csv(SUMMARY_FILE, index_col=0)
        sightings = pd.read_csv(SIGHTINGS_FILE)
        return summary, sightings, "local file system"

# loading state text
data_load_state = st.text("Loading data...")
summary_df, sightings_df, data_source = load_data()

if summary_df is None:
    st.error("No pipeline data found. Run pipeline.py first.")
    st.stop()

data_load_state.empty()

#define the last_updated timestamp by using the latest timestamp of a post. convert UTC to BST
last_updated = (
    pd.to_datetime(sightings_df["created_at"].max())
    .tz_convert("Europe/London")
    .strftime("%d %B %Y, %H:%M %Z")
)

st.caption(f"Most recent post: {last_updated}")

#warning message
st.warning(
    "This dashboard displays raw posts directly from Bluesky. "
    "Posts could contain strong language, movie/tv-show spoilers, and/or opinions that do not reflect those of the developer.", 
    icon="🚨",
    title="Warning"
)

st.info(
    "This dashboard presents what English-language users on Bluesky have been watching, alongside their general sentiment on the content they watched. "
    "Posts are collected from Bluesky Jetstream in 3-hourly batches, pre-filtered using a broad set of keywords, classified as containing a viewing reaction using a fine-tuned [Twitter-roBERTa-base model](https://huggingface.co/cardiffnlp/twitter-roberta-base) trained on 433 hand-labeled posts (fine-tuned model can be found [here](https://huggingface.co/adamdoneo/viewing-reaction-classifier)), "
    "linked to film and TV titles on [TheMovieDatabase API](https://developer.themoviedb.org/docs/getting-started) via the [SpaCy Named-Entity Recognition model](https://spacy.io/models/en#en_core_web_trf), and scored for sentiment "
    "via the [Twitter-roBERTa-base for Sentiment Analysis model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest). Data is automatically refreshed every 3 hours."
    "\n\nPlease note that this project is a work in progress and may not be fully functional or accurate. Known issues include a small training set, off-the-shelf NER and sentiment models, "
    "no consideration for reply posts (posts are looked at individually), and posts with timestamps that may not match the 3-hour data collection window because post timestamps are defined client-side and can differ from when Bluesky Jetstream processes them. "
    "Most notably, movies and TV shows with common words in their titles, or those that are part of franchises with multiple titles, may be misclassified. Also note that the pipeline is capped at processing up to 20,000 posts to save resources.\n\n"
    
    "\n\nBuilt by [Adam Doneo](https://github.com/adamdoneo) | See the "
    "[source code](https://github.com/adamdoneo/bluesky-watching-sentiments) for more information | "
    "Think this is cool and want to connect? Reach out to me via my [LinkedIn](https://www.linkedin.com/in/adam-doneo/)",
    icon="ℹ️",
)

#plot the top 10 titles, by num of mentions, sorted from most to fewest, using plotly
#bars further segmented showing positie, neutral, negative sentiment counts for each title
st.subheader("Top 10 Trending Titles")
top_titles = summary_df.head(10).reset_index()
#truncate long titles to 25 characters for better display in the bar chart
top_titles["tmdb_title"] = top_titles["tmdb_title"].apply(lambda x: x[:25] + "..." if len(x) > 25 else x)
#rename the positive_count, neutral_count, negative_count columns to read better in the legend
top_titles.rename(columns={
    "positive_count": "Positive", 
    "neutral_count": "Neutral", 
    "negative_count": "Negative"}, 
    inplace=True)
fig = px.bar(
    top_titles,
    x="tmdb_title",
    y=["Positive", "Neutral", "Negative"],
    labels={"tmdb_title": "Movie/TV-Show title", 
            "value": "Count of Posts mentioning Title", 
            "variable": "Sentiment"}, 
    color_discrete_map={
        "Positive": "#27ae60", 
        "Neutral": "#bdc3c7", 
        "Negative": "#c0392b"},
)
#update hovertemplate to show title and count of posts for each sentiment
fig.update_traces(
    hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y} posts<extra></extra>"
)
fig.update_xaxes(
    categoryorder="total descending",
    title_font=dict(size=14, weight="bold"),
    tickangle=-45,
    tickfont=dict(size=12),
)
fig.update_yaxes(title_font=dict(size=14, weight="bold"))
fig.update_layout(
    height=550,
    margin=dict(b=180),
    legend_title_font=dict(size=14, weight="bold"),
    legend_font=dict(size=12, weight="bold"),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.55,
        xanchor="center",
        x=0.5,
    ),
)
st.plotly_chart(fig, use_container_width=True)

#dropdown to select title, to reveal all posts and inferred info for that post
st.subheader("Sentiment by Title")
selected = st.selectbox("Select a title", summary_df.index.tolist())

row = summary_df.loc[selected]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mentions", int(row["post_count"]))
col2.metric("Positive", int(row["positive_count"]))
col3.metric("Neutral", int(row["neutral_count"]))
col4.metric("Negative", int(row["negative_count"]))

st.caption(f"{row['media_type']} | Released: {row['release_date']}")

#posts for selected title (sorted by most recent first)
st.subheader(f"Posts mentioning {selected}")
title_posts = sightings_df[sightings_df["tmdb_title"] == selected].sort_values("created_at", ascending=False)

for idx, post in title_posts.iterrows():
    sentiment = post["sentiment"]
    sentiment_probability = post["sentiment_probability"]
    classifier_probability = post["classifier_probability"]
    icon = {"positive": "✅", "neutral": "😐", "negative": "❌"}.get(sentiment, "")
    post_timestamp = (
        pd.to_datetime(post["created_at"])
        .tz_convert("Europe/London")
        .strftime("%d %B %Y, %H:%M %Z")
    )

    st.markdown(
        f"**{icon} {sentiment}** (sentiment model confidence P={sentiment_probability:.2f}, "
        f"viewing reaction classifier confidence P={classifier_probability:.2f})  \n"
        f"{post_timestamp}\n\n"
        f"{post['post_text']}"
    )