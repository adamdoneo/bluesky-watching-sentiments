#utils.py file set up to allow testing of key pipeline functionality without loading ML models
import pandas as pd

WATCHED_SEARCH_QUERIES = [
    "just watched",
    "just finished watching",
    "finally watched",
    "finally saw",
    "just saw",
    "watched last night",
    "binge watched",
    "finished the series",
    "finished the season",
    "movie",
    "film",
    "cinema",
    "rewatching",
]

#check if test text contains any of the keywords defined in WATCHED_SEARCH_QUERIES
def matches_keyword(text):
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in WATCHED_SEARCH_QUERIES)

#aggregate and save results
def aggregate_results(linked_df):
    if len(linked_df) == 0:
        print("No results to aggregate.")
        return None
    
    #aggregate analysis
    #drop_duplicates using combination of post did, post text and title to avoid double counting 
    # identical post text by the same user about the same title
    agg_analysis = linked_df.drop_duplicates(subset=["did", "post_text", "tmdb_title"]).groupby("tmdb_title").agg(
        post_count=("post_text", "nunique"),  #count of unique posts, not rows
        #lambda functions to count case when sentiment occurs in each row
        positive_count=("sentiment", lambda x: (x == "positive").sum()),
        neutral_count=("sentiment", lambda x: (x == "neutral").sum()),
        negative_count=("sentiment", lambda x: (x == "negative").sum()),
        #"first" to get the first value of each group, since each title will have the same media type
        media_type=("media_type", "first"),
        release_date=("release_date", "first"),
    ).sort_values("post_count", ascending=False)

    print(f"\nAggregated results for {len(agg_analysis)} unique titles:")
    print(agg_analysis.head(20).to_string())
    
    return agg_analysis