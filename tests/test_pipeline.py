from utils import matches_keyword, aggregate_results
import pandas as pd


def test_keyword_matching():
    assert matches_keyword("just watched spider-man, it was a good movie") == True
    assert matches_keyword("does anyone know where to buy a good burger in London?") == False

#this tests the aggregate_results function by creating a small DataFrame with sample data and 
# checking the aggregated counts for each title
def test_aggregate_counts_by_title():
    df = pd.DataFrame([
        {"did": "user_1", "post_text": "Severance season 2 was incredible", "tmdb_title": "Severance",
         "sentiment": "positive", "media_type": "tv", "release_date": "2022-02-18"},
        {"did": "user_2", "post_text": "hated the new season of Severance", "tmdb_title": "Severance",
         "sentiment": "negative", "media_type": "tv", "release_date": "2022-02-18"},
        {"did": "user_3", "post_text": "The Bear is best show ever", "tmdb_title": "The Bear",
         "sentiment": "positive", "media_type": "tv", "release_date": "2022-06-23"},
    ])
    result = aggregate_results(df)
    assert result.loc["Severance", "post_count"] == 2
    assert result.loc["Severance", "positive_count"] == 1
    assert result.loc["Severance", "negative_count"] == 1
    assert result.loc["The Bear", "post_count"] == 1