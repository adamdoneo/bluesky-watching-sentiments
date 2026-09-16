import websockets
import json
import time
import os
import asyncio
import pandas as pd
import spacy
import httpx
from datetime import datetime, timezone
from transformers import pipeline as hf_pipeline  # renamed due to clash with variable name
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from utils import WATCHED_SEARCH_QUERIES, matches_keyword, aggregate_results

#Movie/tv show search terms

# API docs: https://developer.themoviedb.org/reference/search-multi
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/multi"

# Jetstream config
#performing server-side filtering using "&kinds=commit" to only receive posts
JETSTREAM_BASE = (
    "wss://jetstream.us-east.bsky.network/subscribe"
    "?wantedCollections=app.bsky.feed.post"
    "&kinds=commit"
)
#replay the past 3 hours
REPLAY_HOURS = 3
#end jetstream post collection at 20k posts
MAX_POSTS = 20000
#timeout to prevent searching indefinitely
TIMEOUT = 300

#setting loading of models in a function to call it in Main, and to allow for 
# testing of the pipeline.py file without loading models each time
def load_models():
    print("Loading models")

    #load fine-tuned viewing reaction classifier from hugginface repo
    #trained on 433 gold-labelled bluesky posts using cardiffnlp/twitter-roberta-base
    #see text_classifier_fine_tuning.ipynb for training
    classifier = hf_pipeline(
        "text-classification",
        model="adamdoneo/viewing-reaction-classifier",
        tokenizer="adamdoneo/viewing-reaction-classifier",
        device=-1,
    )
    print("Fine-tuned Classifier loaded")

    #load SpaCy NER model (using the transformer model, for more info see https://spacy.io/models/en)
    nlp = spacy.load("en_core_web_trf")
    print(f"SpaCy model loaded: {nlp.meta['name']}")

    #load sentiment model (using twitter-roberta-base-sentiment-latest)
    sentiment_model = hf_pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1,
    )
    print("Sentiment model loaded")
    print("All models loaded.\n")

    return classifier, nlp, sentiment_model


#helper functions

#helper function to query the movie database API using a WORK_OF_ART string
#using httpx instead of requests library with an eye on using async via httpx down the line
def search_tmdb(WORK_OF_ART):

    TMDB_API_KEY = os.getenv("TMDB_READ_ACCESS_TOKEN")
    #search the movie database using WORK_OF_ART, returns the most popular matching movie/tv show or nothing
    headers = {
        "Authorization": f"Bearer {TMDB_API_KEY}",
        "accept": "application/json",
    }
    #send GET request to API, with a 10s timeout before raising error
    resp = httpx.get(TMDB_SEARCH_URL, headers=headers, params={"query": WORK_OF_ART}, timeout=10)
    #raise error code
    resp.raise_for_status()

    #filter the API response on movie and tv show - the search/multi endpoint also includes people, but this isnt relevant 
    results = [r for r in resp.json().get("results", []) if r.get("media_type") in ("movie", "tv")]
    if not results:
        return None

    #select the top result based on popularity (this is defined by themoviedb here:https://starlight.mintlify.app/api-reference/popularandtrending)
    top = results[0]

    #return result as dictionary
    return {
        "tmdb_id": top.get("id"),
        #get title or name, depending on movie or tv show
        "tmdb_title": top.get("title") or top.get("name", ""),
        "media_type": top.get("media_type"),
        #get release date or first air date depending on movie or tv show
        "release_date": top.get("release_date") or top.get("first_air_date", ""),
        #get average vote data for some extra context, might use it later
        "vote_average": top.get("vote_average", 0),
    }


#helper function to link NER WORK_OF_ART results against the movie database (using a personal API TMDB_READ_ACCESS_TOKEN set in .env file)
def extract_entity_and_link(text, nlp_model, entity_type):
    doc = nlp_model(text)
    #loop through every entity that the NER model found, keep only WORK_OF_ART entities, 
    # and extract their text. use set to remove duplicates, and convert back to list.
    #using ent.label_ to get human readable label
    works = list(set(ent.text for ent in doc.ents if ent.label_ == entity_type))

    linked = []
    for title in works:
        try:
            result = search_tmdb(title)
        except Exception as e:
            print(f"Error occurred while searching for {title}: {e}")
            result = None
        #sleep for courtesy delay. Docs says rate limit of 40 per second so this is well within range
        time.sleep(0.10)
        if result:
            result["extracted_entity"] = title
            linked.append(result)
    return linked


#connect to Jetstream + collect posts
#code adapted from Jetstream docs https://bsky.network/docs/jetstream/

async def collect_posts():
    rows = []
    total_scanned = 0
    timed_out = False
    consecutive_live = 0
    start_time = time.time()

    #init a cursor to track the last post received
    last_cursor = int((time.time() - REPLAY_HOURS * 3600) * 1_000_000)

    #jetstream cuts off connection during high traffic or high throughput, so using cursor to pick
    #back up where connection drops. This gets cut off at MAX_POSTS to enforce an upper limit on posts processed
    #or when timeout is reached, or when all posts from past 3 hours have been collected

    while len(rows) < MAX_POSTS and not timed_out:
        uri = f"{JETSTREAM_BASE}&cursor={last_cursor}"
        try:
            async with websockets.connect(uri) as ws:
                try:
                    async for frame in ws:
                        event = json.loads(frame)

                        #filter server-side for posts only, skip other events
                        rec = event.get("commit", {}).get("record")
                        if not rec:
                            continue

                        #update cursor
                        last_cursor = event.get("cursor", last_cursor)

                        # detect when caught up to live by detecting 20 consecutive posts which are all within 30 seconds of live.
                        #had to use 20 consecutive because some bluesky clients report incorrect created_at timestamps

                        #wrapping in try/except to avoid crashing on posts with invalid timestamps
                        try:
                            created = datetime.fromisoformat(rec.get("createdAt").replace("Z", "+00:00"))
                        except (ValueError, AttributeError):
                            print(f"Skipping post with invalid timestamp: {rec.get('createdAt')}")
                            continue

                        age_seconds = (datetime.now(timezone.utc) - created).total_seconds()
                        if age_seconds < 30:
                            consecutive_live += 1
                            if consecutive_live >= 20:
                                print(f"Caught up to live, ending Jetstream connection")
                                timed_out = True
                                break
                        else:
                            consecutive_live = 0

                        #add timeout to avoid hanging indefinitely
                        if time.time() - start_time > TIMEOUT:
                            print(f"Timeout reached after {TIMEOUT} seconds.")
                            timed_out = True
                            break
                
                        #filter on english posts only
                        langs = rec.get("langs", [])
                        if "en" not in langs:
                            continue
                        
                        total_scanned += 1

                        text = rec.get("text", "")
                        text_lower = text.lower()        

                        #skip if no match on search terms
                        if not matches_keyword(text):
                            continue
                
                        rows.append({
                            "did": event.get("did", "unknown"),
                            "cursor": event.get("cursor"),
                            "created_at": rec.get("createdAt"),
                            "text": text,
                            "langs": str(rec.get("langs", [])),
                            "is_reply": rec.get("reply") is not None,
                            "has_embed": rec.get("embed") is not None,
                            "matched_keyword": next(kw for kw in WATCHED_SEARCH_QUERIES if kw.lower() in text_lower),
                        })

                        #prints output every 50 matches
                        if len(rows) % 50 == 0:
                            print(f"Matched {len(rows)} / scanned {total_scanned}")

                        #break when MAX_POSTS hit
                        if len(rows) >= MAX_POSTS:
                            break

                except websockets.ConnectionClosedError:
                        print(f"Connection has dropped; collected {len(rows)} matches from {total_scanned} total scanned.")
        except (websockets.exceptions.WebSocketException, OSError) as e:
            print(f"Connection error {e}; Reconnecting from cursor {last_cursor}")
            await asyncio.sleep(5)
            continue
    
    print(f"\nDone collecting. {len(rows)} matches from {total_scanned} scanned posts.")
    return rows


#run the fine-tuned classifier on collected posts to filter on posts which mention watching a movie/tv show and having a reaction

def classify_posts(classifier, rows):

    print(f"\nclassifying {len(rows)} posts")
    
    texts = [row["text"] for row in rows]
    results = classifier(texts, batch_size=32, truncation=True, max_length=514)
    
    reaction_posts = []
    for row, result in zip(rows, results):
        if result["label"] == "viewing_reaction":
            row["classifier_label"] = result["label"]
            row["classifier_probability"] = round(result["score"], 3)
            reaction_posts.append(row)
    
    print(f"Classifier kept {len(reaction_posts)} viewing reactions out of {len(rows)} posts "
          f"({100*len(reaction_posts)/len(rows):.1f}%)")
    return reaction_posts


#run spacy NER + moviedb API linking on reaction_posts

def extract_and_link_posts(nlp, reaction_posts):
    print(f"\nRunning NER + TMDb linking on {len(reaction_posts)} posts")
    
    results = []
    for idx, row in enumerate(reaction_posts):
        for match in extract_entity_and_link(row["text"], nlp, "WORK_OF_ART"):
            match["post_text"] = row["text"]
            match["created_at"] = row["created_at"]
            match["did"] = row["did"]
            match["classifier_probability"] = row["classifier_probability"]
            results.append(match)
        
        if (idx + 1) % 50 == 0:
            print(f"  NER processed {idx + 1} / {len(reaction_posts)} posts ({len(results)} titles linked)")
    
    linked_df = pd.DataFrame(results)
    
    if len(linked_df) > 0:
        print(f"\nTotal titles found in posts: {len(linked_df)}")
        print(f"Unique titles found in posts: {linked_df['tmdb_title'].nunique()}")
    else:
        print("\nNo titles could be extracted and linked.")
    
    return linked_df

#perform sentiment analysis

def analyse_sentiment(sentiment_model, linked_df):
    if len(linked_df) == 0:
        print("no posts to analyse sentiment for")
        return linked_df
    
    print(f"\nRunning sentiment analysis on {len(linked_df)} linked posts")
    
    #batched approach to running sentiment analysis on posts which have a movie/tv show title
    results = sentiment_model(linked_df["post_text"].tolist(), batch_size=32, truncation=True)

    linked_df["sentiment"] = [r["label"] for r in results]
    linked_df["sentiment_probability"] = [round(r["score"], 3) for r in results]

    print(f"Sentiment analysis complete.")
    return linked_df

def save_results(linked_df, agg_analysis):
    os.makedirs("data", exist_ok=True)

    # save locally
    linked_df.to_csv("data/sightings_latest.csv", index=False)
    agg_analysis.to_csv("data/summary_latest.csv")
    print("results saved locally")

    # save to Azure Blob Storage
    conn_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn_string:
        blob_service = BlobServiceClient.from_connection_string(conn_string)
        container = blob_service.get_container_client("sightings")
        try:
            container.upload_blob("sightings_latest.csv", linked_df.to_csv(index=False), overwrite=True)
            container.upload_blob("summary_latest.csv", agg_analysis.to_csv(), overwrite=True)
            print("Results uploaded to Azure Blob Storage")
        except Exception as e:
            print(f"Error occurred while uploading to Azure Blob Storage: {e}")
    else:
        print("no Azure conn string found, skipping blob upload")

#main
async def main():
    print("=" * 60)
    print("Movie/TV bluesky posts sentiment pipeline")
    print(f"replaying last {REPLAY_HOURS} hours from bluesky jetstream")
    print("=" * 60)

    load_dotenv()

    #load models
    classifier, nlp, sentiment_model = load_models()
    
    #collect posts from Jetstream
    rows = await collect_posts()
    
    if not rows:
        print("No posts collected. Exiting.")
        return
    
    #classify posts using fine-tuned model (filter for posts that are likely viewing reactions)
    viewing_reaction_posts = classify_posts(classifier, rows)
    
    if not viewing_reaction_posts:
        print("no viewing reactions found - terminating")
        return
    
    #run spacy NER and moviedb API linking
    linked_df = extract_and_link_posts(nlp, viewing_reaction_posts)
    
    if len(linked_df) == 0:
        print("no titles could be linked, exiting.")
        return
    
    #run sentiment analysis
    linked_df = analyse_sentiment(sentiment_model, linked_df)
    
    #aggregate and save results
    agg_analysis = aggregate_results(linked_df)
    save_results(linked_df, agg_analysis)
    
    print("\n" + "=" * 60)
    print("pipeline complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())