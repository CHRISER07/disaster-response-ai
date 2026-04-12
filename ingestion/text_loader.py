import pandas as pd
from langchain_core.documents import Document
import os
from datetime import datetime

# Twitter Epoch is Nov 4, 2010 01:42:54 UTC
TWITTER_EPOCH = 1288834974657

def snowflake_to_datetime(tweet_id) -> datetime:
    """
    Decodes a Twitter Snowflake ID into a precise UTC datetime object.
    This is critical for time-weighted RAG retrieval during rapid disaster progression!
    """
    try:
        tweet_id_int = int(str(tweet_id).replace('"', ''))
        # Shift 22 bits representing the machine id and sequence number
        timestamp_ms = (tweet_id_int >> 22) + TWITTER_EPOCH
        return datetime.fromtimestamp(timestamp_ms / 1000.0)
    except Exception:
        # Fallback if the ID isn't a valid snowflake format
        return None

def load_crisis_tweets(csv_path: str, max_documents: int = 1500) -> list[Document]:
    """
    Loads CrisisLex tweets from a CSV file, filtering for informative ones
    and wrapping them into LangChain Document objects with precise temporal metadata.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Tweet path {csv_path} not found.")
        return []

    df = pd.read_csv(csv_path)

    # Filter out "Not related" or "Related - but not informative" to only keep useful data
    if ' Informativeness' in df.columns:
        informative_df = df[df[' Informativeness'].str.contains('Related and informative', na=False, case=False)]
    elif 'Informativeness' in df.columns:
        informative_df = df[df['Informativeness'].str.contains('Related and informative', na=False, case=False)]
    else:
        informative_df = df
        
    informative_df = informative_df.head(max_documents)
    
    documents = []
    # Identify column names accounting for potential leading spaces in CSV header
    cols = df.columns.tolist()
    text_col = next((c for c in cols if 'tweet text' in c.lower()), None)
    id_col = next((c for c in cols if 'tweet id' in c.lower()), None)
    type_col = next((c for c in cols if 'information type' in c.lower()), None)
    source_col = next((c for c in cols if 'information source' in c.lower()), None)
    
    for _, row in informative_df.iterrows():
        text = str(row[text_col]) if text_col else ""
        tweet_id = row[id_col] if id_col else ""
        info_type = str(row[type_col]) if type_col else "Unknown"
        source = str(row[source_col]) if source_col else "Unknown"
        
        # Extract precise temporal metadata from Snowflake ID
        tweet_time = snowflake_to_datetime(tweet_id)
        time_str = tweet_time.isoformat() if tweet_time else "Unknown"
        
        metadata = {
            "source": "crisis_lex_tweet",
            "type": info_type,
            "author_source": source,
            "modality": "text",
            "tweet_id": str(tweet_id),
            "timestamp": time_str
        }
        
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
        
    print(f"Loaded {len(documents)} informative tweets with Snowflake Timestamps from CrisisLex.")
    return documents

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "data", "text", "CrisisLexT26", "CrisisLexT26", "2013_Colorado_floods", "2013_Colorado_floods-tweets_labeled.csv")
    docs = load_crisis_tweets(path, max_documents=5)
    for d in docs:
        print(f"[{d.metadata['timestamp']}] {d.metadata['type']} - {d.page_content}")
