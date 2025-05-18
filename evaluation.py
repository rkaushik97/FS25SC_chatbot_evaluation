import json
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Song Metadata ---
songs = [
    {"title": "Blinding Lights", "artist": "The Weeknd", "genre": "Synthwave", "release_year": 2019, "description": "A high-energy synthwave track with a retro vibe, perfect for late-night drives."},
    {"title": "Bohemian Rhapsody", "artist": "Queen", "genre": "Rock", "release_year": 1975, "description": "An epic rock ballad with dramatic shifts in tempo and mood."},
    {"title": "Shape of You", "artist": "Ed Sheeran", "genre": "Pop", "release_year": 2017, "description": "A danceable pop track with catchy lyrics and a memorable melody."},
    {"title": "Imagine", "artist": "John Lennon", "genre": "Pop", "release_year": 1971, "description": "A hopeful anthem calling for world peace and unity."},
    {"title": "Rolling in the Deep", "artist": "Adele", "genre": "Soul", "release_year": 2010, "description": "A soulful track with deep, powerful vocals and emotional lyrics."},
    {"title": "Uptown Funk", "artist": "Mark Ronson feat. Bruno Mars", "genre": "Funk", "release_year": 2014, "description": "An energetic and funky track that encourages dancing and having a good time."},
    {"title": "Let It Be", "artist": "The Beatles", "genre": "Rock", "release_year": 1970, "description": "A timeless ballad with an uplifting message of acceptance."},
    {"title": "Billie Jean", "artist": "Michael Jackson", "genre": "Pop", "release_year": 1982, "description": "A catchy pop track with a deep groove and unforgettable bassline."},
    {"title": "Smells Like Teen Spirit", "artist": "Nirvana", "genre": "Grunge", "release_year": 1991, "description": "A defining track of the grunge movement with raw energy and angst."},
    {"title": "Hurt", "artist": "Nine Inch Nails", "genre": "Industrial", "release_year": 1994, "description": "A dark and haunting track that explores themes of despair and self-reflection."},
    {"title": "Old Town Road", "artist": "Lil Nas X", "genre": "Country Rap", "release_year": 2018, "description": "A catchy fusion of country and rap with a viral hook."},
    {"title": "Bohemian Rhapsody", "artist": "Queen", "genre": "Rock", "release_year": 1975, "description": "An epic rock ballad with dramatic shifts in tempo and mood."},
    {"title": "Hey Jude", "artist": "The Beatles", "genre": "Pop", "release_year": 1968, "description": "A classic anthem with emotional depth and a powerful sing-along outro."},
    {"title": "What a Wonderful World", "artist": "Louis Armstrong", "genre": "Jazz", "release_year": 1967, "description": "A timeless jazz standard with a message of gratitude and beauty."},
    {"title": "Bohemian Rhapsody", "artist": "Queen", "genre": "Rock", "release_year": 1975, "description": "An epic rock ballad with dramatic shifts in tempo and mood."},
    {"title": "Hey Ya!", "artist": "OutKast", "genre": "Hip-Hop", "release_year": 2003, "description": "A funky track that blends hip-hop and pop, famous for its catchy rhythm."},
    {"title": "Like a Rolling Stone", "artist": "Bob Dylan", "genre": "Folk Rock", "release_year": 1965, "description": "A revolutionary track with poignant lyrics and an innovative sound."},
    {"title": "I Will Always Love You", "artist": "Whitney Houston", "genre": "Pop", "release_year": 1992, "description": "A powerful ballad showcasing Whitney Houston's unmatched vocal range."},
    {"title": "Livin' on a Prayer", "artist": "Bon Jovi", "genre": "Rock", "release_year": 1986, "description": "An anthem of resilience and hope, with a catchy chorus and upbeat energy."},
    {"title": "Sweet Child O' Mine", "artist": "Guns N' Roses", "genre": "Hard Rock", "release_year": 1987, "description": "A hard rock classic with iconic guitar riffs and heartfelt lyrics."},
    {"title": "Thriller", "artist": "Michael Jackson", "genre": "Pop", "release_year": 1982, "description": "A groundbreaking pop song with a legendary music video and spooky vibes."}
]

import re

def extract_title_artist(text, default_title="", default_artist=""):
    """
    Extract 'Song' and 'Artist' from a text snippet.
    Falls back on defaults if clean extraction fails.
    """
    match = re.search(r"'([^']+)'\s+by\s+([a-zA-Z0-9\s&\-]+)", text, re.IGNORECASE)
    if match:
        title = match.group(1).strip()
        artist = match.group(2).strip()
    else:
        title = default_title or "Unknown Title"
        artist = default_artist or "Unknown Artist"
    return title, artist

# --- Song Lookup ---
def get_song_description(title, artist):
    """Returns the description of the song based on title and artist."""
    if not title or not artist:
        return "A music track"
    
    title = title.strip().lower()
    artist = artist.strip().lower()
    
    for song in songs:
        s_title = song["title"].strip().lower()
        s_artist = song["artist"].strip().lower()

        # Exact match
        if s_title == title and s_artist == artist:
            return song["description"]
        
        # Relaxed matching
        if title in s_title or s_title in title:
            if artist in s_artist or s_artist in artist:
                return song["description"]

    return "A popular music track"

# --- Evaluation Metrics ---
def feature_alignment_score(user_input, response_text):
    feature_keywords = {
        "energy": ["energetic", "high-energy", "dynamic", "intense", "lively"],
        "tempo": ["fast", "upbeat", "slow", "moderate", "laid-back"],
        "vocals": ["strong vocals", "vocal-driven", "soft singing", "powerful voice"],
        "instrumentation": ["instrumental", "guitar-heavy", "synth-driven", "funky"],
        "melodic": ["melodic", "catchy", "tuneful", "harmonic"],
        "lyrics_sentiment": ["positive", "happy", "neutral", "dark", "sad"]
    }

    alignment_scores = []
    for feature, keywords in feature_keywords.items():
        matched = any(keyword in response_text.lower() for keyword in keywords)
        alignment_scores.append(1.0 if matched else 0.0)

    return sum(alignment_scores) / len(alignment_scores)

def sentiment_coherence_score(user_sentiment, response_text, ground_truth_desc):
    response_sentiment = TextBlob(response_text).sentiment.polarity
    truth_sentiment = TextBlob(ground_truth_desc).sentiment.polarity

    if user_sentiment == 1:
        return max(0, response_sentiment)
    elif user_sentiment == -1:
        return max(0, -response_sentiment)
    else:
        return 1 - abs(response_sentiment)

def relevance_score(response_text, ground_truth_desc):
    if not ground_truth_desc:
        return 0.0
    vectorizer = TfidfVectorizer().fit_transform([response_text, ground_truth_desc])
    return cosine_similarity(vectorizer[0], vectorizer[1])[0][0]

# --- Main Evaluation Function ---
def evaluate_responses(json_data):
    results = []
    for entry in json_data:
        fuzzy_response = entry["fuzzy_logic_response"]
        llm_response = entry["llm_response"]
        user_input = entry["user_input"]

        # Improved title/artist extraction
        title, artist = extract_title_artist(fuzzy_response)

        ground_truth = get_song_description(title, artist)

        fas_fuzzy = feature_alignment_score(user_input, fuzzy_response)
        fas_llm = feature_alignment_score(user_input, llm_response)

        scs_fuzzy = sentiment_coherence_score(user_input["lyrics_sentiment"], fuzzy_response, ground_truth)
        scs_llm = sentiment_coherence_score(user_input["lyrics_sentiment"], llm_response, ground_truth)

        rs_fuzzy = relevance_score(fuzzy_response, ground_truth)
        rs_llm = relevance_score(llm_response, ground_truth)

        results.append({
            "song": f"{title} by {artist}",
            "fuzzy_logic_response": fuzzy_response,
            "llm_response": llm_response,
            "fuzzy_logic": {
                "feature_alignment_score": round(fas_fuzzy, 2),
                "sentiment_coherence_score": round(scs_fuzzy, 2),
                "relevance_score": round(rs_fuzzy, 2)
            },
            "generative_ai": {
                "feature_alignment_score": round(fas_llm, 2),
                "sentiment_coherence_score": round(scs_llm, 2),
                "relevance_score": round(rs_llm, 2)
            }
        })
    return results

# --- Entry Point ---
if __name__ == "__main__":
    try:
        with open("feedback_data_20250508_154112.json") as f:
            data = json.load(f)
        metrics = evaluate_responses(data)
        # Print metrics nicely
        print(json.dumps(metrics, indent=2))
        # Save metrics to a file
        with open("evaluation_results.json", "w") as outfile:
            json.dump(metrics, outfile, indent=2)
        print("Evaluation results saved to evaluation_results.json")
    except Exception as e:
        print(f"Failed to run evaluation: {e}")
