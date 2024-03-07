from weaviate import Client
import librosa

# Step 1: Connect to Weaviate instance
client = Client("http://localhost:8080")

# Step 2: Define schema
schema = {
    "classes": [
        {
            "class": "Song",
            "properties": [
                {"name": "title", "dataType": ["string"], "indexInverted": True},
                {"name": "features", "dataType": ["number[]"]},
            ],
        }
    ]
}
client.schema.create(schema)

# Step 3: Import data
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return mfcc.mean(axis=1).tolist()

songs = [
    {"title": "Song 1", "file_path": "song1.wav"},
    {"title": "Song 2", "file_path": "song2.wav"},
    {"title": "Song 3", "file_path": "song3.wav"},
]

for song in songs:
    features = extract_features(song["file_path"])
    client.data_object.create({"title": song["title"], "features": features}, "Song")

# Step 4: Query for recommendations
def recommend_songs(features, num_recommendations=3):
    query = {
        "query": {
            "vector": features,
            "k": num_recommendations,
        }
    }
    results = client.query.raw_get("Song", query)
    return [hit["_source"]["title"] for hit in results["hits"]]

# Example usage
listened_song_features = extract_features("song1.wav")
recommended_songs = recommend_songs(listened_song_features)
print("Recommended songs:", recommended_songs)
