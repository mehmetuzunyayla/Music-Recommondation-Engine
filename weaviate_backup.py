import weaviate
import librosa
import numpy as np
import warnings
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore", category=DeprecationWarning)

client = weaviate.Client("http://localhost:50010")

schema = {
    "classes": [
        {
            "class": "Song",
            "vectorizer": "none",  # We'll manually provide the vectors
            "properties": [
                {
                    "name": "path",
                    "dataType": ["string"],
                    "indexInverted": True
                },
                {
                    "name": "features",
                    "dataType": ["number[]"],
                    "index": True  # Enable vector indexing
                }
            ]
        }
    ]
}
client.schema.delete_all()
client.schema.create(schema)

'''
existing_classes = client.schema.get()['classes']
class_names = [cls['class'] for cls in existing_classes]
if "Song" not in class_names:
    client.schema.create(schema)
else:
    print("Class 'Song' already exists.")
'''

song_paths = [
    'music/Daylight.wav', 'music/Die.wav', 'music/ECHO.wav', 'music/Enigma.wav',
    'music/Golden_Age.wav', 'music/JAWS.wav', 'music/MOMENTUM.wav', 'music/Night_Fever.wav',
    'music/Rave.wav', 'music/Speedy_Boy.wav', 'music/Tek.wav', 'music/Wir.wav'
]

def load_songs(song_paths):
    """Load songs from the given paths and extract features to create song vectors."""
    song_vectors = []
    for path in song_paths:
        y, sr = librosa.load(path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        song_vector = np.concatenate(([tempo], np.mean(mfcc, axis=1), np.mean(spectral_centroid, axis=1)))
        song_vectors.append(song_vector)
    return song_vectors

song_vectors = load_songs(song_paths)

for path, vector in zip(song_paths, song_vectors):
    song_object = {
        "path": path,
        "features": vector.tolist()
    }
    client.data_object.create(song_object, "Song")

'''# Print feature vectors of a few songs in the database
response = client.query.get("Song", properties=["path", "features"]).with_limit(5).do()
if 'data' in response and 'Get' in response['data'] and 'Song' in response['data']['Get']:
    for song in response['data']['Get']['Song']:
        print(f"Path: {song['path']}, Features: {song['features']}")'''



def recommend_songs(client, song_vectors, listened_songs_indices, num_neighbors=5):
    """Recommend songs based on each listened song using Weaviate."""
    recommended_paths = set()  # Use a set to avoid duplicate recommendations
    for index in listened_songs_indices:
        listened_vector = song_vectors[index]

        # Query Weaviate for similar songs to the current listened song
        response = client.query.get(
            "Song",
            properties=["path"]
        ).with_near_vector(
            {"vector": listened_vector.tolist()}
        ).with_limit(
            num_neighbors
        ).do()

        print(f"Query result for song index {index} ({path}):", response)  # Add this print statement

        if 'data' in response and 'Get' in response['data'] and 'Song' in response['data']['Get']:
            recommended_paths.update(song['path'] for song in response['data']['Get']['Song'])

    return list(recommended_paths)  # Convert the set to a list

# Example usage: Recommend songs based on a list of listened songs
listened_songs_indices = [10]
playlist = recommend_songs(client, song_vectors, listened_songs_indices, num_neighbors=10)
print("Playlist:", playlist)

response = client.query.get("Song", properties=["features"]).do()
print("All songs:", response)
