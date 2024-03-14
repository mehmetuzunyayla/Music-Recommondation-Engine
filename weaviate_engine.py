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
                    "dataType": ["number[]"]  # Removed the 'index' line
                }
            ]
        }
    ]
}

'''
client.schema.delete_all()
client.schema.create(schema)
'''

existing_classes = client.schema.get()['classes']
class_names = [cls['class'] for cls in existing_classes]
if "Song" not in class_names:
    client.schema.create(schema)
else:
    print("Class 'Song' already exists.")

song_paths = [
    'music/Daylight.wav', 'music/Die.wav', 'music/ECHO.wav', 'music/Enigma.wav',
    'music/Golden_Age.wav', 'music/JAWS.wav', 'music/MOMENTUM.wav', 'music/Night_Fever.wav',
    'music/Rave.wav', 'music/Speedy_Boy.wav', 'music/Tek.wav', 'music/Wir.wav'
]

'''
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
    return normalize(song_vectors)

song_vectors = load_songs(song_paths)
'''


'''
for path, vector in zip(song_paths, song_vectors):
    song_object = {
        "path": path,
        "features": vector.tolist()
    }
    client.data_object.create(song_object, "Song")
response = client.query.get("Song", properties=["path", "features"]).with_limit(5).do()
print("Sample songs from Weaviate:", response)'''


def recommend_similar_songs(client, song_path, num_recommendations=3):

  # Ensure at least 3 recommendations are requested
  if num_recommendations < 1:
      num_recommendations = 3

  # Retrieve the vector of the given song
  response = client.query.get("Song", properties=["features"]) \
      .with_where({"path": ["path"], "operator": "Equal", "valueString": song_path}) \
      .with_additional("distance").do()  # Request distance in the response

  if not response["data"]["Get"]["Song"]:
      print(f"Song with path '{song_path}' not found.")
      return []

  song_vector = response["data"]["Get"]["Song"][0]["features"]

  # Query for similar songs based on the vector
  response = client.query.get("Song", properties=["path", "_additional {distance}"]) \
      .with_near_vector({"vector": song_vector}) \
      .with_limit(num_recommendations + 1).do()  # Request one extra for filtering

  if "data" not in response or "Get" not in response["data"] or "Song" not in response["data"]["Get"]:
      print("No similar songs found.")
      return []

  # Filter out the original song and sort by distance
  similar_songs = sorted([(song["path"], song["_additional"]["distance"]) for song in response["data"]["Get"]["Song"] if song["path"] != song_path], key=lambda x: x[1])

  return similar_songs[:num_recommendations]


song_path = 'music/Daylight.wav'  # Path to the song for which you want recommendations
num_recommendations = 5  # Number of recommendations to retrieve

recommended_songs = recommend_similar_songs(client, song_path, num_recommendations)

if recommended_songs:
  print(f"Recommended songs similar to '{song_path}':")
  for song, distance in recommended_songs:
    print(f"\t- {song} (Distance: {distance})")
else:
  print(f"No similar songs found for '{song_path}'.")






'''
# Print feature vectors of a few songs in the database
response = client.query.get("Song", properties=["path", "features"]).with_limit(5).do()
if 'data' in response and 'Get' in response['data'] and 'Song' in response['data']['Get']:
    for song in response['data']['Get']['Song']:
        print(f"Path: {song['path']}, Features: {song['features']}")



def recommend_songs(client, listened_songs_paths, num_neighbors=5):
    """Recommend songs based on each listened song using Weaviate."""
    recommended_paths = set()  # Use a set to avoid duplicate recommendations
    for song_path in listened_songs_paths:
        # Retrieve the vector of the listened song from Weaviate
        response = client.query.get(
            "Song",
            properties=["features"]
        ).with_where(
            {"path": ["path"], "operator": "Equal", "valueString": song_path}  # Corrected filter
        ).do()

        if 'data' in response and 'Get' in response['data'] and 'Song' in response['data']['Get']:
            listened_vector = response['data']['Get']['Song'][0]['features']

            # Query Weaviate for similar songs to the current listened song
            try:
                response = client.query.get(
                    "Song",
                    properties=["path"]
                ).with_near_vector(
                    {"vector": listened_vector}
                ).with_limit(
                    num_neighbors
                ).with_additional(
                    "distance"  # Request distance in the response
                ).do()

                print(f"Query result for song {song_path}:", response)

                if 'data' in response and 'Get' in response['data'] and 'Song' in response['data']['Get']:
                    for song in response['data']['Get']['Song']:
                        recommended_paths.add(song['path'])
                        print(f"Recommended song: {song['path']}, Distance: {song['_additional']['distance']}")
                else:
                    print(f"No recommendations found for song {song_path}.")
            except Exception as e:
                print(f"Error querying Weaviate for song {song_path}: {e}")
        else:
            print(f"Vector not found for song {song_path}.")

    return list(recommended_paths)  # Convert the set to a list

# Example usage: Recommend songs based on a list of listened songs
listened_songs_paths = ['music/Daylight.wav']
playlist = recommend_songs(client, listened_songs_paths, num_neighbors=10)
print("Playlist:", playlist)

#response = client.query.get("Song", properties=["path"]).do()
#print("All songs:", response)

'''