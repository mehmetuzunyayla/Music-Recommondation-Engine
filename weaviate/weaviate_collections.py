import weaviate
import weaviate.classes as wvc
import json

client = weaviate.connect_to_local()

try:
    # Delete the existing collection if it exists
    client.collections.delete("Song")

    # Create a new collection for songs
    client.collections.create(
        "Song",
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=wvc.config.VectorDistances.COSINE
        )
    )

    print("Collection 'Song' created successfully.")

    # Load the song features from the JSON file
    with open('weaviate/song_features.json', 'r') as file:
        song_features = json.load(file)

    # Create a list of song objects
    song_objs = []
    for song in song_features:
        song_objs.append(wvc.data.DataObject(
            properties={
                "title": song["title"],
            },
            vector=song["vector"]
        ))

    # Insert the song objects into the collection
    songs = client.collections.get("Song")

    songs.data.insert_many(song_objs)  # This uses batching under the hood
    print("Successfully batched to Weaviate.")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Close the client connection
    client.close()
