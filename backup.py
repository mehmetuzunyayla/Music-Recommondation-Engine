import librosa
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
import seaborn as sns
import pandas as pd

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

def build_index(song_vectors):
    """Build an Annoy index for the given song vectors."""
    vector_dimension = len(song_vectors[0])
    index = AnnoyIndex(vector_dimension, 'angular')
    for i, vec in enumerate(song_vectors):
        index.add_item(i, vec)
    index.build(10)
    return index

def recommend_songs(index, song_vectors, song_paths, listened_songs_indices, num_neighbors=5):
    """Recommend songs based on the centroid of listened songs using the Annoy index."""
    listened_vectors = [song_vectors[i] for i in listened_songs_indices]
    centroid = np.mean(listened_vectors, axis=0)
    similar_indices = index.get_nns_by_vector(centroid, num_neighbors + len(listened_songs_indices))
    recommended_indices = [i for i in similar_indices if i not in listened_songs_indices][:num_neighbors]
    return [song_paths[i] for i in recommended_indices]

# List of paths to song files
song_paths = [
    'music/Daylight.wav', 'music/Die.wav','music/ECHO.wav', 'music/Enigma.wav', 'music/Golden_Age.wav', 'music/JAWS.wav',
    'music/MOMENTUM.wav', 'music/Night_Fever.wav', 'music/Rave.wav', 'music/Speedy_Boy.wav', 'music/Tek.wav', 'music/Wir.wav',
]

# Load songs and build Annoy index
song_vectors = load_songs(song_paths)
index = build_index(song_vectors)

# Example usage: Recommend songs based on a list of listened songs
listened_songs_indices = [0, 6, 2]
playlist = recommend_songs(index, song_vectors, song_paths, listened_songs_indices, num_neighbors=3)
print("Playlist:", playlist)
