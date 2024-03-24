import librosa
import numpy as np
import psycopg2

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
        #chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        song_vector = np.concatenate(([tempo], np.mean(mfcc, axis=1), np.mean(spectral_centroid, axis=1)))
        song_vectors.append(song_vector)
    return song_vectors

# Connect to your PostgreSQL database
conn = psycopg2.connect("dbname=postgres user=postgres password=123456")
cur = conn.cursor()

# Extract features and insert into the database
song_vectors = load_songs(song_paths)
for path, vector in zip(song_paths, song_vectors):
    song_name = path.split('/')[-1].replace('.wav', '')  # Extract the song name from the path
    cur.execute("INSERT INTO music (title, features) VALUES (%s, %s)", (song_name, vector.tolist()))

# Commit the transaction and close the connection
conn.commit()
cur.close()
conn.close()