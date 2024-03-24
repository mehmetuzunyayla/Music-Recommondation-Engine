import librosa
import json
import numpy as np

def extract_features_and_save_to_json(audio_files, json_filename):
    # List to hold all song features
    all_features = []

    # Iterate over each audio file
    for audio_file in audio_files:
        # Load the audio file
        y, sr = librosa.load(audio_file)

        # Extract features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        # Average the MFCCs and spectral centroid across frames
        mfccs_mean = mfccs.mean(axis=1)
        spectral_centroid_mean = spectral_centroid.mean(axis=1)

        # Create the feature vector by concatenating the features
        feature_vector = np.concatenate(([tempo], mfccs_mean, spectral_centroid_mean))

        # Extract the title from the file name
        title = audio_file.split('/')[-1].replace('.wav', '')

        # Append the features and title to the list
        all_features.append({
            'title': title,
            'vector': feature_vector.tolist()
        })

    # Convert the list of features to a JSON string
    features_json = json.dumps(all_features)

    # Write the JSON string to a file
    with open(json_filename, 'w') as file:
        file.write(features_json)

# Example usage
audio_files = [
    'music/Daylight.wav', 'music/Die.wav', 'music/ECHO.wav', 'music/Enigma.wav',
    'music/Golden_Age.wav', 'music/JAWS.wav', 'music/MOMENTUM.wav', 'music/Night_Fever.wav',
    'music/Rave.wav', 'music/Speedy_Boy.wav', 'music/Tek.wav', 'music/Wir.wav'
]
extract_features_and_save_to_json(audio_files, 'song_features.json')
