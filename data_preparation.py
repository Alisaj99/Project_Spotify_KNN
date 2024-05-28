import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import os

# Pfade zu den Datendateien
data_paths = {
    'albums': "C:/Users/jonuz/Desktop/Spotify/SpotGenTrack/Data Sources/spotify_albums.csv",
    'artists': "C:/Users/jonuz/Desktop/Spotify/SpotGenTrack/Data Sources/spotify_artists.csv",
    'tracks': "C:/Users/jonuz/Desktop/Spotify/SpotGenTrack/Data Sources/spotify_tracks.csv",
    'audio_features': "C:/Users/jonuz/Desktop/Spotify/SpotGenTrack/Features Extracted/low_level_audio_features.csv",
    'lyrics_features': "C:/Users/jonuz/Desktop/Spotify/SpotGenTrack/Features Extracted/lyrics_features.csv"
}

def load_and_merge_data(paths):
    albums = pd.read_csv(paths['albums'])
    artists = pd.read_csv(paths['artists'])
    tracks = pd.read_csv(paths['tracks'])
    audio_features = pd.read_csv(paths['audio_features'])
    lyrics_features = pd.read_csv(paths['lyrics_features'])

    tracks = pd.merge(tracks, albums, left_on='album_id', right_on='id', how='left', suffixes=('', '_album'))
    tracks = pd.merge(tracks, artists, left_on='artists_id', right_on='id', how='left', suffixes=('', '_artist'))
    combined_data = pd.merge(tracks, audio_features, left_on='id', right_on='track_id', how='inner', suffixes=('', '_audio'))
    combined_data = pd.merge(combined_data, lyrics_features, left_on='id', right_on='track_id', how='inner', suffixes=('', '_lyrics'))
    combined_data.dropna(subset=["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"], inplace=True)

    return combined_data

def prepare_features(data):
    # Auswahl und Vorbereitung der Features
    features = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]
    return data[features]

def train_model(features):
    # Training des KNN-Modells
    model = NearestNeighbors(n_neighbors=5)
    model.fit(features)
    return model

def save_model(model, model_path='model/knn_model.pkl'):
    # Speichern des Modells mit Pickle
    model_directory = os.path.dirname(model_path)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

# Ausführung der Vorbereitung und des Trainings
data = load_and_merge_data(data_paths)
features = prepare_features(data)
model = train_model(features)
save_model(model)
print("Das Modell wurde erfolgreich trainiert und gespeichert.")
