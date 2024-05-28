import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

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

data = load_and_merge_data(data_paths)
features = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

def train_model(data, feature_columns):
    model = NearestNeighbors()
    model.fit(data[feature_columns])
    return model

model = train_model(data, features)

# Stellen Sie sicher, dass das Verzeichnis existiert
model_directory = 'model'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

model_path = os.path.join(model_directory, 'knn_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print("Das Modell wurde erfolgreich trainiert und gespeichert.")

