print("hello")

import pandas as pd
import numpy as np
import ast
import pickle

# Laden der Datensätze
tracks_df = pd.read_csv(r'C:\Users\monaa\Desktop\Seminare & Vorlesungen\Analyseanwendungen\spotify_data\SpotGenTrack\Data Sources\spotify_tracks.csv')
albums_df = pd.read_csv(r'C:\Users\monaa\Desktop\Seminare & Vorlesungen\Analyseanwendungen\spotify_data\SpotGenTrack\Data Sources\spotify_albums.csv')
artists_df = pd.read_csv(r'C:\Users\monaa\Desktop\Seminare & Vorlesungen\Analyseanwendungen\spotify_data\SpotGenTrack\Data Sources\spotify_artists.csv')
audio_features_df = pd.read_csv(r'C:\Users\monaa\Desktop\Seminare & Vorlesungen\Analyseanwendungen\spotify_data\SpotGenTrack\Data Sources\low_level_audio_features.csv')
lyrics_features_df = pd.read_csv(r'C:\Users\monaa\Desktop\Seminare & Vorlesungen\Analyseanwendungen\spotify_data\SpotGenTrack\Data Sources\lyrics_features.csv')

# Überprüfen der Spaltennamen
print("Tracks DF Columns:", tracks_df.columns)
print("Audio Features DF Columns:", audio_features_df.columns)
print("Lyrics Features DF Columns:", lyrics_features_df.columns)
print("Albums DF Columns:", albums_df.columns)
print("Artists DF Columns:", artists_df.columns)

# Umbenennen der Spalten für Konsistenz
tracks_df = tracks_df.rename(columns={'id': 'track_id', 'artists_id': 'artist_id'})
albums_df = albums_df.rename(columns={'id': 'album_id'})
artists_df = artists_df.rename(columns={'id': 'artist_id'})

# Konvertieren von artist_id Spaltenwerten in tracks_df von Listen zu einzelnen Werten
tracks_df['artist_id'] = tracks_df['artist_id'].apply(ast.literal_eval)  # Umwandeln von String zu Liste
tracks_df = tracks_df.explode('artist_id')  # Auflösen der Listen zu separaten Zeilen

# Überprüfen der Inhalte der relevanten Spalten
print("Sample artist_id in tracks_df:", tracks_df['artist_id'].head())
print("Sample artist_id in artists_df:", artists_df['artist_id'].head())

# Entfernen von unnötigen Spalten
def drop_columns_safely(df, columns):
    return df.drop(columns=[col for col in columns if col in df.columns])

tracks_df = drop_columns_safely(tracks_df, ['Unnamed: 0', 'analysis_url', 'available_markets', 'country', 'href', 'playlist', 'preview_url', 'track_href', 'track_name_prev', 'uri', 'type'])
albums_df = drop_columns_safely(albums_df, ['Unnamed: 0', 'available_markets', 'external_urls', 'href', 'images', 'release_date', 'release_date_precision', 'track_name_prev', 'uri', 'type'])
artists_df = drop_columns_safely(artists_df, ['Unnamed: 0', 'genres', 'track_name_prev', 'type'])
audio_features_df = drop_columns_safely(audio_features_df, ['Unnamed: 0'])
lyrics_features_df = drop_columns_safely(lyrics_features_df, ['Unnamed: 0'])

# Zusammenführen der Daten
merged_df_1 = pd.merge(tracks_df, audio_features_df, on='track_id')
merged_df_2 = pd.merge(merged_df_1, lyrics_features_df, on='track_id')
albums_df = albums_df.rename(columns={'artist_id': 'album_artist_id'})  # Umbenennen zur Vermeidung von Konflikten
merged_df_3 = pd.merge(merged_df_2, albums_df, on='album_id')
merged_df_4 = pd.merge(merged_df_3, artists_df, left_on='artist_id', right_on='artist_id')

# Zusammengeführte Daten als CSV-Datei speichern
merged_df_4.to_csv(r'C:\Users\monaa\Desktop\Seminare & Vorlesungen\Analyseanwendungen\spotify_data\SpotGenTrack\Data Sources\merged_df_4.csv', index=False)

print("Merged DF Columns:", merged_df_4.columns)


from sklearn.preprocessing import StandardScaler

# Relevante Features auswählen
features = ['danceability', 'energy', 'tempo', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']

# Features und Track IDs extrahieren
X = merged_df_4[features]
track_ids = merged_df_4['track_id']

# Normalisierung der Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalisierte Features in ein DataFrame konvertieren
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
X_scaled_df['track_id'] = track_ids

# Speichern des Scalers für zukünftige Verwendung
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


from sklearn.neighbors import NearestNeighbors

# KNN-Modell trainieren
knn = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
knn.fit(X_scaled_df[features])

# Modell speichern
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)


import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Laden der Modelle und Daten
scaler = pickle.load(open('scaler.pkl', 'rb'))
knn = pickle.load(open('knn_model.pkl', 'rb'))
merged_df_4 = pd.read_csv(r'C:\Users\monaa\Desktop\Seminare & Vorlesungen\Analyseanwendungen\spotify_data\SpotGenTrack\Data Sources\merged_df_4.csv')

# Feature-Auswahl
features = ['danceability', 'energy', 'tempo', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']

# Streamlit-Benutzeroberfläche
st.title('Spotify Song Recommendation System')

# Song-Auswahl
song_list = merged_df_4['name'].unique()
selected_song = st.selectbox('Select a song', song_list)

# Song-Informationen abrufen
selected_song_data = merged_df_4[merged_df_4['name'] == selected_song].iloc[0]
selected_song_features = selected_song_data[features].values.reshape(1, -1)

# Merkmale normalisieren
selected_song_features_scaled = scaler.transform(selected_song_features)

# Empfehlungen generieren
distances, indices = knn.kneighbors(selected_song_features_scaled)
recommendations = merged_df_4.iloc[indices[0]]

# Empfehlungen anzeigen
st.subheader('Recommendations:')
for i, row in recommendations.iterrows():
    st.write(f"{row['name']} by {row['artist_id']} - Danceability: {row['danceability']}, Energy: {row['energy']}")




