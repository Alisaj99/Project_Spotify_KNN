# Import der benötigten Bibliotheken
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# Laden der Datensätze
# Laden der Datensätze
tracks_df = pd.read_csv('spotify_tracks.csv')
albums_df = pd.read_csv('spotify_albums.csv')
artists_df = pd.read_csv('spotify_artists.csv')
audio_features_df = pd.read_csv('low_level_audio_features.csv')
lyrics_features_df = pd.read_csv('lyrics_features.csv') 


# Explorative Datenanalyse (EDA) durchführen
numerical_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']
for feature in numerical_features:
    plt.figure(figsize=(10, 4))
    sns.histplot(tracks_df[feature], kde=True)
    plt.title(f'Verteilung von {feature}')
    plt.xlabel(feature)
    plt.ylabel('Häufigkeit')
    plt.show()

# Features auswählen und Daten vorbereiten
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']
X = tracks_df[features]
y = tracks_df['genre']  # Beispiel für ein Label, Anpassung erforderlich

# Daten skalieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modell trainieren
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_scaled)

# Modellvalidierung
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
predictions = knn.predict(X_test)
print("Validierungsgenauigkeit:", accuracy_score(y_test, predictions))

# Streamlit App
st.title('Spotify Song-Empfehlungssystem')

# Benutzerauswahl für Genre
genre_choice = st.selectbox('Wähle ein Genre:', options=['Alle'] + sorted(tracks_df['genre'].unique().tolist()))
if genre_choice != 'Alle':
    filtered_tracks = tracks_df[tracks_df['genre'] == genre_choice]
else:
    filtered_tracks = tracks_df

# Benutzerauswahl für Song
song_choice = st.selectbox('Wähle einen Song:', options=filtered_tracks['song_title'].tolist())

# Anzeigen der ausgewählten Songmerkmale
selected_song_features = filtered_tracks[filtered_tracks['song_title'] == song_choice][features]
st.write('Ausgewählte Songmerkmale:', selected_song_features)

# Normalisierung der Merkmale für die Vorhersage
scaled_features = scaler.transform(selected_song_features)

# Empfehlungen generieren
num_recommendations = st.slider("Anzahl der Empfehlungen", min_value=1, max_value=10, value=5)
distances, indices = knn.kneighbors(scaled_features, n_neighbors=num_recommendations)
recommended_songs = tracks_df.iloc[indices[0]].dropna()

# Empfehlungen anzeigen
st.write('Empfohlene Songs:')
for idx in indices[0]:
    song = tracks_df.iloc[idx]
    st.write(f"{song['song_title']} von {song['artist_name']} - Genre: {song['genre']}")



