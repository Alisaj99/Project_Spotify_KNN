import streamlit as st
import pandas as pd
import pickle

@st.cache(allow_output_mutation=True)
def load_model():
    # Laden des Modells
    with open('model/knn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache
def load_data():
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

    return load_and_merge_data(data_paths)

model = load_model()
data = load_data()

st.title('Spotify Song Recommendation System')

selected_genre = st.selectbox('Select Genre', options=['All'] + sorted(data['genre'].unique().tolist()))
if selected_genre != 'All':
    data = data[data['genre'] == selected_genre]

song_id = st.selectbox('Choose a song', options=data['id'].unique())
selected_song = data[data['id'] == song_id]

features = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]
selected_features = selected_song[features].values[0]

def get_recommendations(song_features, model, num_recommendations=5):
    distances, indices = model.kneighbors([song_features], n_neighbors=num_recommendations+1)
    return data.iloc[indices[0]]

recommendations = get_recommendations(selected_features, model)

st.write("Recommended Songs:")
for _, row in recommendations.iterrows():
    st.text(f"{row['name']} by {row['artists_id']}")

# Audio features display
st.write("Audio Features for the Selected Song:")
st.write(pd.DataFrame(selected_features.reshape(-1, len(features)), columns=features))




