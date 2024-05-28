import streamlit as st
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
import sklearn

@st.cache(allow_output_mutation=True)
def load_model():
    with open('model/knn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache
def load_data():
    path = "C:/Users/jonuz/Desktop/Spotify/SpotGenTrack/Data Sources/spotify_tracks.csv"
    return pd.read_csv(path)

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
    st.text(f"{row['name']} by {row['artist_id']}")

# Audio features display
st.write("Audio Features for the Selected Song:")
st.write(pd.DataFrame(selected_features.reshape(-1, len(features)), columns=features))





