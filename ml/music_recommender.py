import pandas as pd

df = pd.read_csv(r"C:\Users\user\python-class\emotion_music_flask\data\SpotifyFeatures.csv")

EMOTION_MAP = {
    "happy": {"valence": (0.6, 1.0), "energy": (0.6, 1.0)},
    "sad": {"valence": (0.0, 0.4), "energy": (0.0, 0.5)},
    "angry": {"valence": (0.0, 0.3), "energy": (0.7, 1.0)},
    "fear": {"valence": (0.2, 0.5), "energy": (0.3, 0.6)},
    "neutral": {"valence": (0.4, 0.6), "energy": (0.4, 0.6)},
    "surprise": {"valence": (0.5, 0.8), "energy": (0.5, 0.8)},
    "disgust": {"valence": (0.1, 0.4), "energy": (0.2, 0.5)}
}

def recommend_music(emotion, n=5):
    rules = EMOTION_MAP[emotion]

    songs = df[
        (df['valence'].between(*rules['valence'])) &
        (df['energy'].between(*rules['energy']))
    ]

    return songs.sample(n)[['track_name', 'artist_name']]
