"""Data cleaning and feature prep for the Spotify popularity project (Track 2)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42

# Kaggle column names -> names used in the assignment's data dictionary
COLUMN_RENAMES = {
    "track_id": "song_id",
    "track_name": "song_name",
    "track_genre": "song_genre",
}

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Spotify only counts a play as a stream after 30 seconds
MIN_DURATION_MS = 30_000


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path, index_col=0)
    return df.rename(columns=COLUMN_RENAMES)


def build_genre_table(df):
    """All (song_id, genre) pairs, saved before we collapse to one row per song."""
    return df[["song_id", "song_genre"]].drop_duplicates().reset_index(drop=True)


def collapse_song_ids(df):
    """One row per song_id, keeping the highest popularity copy.

    Copies only differ by genre label (and sometimes by ~1 popularity point).
    """
    n_genres = df.groupby("song_id")["song_genre"].nunique().rename("n_genres")
    deduped = (
        df.sort_values("popularity", ascending=False)
        .drop_duplicates("song_id")
        .join(n_genres, on="song_id")
    )
    return deduped


def collapse_rereleases(df):
    """One row per (artists, song_name), keeping the most popular release.

    The same recording often shows up as a single, on an album and on
    compilations. Spotify credits plays to one version, so the others sit near 0.
    """
    return df.sort_values("popularity", ascending=False).drop_duplicates(
        ["artists", "song_name"]
    )


def drop_invalid_audio(df):
    failed_analysis = (df["tempo"] == 0) | (df["time_signature"] == 0)
    too_short = df["duration_ms"] < MIN_DURATION_MS
    return df[~failed_analysis & ~too_short]


def add_features(df):
    df = df.copy()
    df["explicit"] = df["explicit"].astype(int)
    df["duration_min"] = df["duration_ms"] / 60_000
    df["n_artists"] = df["artists"].str.count(";") + 1
    df["primary_artist"] = df["artists"].str.split(";").str[0]

    # thresholds from Spotify's feature definitions
    df["is_instrumental"] = (df["instrumentalness"] > 0.5).astype(int)
    df["is_live"] = (df["liveness"] > 0.8).astype(int)
    df["speech_level"] = pd.cut(
        df["speechiness"],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["music", "mixed", "spoken"],
    ).astype(str)
    return df


def assign_split(df, test_size=0.2, random_state=RANDOM_STATE):
    """80/20 split grouped by primary artist.

    Keeps each artist on one side, so the model can't score well just by
    recognising the artist.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, _ = next(splitter.split(df, groups=df["primary_artist"]))
    df = df.copy()
    df["split"] = "test"
    df.iloc[train_idx, df.columns.get_loc("split")] = "train"
    return df


OUTPUT_COLUMNS = [
    "song_id",
    "artists",
    "primary_artist",
    "album_name",
    "song_name",
    "song_genre",
    "n_genres",
    "popularity",
    "duration_ms",
    "duration_min",
    "explicit",
    *AUDIO_FEATURES,
    "key",
    "mode",
    "time_signature",
    "n_artists",
    "is_instrumental",
    "is_live",
    "speech_level",
    "split",
]


def prepare_dataset(raw):
    """Run all cleaning steps. Returns (clean table, genre table, row-count log)."""
    steps = [("raw file", len(raw))]

    df = raw.dropna(subset=["artists", "album_name", "song_name"])
    steps.append(("drop rows missing artist/album/name", len(df)))

    genres = build_genre_table(df)

    df = df.drop_duplicates()
    steps.append(("drop exact duplicate rows", len(df)))

    df = collapse_song_ids(df)
    steps.append(("one row per song_id", len(df)))

    df = collapse_rereleases(df)
    steps.append(("one row per artist + song name", len(df)))

    df = drop_invalid_audio(df)
    steps.append(("drop failed audio analysis / <30s", len(df)))

    df = add_features(df)
    df = assign_split(df)

    clean = df[OUTPUT_COLUMNS].sort_values("song_id").reset_index(drop=True)
    genres = genres[genres["song_id"].isin(clean["song_id"])].reset_index(drop=True)

    log = pd.DataFrame(steps, columns=["step", "rows"])
    log["rows_removed"] = (-log["rows"].diff()).fillna(0).astype(int)
    return clean, genres, log


def save_outputs(clean, genres, out_dir=PROCESSED_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_dir / "spotify_clean.csv", index=False)
    genres.to_csv(out_dir / "song_genres_long.csv", index=False)
