"""Cleaning and feature preparation for the Spotify popularity project (Track 2).

Everything that happens to the raw Kaggle file before modeling lives here, so the
notebooks only call these functions and the processed output is reproducible.
"""

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
    """Read the Kaggle CSV and align column names with the assignment."""
    df = pd.read_csv(path, index_col=0)
    return df.rename(columns=COLUMN_RENAMES)


class CleaningLog:
    """Keeps a row count after each cleaning step for the write-up."""

    def __init__(self):
        self.steps = []

    def record(self, step, df):
        self.steps.append({"step": step, "rows": len(df)})

    def to_frame(self):
        log = pd.DataFrame(self.steps)
        log["rows_removed"] = (-log["rows"].diff()).fillna(0).astype(int)
        return log


def build_genre_table(df):
    """Long table of every (song_id, genre) pair before rows are collapsed.

    The same song_id appears under several genres, so this table is kept
    separately in case the modeling side wants multi-hot genre features.
    """
    return df[["song_id", "song_genre"]].drop_duplicates().reset_index(drop=True)


def collapse_song_ids(df):
    """Keep one row per song_id.

    Copies of a song_id only differ in genre label and, for a few hundred songs,
    by about one popularity point (the scrape ran over several days). We keep the
    highest popularity copy and remember how many genres the song was listed under.
    """
    n_genres = df.groupby("song_id")["song_genre"].nunique().rename("n_genres")
    deduped = (
        df.sort_values("popularity", ascending=False)
        .drop_duplicates("song_id")
        .join(n_genres, on="song_id")
    )
    return deduped


def collapse_rereleases(df):
    """Keep one row per (artists, song_name).

    The same recording is often published several times (single, album,
    compilation) under different song_ids. Spotify credits most plays to one
    version, so the other copies sit at or near 0 popularity. Leaving them in
    teaches the model that identical audio is both a hit and a flop. We keep the
    most popular release.
    """
    return df.sort_values("popularity", ascending=False).drop_duplicates(
        ["artists", "song_name"]
    )


def drop_invalid_audio(df):
    """Remove songs where Spotify's audio analysis clearly failed or the track is too short to stream."""
    failed_analysis = (df["tempo"] == 0) | (df["time_signature"] == 0)
    too_short = df["duration_ms"] < MIN_DURATION_MS
    return df[~failed_analysis & ~too_short]


def add_features(df):
    """Derived columns that only use song attributes, never the target."""
    df = df.copy()
    df["explicit"] = df["explicit"].astype(int)
    df["duration_min"] = df["duration_ms"] / 60_000
    df["n_artists"] = df["artists"].str.count(";") + 1
    df["primary_artist"] = df["artists"].str.split(";").str[0]

    # Thresholds below come from Spotify's own feature definitions
    df["is_instrumental"] = (df["instrumentalness"] > 0.5).astype(int)
    df["is_live"] = (df["liveness"] > 0.8).astype(int)
    df["speech_level"] = pd.cut(
        df["speechiness"],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["music", "mixed", "spoken"],
    ).astype(str)
    return df


def assign_split(df, test_size=0.2, random_state=RANDOM_STATE):
    """Train/test split grouped by primary artist.

    An artist's songs are all on one side of the split. Otherwise a model can
    score well just by recognising the artist, which says nothing about how
    the song itself sounds.
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
    """Run the full cleaning pipeline.

    Returns the model-ready table, the long genre table and the cleaning log.
    """
    log = CleaningLog()
    log.record("raw file", raw)

    df = raw.dropna(subset=["artists", "album_name", "song_name"])
    log.record("drop rows missing artist/album/name", df)

    genres = build_genre_table(df)

    df = df.drop_duplicates()
    log.record("drop exact duplicate rows", df)

    df = collapse_song_ids(df)
    log.record("one row per song_id", df)

    df = collapse_rereleases(df)
    log.record("one row per artist + song name", df)

    df = drop_invalid_audio(df)
    log.record("drop failed audio analysis / <30s", df)

    df = add_features(df)
    df = assign_split(df)

    clean = df[OUTPUT_COLUMNS].sort_values("song_id").reset_index(drop=True)
    genres = genres[genres["song_id"].isin(clean["song_id"])].reset_index(drop=True)
    return clean, genres, log.to_frame()


def save_outputs(clean, genres, out_dir=PROCESSED_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_dir / "spotify_clean.csv", index=False)
    genres.to_csv(out_dir / "song_genres_long.csv", index=False)
