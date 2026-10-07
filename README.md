# Spotify song popularity (EAS 5740 final project, Track 2)

Can we estimate a song's Spotify popularity from its audio features, and which features matter most?
This repo covers data preparation and exploratory analysis. The modeling notebooks build on
`data/processed/spotify_clean.csv`.

## Layout

```
data/raw/dataset.csv                 Kaggle "Spotify Tracks Dataset", unmodified
data/processed/spotify_clean.csv     one row per unique song, model-ready, with train/test split
data/processed/song_genres_long.csv  every (song_id, genre) pair, for multi-genre encoding
src/data_prep.py                     cleaning pipeline and derived features
src/plots.py                         figure helpers used by the EDA notebook
notebooks/01_data_preparation.ipynb  what is wrong with the raw file and how we fixed it
notebooks/02_exploratory_analysis.ipynb  patterns in the data and what they mean for modeling
reports/figures/                     all EDA charts as PNG, ready for slides
```

## Reproduce

```
pip install -r requirements.txt
jupyter notebook notebooks/01_data_preparation.ipynb   # writes data/processed/
jupyter notebook notebooks/02_exploratory_analysis.ipynb   # writes reports/figures/
```

The raw file can be re-downloaded from
[Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) or the course
Google Drive folder. Column names are renamed to match the assignment brief
(`track_id` → `song_id`, `track_name` → `song_name`, `track_genre` → `song_genre`).

## Cleaning in short

| step | rows left | why |
|---|---|---|
| raw file | 114,000 | |
| drop missing artist/album/name | 113,999 | one unusable row |
| drop exact duplicates | 113,549 | |
| one row per `song_id` | 89,740 | same song listed under several genres |
| one row per artist + title | 81,343 | single/album/compilation copies of one recording, most stuck at popularity 0 |
| drop failed audio analysis and <30 s tracks | 81,181 | tempo/time signature of 0, too short to count as a stream |

After cleaning, songs at popularity 0 drop from 14.1% to 5.8%.

## Notes for modeling

- **Split:** use the `split` column. It is 80/20 and grouped by primary artist (seed 42), so no
  artist appears in both train and test.
- **Baseline:** predicting the training mean gives test MAE ≈ 16.6.
- **Feature sets:** A = audio only (danceability, energy, loudness, speechiness, acousticness,
  instrumentalness, liveness, valence, tempo, duration_min, explicit, mode, key, time_signature);
  B = A + one-hot `song_genre`. Optional: n_artists, is_instrumental, is_live, n_genres.
- **Not inputs:** song_id, song_name, album_name, artists, primary_artist.
- **Genre explains ~41% of popularity variance on its own**, and within a genre the audio
  correlations nearly vanish. Compare A vs B, and use model A's feature importance when advising
  artists.
- **Several features relate to popularity in an inverted U** (energy, acousticness, valence,
  duration), so expect tree-based models to beat linear regression.
- `energy`, `loudness` and `acousticness` are strongly correlated (|r| ≈ 0.6–0.76). Regularise or
  pick one for linear models.
- `n_genres` is strongly associated with popularity but may be a side effect of how the data was
  collected. Test it separately rather than including it by default.
- Any artist- or genre-level average of popularity used as a feature must be computed on training
  folds only.
