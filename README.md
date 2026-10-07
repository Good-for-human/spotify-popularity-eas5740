# Spotify song popularity (EAS 5740 final project, Track 2)

Can we estimate a song's Spotify popularity from its audio features, and which features matter most?
This repo has the data preparation and exploratory analysis. The modeling notebooks start from
`data/processed/spotify_clean.csv`.

## Layout

```
data/raw/dataset.csv                     Kaggle "Spotify Tracks Dataset", unmodified
data/processed/spotify_clean.csv         one row per unique song, with train/test split
data/processed/song_genres_long.csv      every (song_id, genre) pair, for multi-genre encoding
src/data_prep.py                         cleaning pipeline and derived features
src/plots.py                             plotting functions for the EDA notebook
notebooks/01_data_preparation.ipynb      problems in the raw file and how we fixed them
notebooks/02_exploratory_analysis.ipynb  patterns in the data and what they mean for modeling
reports/figures/                         EDA charts (PNG) for the slides
```

## Reproduce

```
pip install -r requirements.txt
jupyter notebook notebooks/01_data_preparation.ipynb      # writes data/processed/
jupyter notebook notebooks/02_exploratory_analysis.ipynb  # writes reports/figures/
```

The raw file is from [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
(also in the course Google Drive folder). We rename `track_id` -> `song_id`, `track_name` -> `song_name`
and `track_genre` -> `song_genre` to match the assignment brief.

## Cleaning steps

| step | rows left | why |
|---|---|---|
| raw file | 114,000 | |
| drop missing artist/album/name | 113,999 | one unusable row |
| drop exact duplicates | 113,549 | |
| one row per `song_id` | 89,740 | same song listed under several genres |
| one row per artist + title | 81,343 | single/album/compilation copies of the same recording, most at popularity 0 |
| drop failed audio analysis and <30 s tracks | 81,181 | tempo/time signature of 0, too short to count as a stream |

Songs at popularity 0 go from 14.1% to 5.8% after cleaning.

## Notes for modeling

- Use the `split` column (80/20, grouped by primary artist, seed 42). No artist is in both train and test.
- Baseline: predicting the train mean gives test MAE ~16.6.
- Feature set A = audio only (danceability, energy, loudness, speechiness, acousticness,
  instrumentalness, liveness, valence, tempo, duration_min, explicit, mode, key, time_signature).
  Set B = A + one-hot `song_genre`. Don't use song_id, song_name, album_name, artists or primary_artist.
- Genre alone explains ~41% of the variance, and within a genre the audio correlations mostly go away.
  Compare A vs B, and use model A's feature importance for advice to artists.
- energy, acousticness, valence and duration have an inverted-U relationship with popularity, so tree
  models should beat linear regression. energy/loudness/acousticness are also highly correlated
  (|r| ~0.6-0.76).
- `n_genres` may be a side effect of how the data was collected, test it separately. Any artist- or
  genre-level popularity average used as a feature has to be computed on the training folds only.
