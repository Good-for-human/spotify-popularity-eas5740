# Spotify song popularity

Can we estimate a song's Spotify popularity from its audio features, and which features matter most?
This repo has the data preparation and exploratory analysis. The modeling notebooks start from
`data/processed/spotify_clean.csv`.

## What the data looks like

Raw file is 114,000 rows, pulled as 1,000 songs per genre. After cleaning we have 81,181 songs.
The big spike at popularity 0 is mostly duplicate releases, not songs nobody plays.

![popularity before and after cleaning](reports/figures/01_popularity_raw_vs_clean.png)

Songs at popularity 0 go from 14.1% to 5.8%. Mean popularity after cleaning is about 35, and only
about 10% of songs reach 61 or higher.

Audio features on their own barely correlate with popularity (strongest is instrumentalness, about
-0.19). The decile plot is more useful: energy, acousticness, valence and duration peak in the
middle and drop at both ends. A linear correlation averages that shape out and looks like zero.

![mean popularity by feature decile](reports/figures/04_popularity_by_decile.png)

Genre is the strong signal. Knowing the genre alone explains about 41% of the variance in
popularity. EDM, pop and k-pop sit near the top. Iranian and romance average under 4, which is
probably how those lists were filled, not a real statement about the genre.

![top and bottom genres by mean popularity](reports/figures/05_genre_popularity.png)

Once you compare songs only inside the same genre, the audio correlations mostly disappear. So a
lot of "instrumental songs are less popular" is really "instrumental genres are less popular".

![correlation with popularity, overall vs within genre](reports/figures/08_within_genre_correlation.png)

Top 10% songs (popularity >= 61) are more danceable, louder and more often explicit, and less
instrumental, live or acoustic. Tempo, energy and valence barely differ. Same caveat: a lot of
this is the genre mix.

![how top 10% songs differ from the rest](reports/figures/07_hits_vs_rest.png)

Explicit songs average about 4.5 points higher, instrumental songs about 8.6 lower. Songs listed
under several genres average about 10 points higher, but `n_genres` may just reflect how the file
was collected (a popular song is easier to pull under more than one genre).

![mean popularity by explicit, instrumental, genre count and artist count](reports/figures/06_flags_popularity.png)

Other charts, if you want them: feature distributions
(`reports/figures/02_feature_distributions.png`) and the audio correlation heatmap
(`reports/figures/03_feature_correlation.png`). energy, loudness and acousticness move together
(|r| about 0.6-0.76).

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

## Notes for modeling

- Use the `split` column (80/20, grouped by primary artist, seed 42). No artist is in both train and test.
- Baseline: predicting the train mean gives test MAE ~16.6.
- Feature set A = audio only (danceability, energy, loudness, speechiness, acousticness,
  instrumentalness, liveness, valence, tempo, duration_min, explicit, mode, key, time_signature).
  Set B = A + one-hot `song_genre`. Don't use song_id, song_name, album_name, artists or primary_artist.
- Compare A vs B. Use model A's feature importance when the advice is for artists, because genre
  will dominate model B.
- Tree models should beat plain linear regression because of the inverted-U shapes above.
  For linear models, scale the features and don't keep energy, loudness and acousticness all at once.
- Leave `n_genres` out of the main model and test it separately. Any artist- or genre-level
  popularity average has to be computed on the training folds only.

## Editing

The repo is public, so anyone can read it. To change files, fork the repo and open a pull request.
If you are on the team and want to push straight to `main`, you need to be added as a collaborator
(Write). GitHub does not allow anonymous pushes.
