"""Plot helpers shared by the EDA notebook. Each function draws one figure and saves it."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

ACCENT = "#1DB954"  # Spotify green, used for the series we want the eye on
MUTED = "#9AA0A6"
DARK = "#191414"


def set_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")


def popularity_before_after(raw, clean):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, data, label, color in [
        (axes[0], raw["popularity"], f"Raw file (n={len(raw):,})", MUTED),
        (axes[1], clean["popularity"], f"After cleaning (n={len(clean):,})", ACCENT),
    ]:
        ax.hist(data, bins=50, color=color, edgecolor="white")
        zero_share = (data == 0).mean()
        ax.set_title(f"{label}\n{zero_share:.1%} of songs at popularity 0")
        ax.set_xlabel("popularity")
    axes[0].set_ylabel("songs")
    save(fig, "01_popularity_raw_vs_clean")
    return fig


def feature_distributions(df, features):
    n_cols = 4
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2.8 * n_rows))
    for ax, feat in zip(axes.flat, features):
        ax.hist(df[feat], bins=40, color=ACCENT, edgecolor="white")
        ax.set_title(feat)
        ax.set_ylabel("")
    for ax in axes.flat[len(features):]:
        ax.set_visible(False)
    fig.suptitle("Distribution of audio features (cleaned data)", fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "02_feature_distributions")
    return fig


def correlation_heatmap(df, features):
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=ax,
    )
    ax.set_title("Correlation between audio features")
    save(fig, "03_feature_correlation")
    return fig


def decile_profile(df, features, target="popularity"):
    """Mean popularity per decile of each feature. Shows shapes a single correlation hides."""
    n_cols = 5
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows), sharey=True)
    overall = df[target].mean()
    for ax, feat in zip(axes.flat, features):
        deciles = pd.qcut(df[feat].rank(method="first"), 10, labels=False) + 1
        means = df.groupby(deciles)[target].mean()
        ax.plot(means.index, means.values, marker="o", color=ACCENT, lw=2)
        ax.axhline(overall, color=MUTED, ls="--", lw=1)
        ax.set_title(feat)
        ax.set_xticks([1, 5, 10])
        ax.set_xlabel("decile (low → high)")
    for ax in axes.flat[len(features):]:
        ax.set_visible(False)
    axes.flat[0].set_ylabel(f"mean {target}")
    fig.suptitle(
        "Mean popularity across feature deciles (dashed line = overall mean)",
        fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save(fig, "04_popularity_by_decile")
    return fig


def genre_ranking(df, n=15, target="popularity"):
    stats = df.groupby("song_genre")[target].agg(["mean", "size"])
    stats = stats[stats["size"] >= 100].sort_values("mean")
    picked = pd.concat([stats.head(n), stats.tail(n)])
    colors = [MUTED] * n + [ACCENT] * n

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(picked.index, picked["mean"], color=colors)
    ax.axvline(df[target].mean(), color=DARK, ls="--", lw=1)
    ax.set_xlabel(f"mean {target}")
    ax.set_title(f"Top and bottom {n} genres by mean popularity")
    save(fig, "05_genre_popularity")
    return fig


def flag_comparison(df, target="popularity"):
    """Mean popularity with 95% CI for a few binary / count attributes."""
    panels = [
        ("explicit", df["explicit"].map({0: "clean", 1: "explicit"})),
        ("instrumental (>0.5)", df["is_instrumental"].map({0: "vocal", 1: "instrumental"})),
        ("genres listed under", df["n_genres"].clip(upper=4).astype(str).replace({"4": "4+"})),
        ("credited artists", df["n_artists"].clip(upper=4).astype(str).replace({"4": "4+"})),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.6), sharey=True)
    for ax, (title, groups) in zip(axes, panels):
        sns.barplot(
            x=groups, y=df[target], order=sorted(groups.unique()),
            color=ACCENT, errorbar=("ci", 95), ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
    axes[0].set_ylabel(f"mean {target}")
    fig.tight_layout()
    save(fig, "06_flags_popularity")
    return fig


def hits_vs_rest(df, features, quantile=0.9, target="popularity"):
    """Standardised mean difference between the top decile of songs and everyone else."""
    threshold = df[target].quantile(quantile)
    is_hit = df[target] >= threshold
    z = (df[features] - df[features].mean()) / df[features].std()
    diff = (z[is_hit].mean() - z[~is_hit].mean()).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(diff.index, diff.values, color=[ACCENT if v > 0 else MUTED for v in diff.values])
    ax.axvline(0, color=DARK, lw=1)
    ax.set_xlabel("difference in means (standard deviations)")
    ax.set_title(f"How top-10% songs (popularity ≥ {threshold:.0f}) differ from the rest")
    save(fig, "07_hits_vs_rest")
    return fig, diff


def overall_vs_within_genre(df, features, target="popularity"):
    """Correlation with the target before and after removing each genre's mean."""
    cols = features + [target]
    overall = df[cols].corr()[target].drop(target)
    demeaned = df[cols] - df.groupby("song_genre")[cols].transform("mean")
    within = demeaned.corr()[target].drop(target)
    table = pd.DataFrame({"overall": overall, "within genre": within}).sort_values("overall")

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(table))
    ax.barh(y - 0.2, table["overall"], height=0.4, color=MUTED, label="across all songs")
    ax.barh(y + 0.2, table["within genre"], height=0.4, color=ACCENT, label="within the same genre")
    ax.set_yticks(y, table.index)
    ax.axvline(0, color=DARK, lw=1)
    ax.set_xlabel(f"Pearson correlation with {target}")
    ax.set_title("Most of the feature–popularity link runs through genre")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "08_within_genre_correlation")
    return fig, table
