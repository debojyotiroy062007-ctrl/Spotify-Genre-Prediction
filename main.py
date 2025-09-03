# spotify_genre_project.py
# ------------------------------------------------------------
# Spotify Songs’ Genre Segmentation
# Preprocessing -> EDA -> Correlation -> Clustering -> Classification
# ------------------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# -------------------------
# Paths & Setup
# -------------------------
DATA_FILE = "spotify dataset.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv(DATA_FILE)
df = df.drop_duplicates()

# Release year
if "track_album_release_date" in df.columns:
    df["release_year"] = df["track_album_release_date"].astype(str).str[:4].astype(int, errors="ignore")

# -------------------------
# Features & Target
# -------------------------
drop_cols = ["track_id","track_name","track_artist","track_album_id","track_album_name","playlist_id","playlist_name"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

features = [
    "danceability","energy","key","loudness","mode","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms","track_popularity","release_year"
]
features = [c for c in features if c in df.columns]
target = "playlist_genre"

df = df.dropna(subset=[target]+features)

# -------------------------
# EDA
# -------------------------
plt.figure(figsize=(9,5))
sns.countplot(data=df, x=target, order=df[target].value_counts().index)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_class_balance.png"))
plt.close()

num_cols = [c for c in features if df[c].dtype != "O"]
fig, axes = plt.subplots(nrows=int(np.ceil(len(num_cols)/4)), ncols=4, figsize=(18,12))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
for j in range(i+1, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_feature_distributions.png"))
plt.close()

plt.figure(figsize=(12,10))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_correlation_matrix.png"))
plt.close()

# -------------------------
# Clustering
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=["PC1","PC2"])
pca_df[target] = df[target].values

best_k, best_sil = None, -1
for k in range(3, 13):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    sil = silhouette_score(X_scaled, km.fit_predict(X_scaled))
    if sil > best_sil:
        best_sil, best_k = sil, k

kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init="auto")
pca_df["cluster"] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=12, linewidth=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_pca_clusters.png"))
plt.close()

# -------------------------
# Classification
# -------------------------
X, y = df[features], df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

models = {
    "LogisticRegression": Pipeline([("scaler", StandardScaler()),("clf", LogisticRegression(max_iter=2000))]),
    "SVM": Pipeline([("scaler", StandardScaler()),("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
    "KNN": Pipeline([("scaler", StandardScaler()),("clf", KNeighborsClassifier(n_neighbors=15))]),
    "RandomForest": Pipeline([("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))])
}

results, best_model, best_name, best_acc = [], None, None, -1
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append([name, acc])
    if acc > best_acc:
        best_acc, best_name, best_model = acc, name, model

res_df = pd.DataFrame(results, columns=["Model","Accuracy"]).sort_values("Accuracy", ascending=False)
res_df.to_csv(os.path.join(OUTPUT_DIR, "05_model_performance.csv"), index=False)

report = classification_report(y_test, best_model.predict(X_test))
with open(os.path.join(OUTPUT_DIR, "06_classification_report.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, best_model.predict(X_test), labels=y_test.unique(), normalize="true")
cm_df = pd.DataFrame(cm, index=y_test.unique(), columns=y_test.unique())
plt.figure(figsize=(10,8))
sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_confusion_matrix.png"))
plt.close()

# Feature importance if RF
if best_name == "RandomForest":
    importances = pd.Series(best_model.named_steps["clf"].feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(8,6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_feature_importance.png"))
    plt.close()
    importances.to_csv(os.path.join(OUTPUT_DIR, "08_feature_importance.csv"))

# -------------------------
# Save final model
# -------------------------
joblib.dump(best_model, os.path.join(OUTPUT_DIR, f"best_model_{best_name}.joblib"))

