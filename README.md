\# Spotify Songs Genre Segmentation 🎵



\## Project Overview

This project focuses on analyzing and classifying Spotify songs based on their audio and metadata features.  

The goal is to explore patterns in music data, perform clustering for genre grouping, and build machine learning models to predict playlist genres.  

The workflow demonstrates how such an approach can serve as the foundation of a music recommendation system.



\## Objectives

\- Perform data preprocessing and cleaning.

\- Conduct Exploratory Data Analysis (EDA) with visualizations.

\- Generate and analyze the correlation matrix of features.

\- Apply clustering (PCA + KMeans) to identify natural groupings of songs.

\- Train and evaluate classification models to predict playlist genres.

\- Compare model performances and select the best one.

\- Save the trained model for future predictions.



\## Dataset

\- \*\*Source\*\*: Provided dataset containing 32,833 tracks with 23 attributes.  

\- \*\*Target Variable\*\*: `playlist\_genre`  

\- \*\*Features Used\*\*:  

&nbsp; Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness,  

&nbsp; Liveness, Valence, Tempo, Duration (ms), Track Popularity, Release Year  



\*(If the dataset cannot be shared publicly, please obtain it from the course instructor.)\*



\## Methodology

1\. \*\*Data Preprocessing\*\*  

&nbsp;  - Removal of duplicates and handling of missing values.  

&nbsp;  - Feature selection and conversion of release dates to release years.  



2\. \*\*Exploratory Data Analysis (EDA)\*\*  

&nbsp;  - Distribution plots for audio features.  

&nbsp;  - Class balance visualization for genres.  

&nbsp;  - Correlation heatmap of features.  



3\. \*\*Clustering\*\*  

&nbsp;  - Standardization of audio features.  

&nbsp;  - PCA for dimensionality reduction.  

&nbsp;  - KMeans clustering with silhouette score optimization.  



4\. \*\*Classification Models\*\*  

&nbsp;  - Logistic Regression  

&nbsp;  - K-Nearest Neighbors (KNN)  

&nbsp;  - Support Vector Machine (SVM)  

&nbsp;  - Random Forest  



&nbsp;  Models were compared on \*\*Accuracy, Precision, Recall, and F1-score\*\*.  



5\. \*\*Model Selection\*\*  

&nbsp;  - Random Forest achieved the highest overall performance.  

&nbsp;  - Feature importance was analyzed to understand key contributors.  

&nbsp;  - The final model was saved using Joblib for future use.  



\## Results

\- \*\*Best Model\*\*: Random Forest Classifier  

\- \*\*Outputs\*\*:  

&nbsp; - Plots: genre distribution, feature distributions, correlation heatmap, PCA clusters, confusion matrix, feature importance.  

&nbsp; - Reports: classification report, model performance summary.  

&nbsp; - Saved model: `best\_model\_RandomForest.joblib`  



\## How to Run

1\. Install dependencies:

&nbsp;  ```bash

&nbsp;  pip install numpy pandas matplotlib seaborn scikit-learn joblib



