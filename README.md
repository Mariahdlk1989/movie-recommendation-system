# \#Movie Recommendation System

# 

# A machine learning-based movie recommendation system that combines collaborative filtering, content-based filtering, and popularity signals to generate personalized movie recommendations.

# 

# The project uses the MovieLens dataset and implements several recommendation models, evaluation methods, and hybrid approaches, with a final hybrid model selected based on experimental evaluation.

# 

# Project Overview

# 

# The goal of this project is to build a personalized movie recommendation system capable of recommending movies that a user is likely to enjoy based on:

# 

# The user's previous ratings

# Similarity between movies

# Patterns learned from other users

# Movie popularity

# A combination of these signals

# 

# The project follows an experimental machine learning workflow:

# 

# Explore the dataset

# Preprocess the data

# Build recommendation models

# Experiment with different hybrid configurations

# Evaluate the models

# Compare the final model against a popularity baseline

# Select and document the final configuration

# Dataset

# 

# This project uses the MovieLens Latest Dataset.

# 

# The dataset used in the final experiments contains approximately:

# 

# Dataset Component	Size

# Movies	86,537

# Ratings	33,832,162

# Users	330,975

# 

# The rating data contains user-movie interactions with ratings and timestamps.

# 

# Recommendation Approach

# 

# The final system combines three main signals:

# 

# 1\. Collaborative Filtering

# 

# Collaborative filtering learns user and movie representations from the user-item rating matrix.

# 

# The project uses:

# 

# Sparse User-Item Matrix

# Truncated SVD

# 30 latent components

# 

# The user-item matrix used for training has approximately:

# 

# 322,397 users

# 83,101 movies

# 33,501,187 non-zero ratings

# 

# The SVD model produces:

# 

# User factor matrix: (322397, 30)

# Item factor matrix: (30, 83101)

# 

# The collaborative score is calculated from the interaction between the user's latent representation and the movie's latent representation.

# 

# 2\. Content-Based Filtering

# 

# The content-based component uses movie genres to determine similarity between movies.

# 

# Movie genres such as:

# 

# Action|Adventure|Sci-Fi

# 

# are converted into textual features:

# 

# Action Adventure Sci-Fi

# 

# TF-IDF is then applied to create movie feature vectors.

# 

# The final configuration uses:

# 

# TF-IDF features: 20,000

# 

# For each user, a content profile is constructed from movies the user rated positively.

# 

# Movies with ratings of 4 or higher are treated as liked movies when constructing the user profile.

# 

# 3\. Popularity

# 

# A popularity score is calculated using both:

# 

# Number of ratings

# Average rating

# 

# The final popularity score uses a logarithmic transformation of the rating count so that movies with extremely large numbers of ratings do not completely dominate the recommendation system.

# 

# Final Hybrid Model

# 

# The final recommendation score combines the three components using weighted scoring.

# 

# The selected configuration is:

# 

# Component	Weight

# Collaborative Filtering	0.90

# Content-Based Filtering	0.08

# Popularity	0.02

# 

# Therefore:

# 

# Hybrid Score =

# &#x20;   0.90 × Collaborative Score

# &#x20; + 0.08 × Content Score

# &#x20; + 0.02 × Popularity Score

# 

# The collaborative component has the largest contribution because experiments showed that user-item interaction patterns provide the strongest signal for personalization.

# 

# Recommendation Pipeline

# 

# The final model follows this pipeline:

# 

# MovieLens Dataset

# &#x20;       │

# &#x20;       ▼

# Train / Test Split

# &#x20;       │

# &#x20;       ├───────────────┐

# &#x20;       ▼               ▼

# &#x20;User-Item Matrix     Movie Genres

# &#x20;       │               │

# &#x20;       ▼               ▼

# &#x20;     SVD             TF-IDF

# &#x20;       │               │

# &#x20;       ▼               ▼

# &#x20;Collaborative       Content

# &#x20;   Scores            Scores

# &#x20;       │               │

# &#x20;       └───────┬───────┘

# &#x20;               │

# &#x20;               ▼

# &#x20;       Popularity Score

# &#x20;               │

# &#x20;               ▼

# &#x20;       Hybrid Scoring

# &#x20;               │

# &#x20;               ▼

# &#x20;       Candidate Pool

# &#x20;               │

# &#x20;               ▼

# &#x20;     Remove Watched Movies

# &#x20;               │

# &#x20;               ▼

# &#x20;      Top-N Recommendations

# Candidate Generation

# 

# The recommendation system first generates a candidate pool using the collaborative filtering scores.

# 

# The final configuration uses:

# 

# Candidate Pool: 300

# Top-N Recommendations: 10

# 

# Only the strongest collaborative candidates are considered for the final hybrid ranking.

# 

# Movies already rated by the user are removed before generating the final recommendations.

# 

# Train/Test Split

# 

# The evaluation uses a chronological split.

# 

# For each user:

# 

# Ratings are sorted by timestamp.

# The user's latest rating is placed in the test set.

# All previous ratings are used for training.

# 

# This creates:

# 

# Training ratings: 33,501,187

# Test ratings:        330,975

# 

# This approach prevents future interactions from being used to train the model when evaluating previous recommendations.

# 

# Model Evaluation

# 

# The recommendation system is evaluated using:

# 

# Precision@K

# Recall@K

# NDCG@K

# 

# A rating of:

# 

# rating >= 4

# 

# is considered a relevant interaction.

# 

# The final evaluation uses:

# 

# Evaluation users: 500

# K values: 5, 10, 20

# Final Evaluation Results

# 

# The final hybrid model was compared against a popularity-based baseline.

# 

# Model	K	Precision	Recall	NDCG

# Hybrid Final	5	0.0163	0.0815	0.0591

# Hybrid Final	10	0.0144	0.1442	0.0793

# Hybrid Final	20	0.0107	0.2132	0.0969

# Popularity Baseline	5	0.0094	0.0470	0.0246

# Popularity Baseline	10	0.0075	0.0752	0.0335

# Popularity Baseline	20	0.0055	0.1097	0.0424

# 

# The final hybrid model outperforms the popularity baseline across all evaluated K values and all three evaluation metrics.

# 

# Improvement at K=10

# 

# Compared with the popularity baseline:

# 

# Precision improves from 0.0075 → 0.0144

# Recall improves from 0.0752 → 0.1442

# NDCG improves from 0.0335 → 0.0793

# 

# This shows that the hybrid model provides substantially better personalized ranking than simply recommending popular movies.

# 

# Example Recommendations

# 

# For an example user, the final system generated recommendations such as:

# 

# Movie	Collaborative	Content	Popularity	Hybrid

# Raiders of the Lost Ark	0.6225	0.6065	0.8910	0.6266

# Finding Nemo	0.4503	0.6302	0.7970	0.4717

# Back to the Future	0.4234	0.5188	0.8518	0.4396

# Jurassic Park	0.3896	0.5289	0.8079	0.4091

# Terminator 2: Judgment Day	0.3932	0.3044	0.8571	0.3954

# 

# The system combines the different scores rather than relying only on popularity or genre similarity.

# 

# Project Structure

# movie-recommendation-system/

# │

# ├── README.md

# ├── requirements.txt

# ├── .gitignore

# │

# └── ml/

# &#x20;   │

# &#x20;   ├── data/

# &#x20;   │   ├── movies.csv

# &#x20;   │   ├── ratings.csv

# &#x20;   │   ├── movies\_processed.csv

# &#x20;   │   └── movies\_with\_stats.csv

# &#x20;   │

# &#x20;   ├── explore\_data.py

# &#x20;   ├── preprocess.py

# &#x20;   ├── movie\_statistics.py

# &#x20;   │

# &#x20;   ├── recommendation.py

# &#x20;   ├── collaborative\_filtering.py

# &#x20;   ├── collaborative\_v2.py

# &#x20;   │

# &#x20;   ├── hybrid\_model.py

# &#x20;   ├── hybrid\_recommendation.py

# &#x20;   ├── hybrid\_v2.py

# &#x20;   ├── hybrid\_v3.py

# &#x20;   ├── hybrid\_v4.py

# &#x20;   ├── hybrid\_v5.py

# &#x20;   ├── hybrid\_v6.py

# &#x20;   ├── hybrid\_v7.py

# &#x20;   │

# &#x20;   ├── hybrid\_evaluation.py

# &#x20;   ├── hybrid\_v2\_evaluation.py

# &#x20;   ├── hybrid\_v3\_evaluation.py

# &#x20;   ├── hybrid\_v4\_evaluation.py

# &#x20;   ├── hybrid\_v5\_evaluation.py

# &#x20;   ├── collaborative\_evaluation.py

# &#x20;   ├── evaluation.py

# &#x20;   │

# &#x20;   ├── hybrid\_final.py

# &#x20;   ├── hybrid\_final\_evaluation.py

# &#x20;   ├── final\_evaluation.py

# &#x20;   │

# &#x20;   ├── popularity\_baseline.py

# &#x20;   ├── tune\_hybrid.py

# &#x20;   ├── tune\_svd.py

# &#x20;   │

# &#x20;   └── hybrid\_tuning\_results.csv

# Installation

# 

# Clone the repository:

# 

# git clone https://github.com/Mariahdlk1989/movie-recommendation-system.git

# 

# Move into the project directory:

# 

# cd movie-recommendation-system

# 

# Create a virtual environment:

# 

# python -m venv .venv

# 

# Activate it on Windows PowerShell:

# 

# .venv\\Scripts\\Activate.ps1

# 

# Install the dependencies:

# 

# pip install -r requirements.txt

# Running the Final Recommendation Model

# 

# From the project root, run:

# 

# python ml/hybrid\_final.py

# 

# The model will:

# 

# Load the MovieLens dataset

# Create the chronological train/test split

# Build the sparse user-item matrix

# Train the SVD model

# Build the TF-IDF content model

# Calculate popularity scores

# Generate personalized recommendations

# Display the final hybrid scores

# Running the Evaluation

# 

# To evaluate the final model, run:

# 

# python ml/final\_evaluation.py

# 

# The evaluation reports:

# 

# Precision@5

# Recall@5

# NDCG@5

# 

# Precision@10

# Recall@10

# NDCG@10

# 

# Precision@20

# Recall@20

# NDCG@20

# 

# The evaluation also compares the final hybrid system against the popularity baseline.

# 

# Model Configuration

# 

# The final configuration is:

# 

# Collaborative Weight: 0.90

# Content Weight:       0.08

# Popularity Weight:    0.02

# 

# SVD Components:       30

# TF-IDF Features:      20,000

# Candidate Pool:       300

# Top N:                10

# Evaluation Users:     500

# Relevant Threshold:   rating >= 4

# Experimental Development

# 

# The project contains multiple versions of the recommendation model.

# 

# The development process included experiments with:

# 

# Collaborative filtering

# Content-based recommendation

# Popularity-based recommendation

# Different hybrid weight configurations

# Different SVD configurations

# Candidate pool sizes

# Multiple evaluation approaches

# 

# The files with v2, v3, v4, v5, v6, and v7 represent different experimental versions.

# 

# The hybrid\_final.py model represents the selected final configuration used for the final evaluation.

# 

# Limitations

# 

# Although the final hybrid model performs better than the popularity baseline, several limitations remain.

# 

# Sparse User-Movie Interactions

# 

# Most users have rated only a small subset of the available movies. This creates a highly sparse user-item matrix.

# 

# Limited Content Features

# 

# The current content-based model primarily uses movie genres. Additional features such as:

# 

# Movie descriptions

# Directors

# Actors

# Keywords

# Release year

# 

# could improve content-based recommendations.

# 

# Cold Start Problem

# 

# New users with no rating history cannot receive highly personalized collaborative recommendations.

# 

# Similarly, movies with no interaction history are difficult to rank using collaborative filtering.

# 

# Evaluation Size

# 

# The final evaluation uses a sample of 500 users rather than the entire user population. A larger evaluation could provide a more comprehensive estimate of model performance.

# 

# Future Improvements

# 

# Possible future improvements include:

# 

# Add movie descriptions to the content model

# Add actors and directors as features

# Implement more advanced collaborative filtering

# Experiment with implicit-feedback recommendation

# Use neural recommendation models

# Improve cold-start handling

# Add user and movie embeddings

# Tune the hybrid weights automatically

# Build a FastAPI backend

# Add a React frontend

# Add PostgreSQL integration

# Containerize the complete application using Docker

# Deploy the recommendation service

# Technologies

# 

# The machine learning component uses:

# 

# Python

# Pandas

# NumPy

# SciPy

# Scikit-learn

# Truncated SVD

# TF-IDF

# Sparse matrices

# 

# The planned full-stack architecture can additionally use:

# 

# FastAPI

# PostgreSQL

# React

# Docker

# Project Status

# 

# Machine Learning Recommendation Engine: Completed

# 

# The final hybrid recommendation model has been implemented and evaluated against a popularity baseline.

# 

# The full-stack application components are planned for future development.

# 

# Author

# 

# Maria Delkash

# 

# Computer Engineering Student

# 

# Machine Learning \& Recommendation Systems Project

