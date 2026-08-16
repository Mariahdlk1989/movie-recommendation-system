# 🎬 Movie Recommendation System

A machine learning-based movie recommendation system that combines **collaborative filtering, content-based filtering, and popularity signals** to generate personalized movie recommendations.

The project uses the **MovieLens dataset** and implements an end-to-end recommendation pipeline using Python and Scikit-learn.

---

## 📌 Project Overview

The goal of this project is to build a personalized movie recommendation system capable of learning users' preferences from their previous ratings and recommending movies they are likely to enjoy.

The final system combines multiple recommendation signals:

* 🤝 Collaborative Filtering
* 🎭 Content-Based Filtering
* ⭐ Popularity
* 🔀 Hybrid Recommendation

The final model was evaluated against a popularity-based baseline to measure its recommendation quality.

---

## 🧠 Recommendation Architecture

The final recommendation score is calculated using a weighted hybrid approach:

```text
Hybrid Score =
    0.90 × Collaborative Score
  + 0.08 × Content Score
  + 0.02 × Popularity Score
```

### 1. Collaborative Filtering

Collaborative filtering learns hidden relationships between users and movies from the rating matrix.

The implementation uses:

* Sparse User-Item Matrix
* Truncated SVD
* 30 latent components

The SVD model learns latent representations for both users and movies and uses them to estimate user preferences.

### 2. Content-Based Filtering

The content-based component uses movie genres to identify movies that are similar to those previously liked by a user.

The implementation uses:

* TF-IDF Vectorization
* Movie genres as features
* User preference profiles
* Cosine-style similarity through TF-IDF vectors

### 3. Popularity

A popularity score is calculated using both:

* Number of ratings
* Average rating

This signal provides a small contribution to the final recommendation score.

### 4. Hybrid Model

The three signals are combined into a single score.

The final configuration is:

| Component               | Weight |
| ----------------------- | -----: |
| Collaborative Filtering |   0.90 |
| Content-Based Filtering |   0.08 |
| Popularity              |   0.02 |

The collaborative component has the largest weight because it provided the strongest personalization signal during experimentation.

---

## 📊 Dataset

The project uses the **MovieLens Latest Dataset**.

Dataset statistics used in the final experiment:

| Dataset |      Count |
| ------- | ---------: |
| Movies  |     86,537 |
| Ratings | 33,832,162 |
| Users   |    330,975 |

The dataset contains movie metadata and millions of user ratings.

---

## 🧪 Train/Test Strategy

The evaluation uses a **leave-one-out temporal split**.

For each user:

* Ratings are sorted by timestamp.
* The user's latest rating is placed in the test set.
* Earlier ratings are used for training.

This resulted in:

```text
Users:              330,975
Training ratings:  33,501,187
Test ratings:         330,975
```

For evaluation, a movie is considered relevant when:

```text
rating >= 4
```

---

## 📈 Final Evaluation Results

The final hybrid model was evaluated on **500 users** and compared against a popularity baseline.

| Model               |  K | Precision | Recall |   NDCG |
| ------------------- | -: | --------: | -----: | -----: |
| Hybrid Final        |  5 |    0.0163 | 0.0815 | 0.0591 |
| Hybrid Final        | 10 |    0.0144 | 0.1442 | 0.0793 |
| Hybrid Final        | 20 |    0.0107 | 0.2132 | 0.0969 |
| Popularity Baseline |  5 |    0.0094 | 0.0470 | 0.0246 |
| Popularity Baseline | 10 |    0.0075 | 0.0752 | 0.0335 |
| Popularity Baseline | 20 |    0.0055 | 0.1097 | 0.0424 |

### 🏆 Key Result

The final hybrid model outperformed the popularity baseline across **Precision, Recall, and NDCG** for all evaluated values of K.

For example, at **K = 10**:

```text
                    Hybrid       Baseline
Precision@10        0.0144        0.0075
Recall@10           0.1442        0.0752
NDCG@10             0.0793        0.0335
```

This demonstrates that the hybrid model provides a stronger personalized recommendation signal than simply recommending popular movies.

---

## ⚙️ Final Model Configuration

```text
Collaborative Weight: 0.90
Content Weight:       0.08
Popularity Weight:    0.02

SVD Components:       30
TF-IDF Features:      20,000
Candidate Pool:       300
Top N:                10
Evaluation Users:     500
Relevant Threshold:   rating >= 4
```

---

## 📁 Project Structure

```text
movie-recommendation-system/
│
├── ml/
│   │
│   ├── data/
│   │   ├── movies.csv
│   │   ├── ratings.csv
│   │   ├── movies_processed.csv
│   │   └── movies_with_stats.csv
│   │
│   ├── hybrid_model.py
│   ├── hybrid_recommendation.py
│   ├── hybrid_final.py
│   ├── final_evaluation.py
│   ├── hybrid_final_evaluation.py
│   │
│   ├── collaborative_filtering.py
│   ├── recommendation.py
│   ├── popularity_baseline.py
│   ├── preprocess.py
│   ├── explore_data.py
│   └── movie_statistics.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Model Development Files

Several experimental versions of the hybrid model were developed during tuning and experimentation.

Files such as:

```text
hybrid_v2.py
hybrid_v3.py
hybrid_v4.py
hybrid_v5.py
hybrid_v6.py
hybrid_v7.py
```

represent intermediate development stages and are not part of the final recommendation pipeline.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Mariahdlk1989/movie-recommendation-system.git
cd movie-recommendation-system
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Final Recommendation Model

From the project root:

```bash
python ml/hybrid_final.py
```

The program:

1. Loads the MovieLens dataset.
2. Creates the training/test split.
3. Builds the sparse user-item matrix.
4. Trains the SVD collaborative model.
5. Builds the TF-IDF content model.
6. Calculates movie popularity.
7. Generates personalized recommendations.
8. Displays the final hybrid scores.

Example:

```text
============================================================
FINAL RECOMMENDATIONS
============================================================

Raiders of the Lost Ark (Indiana Jones and the Raiders
of the Lost Ark) (1981)

Finding Nemo (2003)

Back to the Future (1985)

Jurassic Park (1993)

Terminator 2: Judgment Day (1991)
```

---

## 📊 Running the Evaluation

To evaluate the final model:

```bash
python ml/final_evaluation.py
```

The evaluation compares the hybrid recommendation system with a popularity baseline using:

* Precision@K
* Recall@K
* NDCG@K

The tested values of K are:

```text
K = 5
K = 10
K = 20
```

---

## 🛠️ Technologies

* **Python**
* **NumPy**
* **Pandas**
* **SciPy**
* **Scikit-learn**
* **Truncated SVD**
* **TF-IDF**
* **Git / GitHub**

---

## 🔬 Future Improvements

Possible future improvements include:

* Adding movie descriptions and tags to the content model
* Incorporating movie metadata such as directors and actors
* Using a more advanced collaborative filtering algorithm
* Testing Neural Collaborative Filtering
* Using implicit feedback
* Improving cold-start recommendations
* Adding a FastAPI backend
* Adding a React frontend
* Adding PostgreSQL for application data
* Containerizing the application with Docker
* Deploying the recommendation service

---

## 📌 Project Status

**Current status: Machine Learning recommendation engine completed.**

The final hybrid recommendation model has been implemented and evaluated against a popularity baseline.

Future development will focus on integrating the recommendation engine into a complete full-stack application.

---

## 👩‍💻 Author

**Maria Delkash**

GitHub: [Mariahdlk1989](https://github.com/Mariahdlk1989)

---

## 📄 License

This project is developed for educational and research purposes.

