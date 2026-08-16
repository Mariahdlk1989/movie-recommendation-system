# 🎬 Movie Recommendation System

A machine learning-based movie recommendation system that combines **collaborative filtering, content-based filtering, and popularity signals** to generate personalized movie recommendations.

The project uses the **MovieLens Latest Dataset** and implements an end-to-end recommendation pipeline using Python and Scikit-learn.

---

## 📌 Project Overview

The goal of this project is to build a personalized movie recommendation system capable of learning users' preferences from their previous ratings and recommending movies they are likely to enjoy.

The final system combines three recommendation signals:

* 🤝 Collaborative Filtering
* 🎭 Content-Based Filtering
* ⭐ Popularity
* 🔀 Hybrid Recommendation

The final hybrid model was evaluated against a popularity-based baseline using standard recommendation-system metrics.

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

The collaborative filtering component learns hidden relationships between users and movies from the user-item rating matrix.

The implementation uses:

* Sparse User-Item Matrix
* Truncated SVD
* 30 latent components

The SVD model learns latent representations of users and movies and uses these representations to estimate user preferences.

### 2. Content-Based Filtering

The content-based component identifies movies with genres similar to movies that a user previously liked.

The implementation uses:

* TF-IDF Vectorization
* Movie genres as features
* User preference profiles
* Cosine-style similarity using TF-IDF vectors

### 3. Popularity

A popularity score is calculated using:

* Number of ratings
* Average rating

Popularity contributes only a small portion of the final score so that the model remains primarily personalized.

### 4. Hybrid Model

The three recommendation signals are combined into a single hybrid score.

| Component               | Weight |
| ----------------------- | -----: |
| Collaborative Filtering |   0.90 |
| Content-Based Filtering |   0.08 |
| Popularity              |   0.02 |

The collaborative component has the highest weight because it provided the strongest personalization signal during experimentation.

---

## 📊 Dataset

The project uses the **MovieLens Latest Dataset**.

Dataset statistics used in the final experiment:

| Dataset |      Count |
| ------- | ---------: |
| Movies  |     86,537 |
| Ratings | 33,832,162 |
| Users   |    330,975 |

The dataset contains movie metadata and more than 33 million user ratings.

---

## 🧪 Train/Test Strategy

The evaluation uses a **leave-one-out temporal split**.

For each user:

1. Ratings are sorted chronologically using timestamps.
2. The user's latest rating is placed in the test set.
3. Earlier ratings are used as training data.

The resulting split contains:

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

The final hybrid model was evaluated on **500 users** and compared against a popularity-based baseline.

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

At **K = 10**:

| Metric       | Hybrid | Baseline |
| ------------ | -----: | -------: |
| Precision@10 | 0.0144 |   0.0075 |
| Recall@10    | 0.1442 |   0.0752 |
| NDCG@10      | 0.0793 |   0.0335 |

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
│   ├── hybrid_final.py
│   ├── final_evaluation.py
│   │
│   ├── collaborative_filtering.py
│   ├── collaborative_evaluation.py
│   ├── recommendation.py
│   ├── popularity_baseline.py
│   ├── preprocess.py
│   ├── explore_data.py
│   └── movie_statistics.py
│
│   └── experiments/
│       ├── hybrid_model.py
│       ├── hybrid_recommendation.py
│       ├── hybrid_evaluation.py
│       ├── hybrid_v2.py
│       ├── hybrid_v2_evaluation.py
│       ├── hybrid_v3.py
│       ├── hybrid_v3_evaluation.py
│       ├── hybrid_v4.py
│       ├── hybrid_v4_evaluation.py
│       ├── hybrid_v5.py
│       ├── hybrid_v5_evaluation.py
│       ├── hybrid_v6.py
│       ├── hybrid_v7.py
│       ├── hybrid_final_evaluation.py
│       ├── hybrid_tuning_results.csv
│       ├── tune_hybrid.py
│       └── tune_svd.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Model Development Files

The `ml/experiments/` directory contains intermediate models and experiments developed during the tuning process.

These files include different versions of the hybrid recommendation model, evaluation scripts, and hyperparameter tuning experiments.

The main production-oriented files are:

```text
ml/hybrid_final.py
ml/final_evaluation.py
```

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
2. Creates the temporal train/test split.
3. Builds the sparse user-item matrix.
4. Trains the SVD collaborative filtering model.
5. Builds the TF-IDF content model.
6. Calculates movie popularity.
7. Generates personalized recommendations.
8. Calculates the final hybrid score.
9. Displays the top recommendations.

Example recommendations include:

```text
Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)
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

The evaluated values are:

```text
K = 5
K = 10
K = 20
```

---

## 🛠️ Technologies

* Python
* NumPy
* Pandas
* SciPy
* Scikit-learn
* Truncated SVD
* TF-IDF
* Git
* GitHub

---

## 🔮 Future Improvements

Possible future improvements include:

* Adding movie descriptions and tags to the content model
* Incorporating directors and actors into the content model
* Using more advanced collaborative filtering algorithms
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

The next development stage is to integrate the recommendation engine into a complete full-stack application.

---

## 👩‍💻 Author

**Maria Delkash**

GitHub: [Mariahdlk1989](https://github.com/Mariahdlk1989)

---

## 📄 License

This project is developed for educational and research purposes.
