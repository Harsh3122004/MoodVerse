# 🌌 MoodVerse — AI Entertainment Intelligence

> *Discover movies, anime, and music perfectly matched to your mood using state-of-the-art Machine Learning.*

MoodVerse is a full-stack, AI-powered entertainment recommendation platform. It combines **Neural Collaborative Filtering (NCF)**, **DistilBERT Sentiment Analysis**, **TF-IDF Genre Prediction**, and a real-time **Vibe Slider** engine to deliver hyper-personalized recommendations across Movies, Anime, and Music.

---

## 🚀 Quick Start (Easiest Way to Run)

We've made it incredibly simple to get MoodVerse running on your local machine.

### 1. Clone the repository
```bash
git clone https://github.com/Harsh3122004/MoodVerse.git
cd MoodVerse
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Auto-Download Datasets & Models
Because the ML models and datasets exceed GitHub's file size limits, they are securely hosted on Google Drive. Simply run our setup script to automatically download (`~250MB`) and configure everything:
```bash
python setup_datasets.py
```
*(This will also pre-calculate analytics dashboard data so the app runs at lightning speed).*

### 4. Launch the App
```bash
python app.py
```
Open your browser and navigate to **http://localhost:5000**. Enjoy! 🎉

---

## ✨ Features

- 🎬 **Mood-Based Discovery** — Select one or multiple moods to get tailored recommendations across Movies, Anime, and Music.
- 🔍 **Semantic Similarity Search** — Search for any title and instantly find similar content using our AI similarity engine.
- 🎛️ **Vibe Sliders** — Fine-tune your recommendations in real-time by adjusting *Energy, Darkness, Nostalgia,* and *Pacing*.
- 🤖 **AI Tools (Review Analyzer)** — Live HuggingFace DistilBERT sentiment analysis and multi-label genre prediction on custom text.
- ⚡ **Lightning Fast** — Features an optimized SQLite caching system for instant poster loading and a pre-computed Analytics Dashboard.
- 📊 **Analytics Dashboard** — Interactive Chart.js visualizations built over 1M+ MovieLens ratings.
- 🔐 **User Auth** — Register/login with a session-based watchlist and review history.

---

## 🧠 How It Works (The ML Engines)

MoodVerse isn't just a simple database query; it uses real Machine Learning to understand content:
- **Neural Collaborative Filtering (NCF):** Uses deep learning to predict what movies a user will like based on complex interaction patterns.
- **DistilBERT Transformer:** Evaluates the sentiment (Positive/Negative) of reviews using a state-of-the-art Natural Language Processing model.
- **TF-IDF & Cosine Similarity:** Analyzes the semantic meaning of titles, genres, and descriptions to find items that "feel" the same.

---

## 📂 Project Structure

```text
MoodVerse/
├── app.py                # Flask backend & all API routes
├── recommend.py          # Core AI recommendation & prediction logic
├── database.py           # Permanent SQLite cache & user data
├── setup_datasets.py     # Automated environment configuration script
├── index.html            # Full React frontend (no build step required)
├── requirements.txt      # Python dependencies
├── models/               # Trained ML model binaries (auto-downloaded)
├── datasets/             # Processed datasets (auto-downloaded)
└── notebooks/            # Jupyter Notebooks used for training & EDA
```

---

## 🛠️ Advanced: Manual Setup & Model Training

If you are a developer looking to explore the data science aspect or re-train the models from scratch, follow these steps.

### Manual Dataset Download
If you prefer not to use `setup_datasets.py`, manually download the datasets here:
> **[MoodVerse Datasets — Google Drive (ZIP)](https://drive.google.com/file/d/1i4liNGcDJs0fLjNXL6SnT2-9k20DrXQs/view?usp=sharing)**

Extract the ZIP and place `svd_artifacts.pkl` into the `models/` folder, and the contents of `raw` and `processed` into `datasets/raw/` and `datasets/processed/` respectively.

### Re-training the Models
To regenerate the recommendations and models yourself, run the Jupyter Notebooks sequentially:
| Notebook | Purpose | Required Dataset (in `datasets/raw/`) |
|---|---|---|
| `01_EDA_MovieLens1M.ipynb` | Exploratory Data Analysis | `ml-1m/` |
| `02_Recommendation_System.ipynb`| Builds the NCF and SVD engines | `ml-25m/` |
| `03_Sentiment_Analysis.ipynb` | Trains the BiLSTM Fallback model | `ml-1m/` |
| `04_Popularity_Churn.ipynb` | Analyzes viewer trends over time | `ml-10M100K/` |
| `05_Genre_Classification.ipynb` | Builds the Genre Predictor | `ml-1m/` |

---

## 💻 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, Flask |
| **Frontend** | React (via CDN), Chart.js, Vanilla CSS |
| **ML Models** | TensorFlow/Keras (NCF, BiLSTM), Scikit-learn (SVD, MLP) |
| **NLP** | HuggingFace DistilBERT, TF-IDF |
| **Database** | SQLite (User data & persistent poster cache) |
| **External APIs** | OMDB (Movies), Jikan v4 & Kitsu (Anime) |

---

## 👤 Author

**Harsh** — [@Harsh3122004](https://github.com/Harsh3122004)
