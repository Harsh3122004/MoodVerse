# MoodVerse: AI Entertainment Intelligence

MoodVerse is a full-stack, AI-powered entertainment recommendation platform. It leverages state-of-the-art Neural Collaborative Filtering (NCF), NLP-based mood interpretation, and content-based similarity to recommend Movies, Music, and Anime tailored to your emotional state.

## Project Structure

This project contains both the **Machine Learning Pipelines** (Notebooks) and the **Full-Stack Application** (Flask Backend + Javascript Frontend) in a single unified monorepo.

* `app.py`: Main Flask application.
* `recommend.py`: Core AI prediction and pipeline integrations.
* `notebooks/`: Jupyter Notebooks containing the full EDA, Modeling, and Data Processing pipelines.
* `models/`: Trained model binaries (`.h5`, `.pkl`).
* `datasets/`: Storage for both raw datasets and processed `.csv` outputs.

## Setup Instructions

Because the datasets and certain models are extremely large, they are **omitted from this repository** via `.gitignore` to comply with GitHub's 100MB file limit. You must download and process the datasets locally before running the app.

### 1. Download Datasets
* **MovieLens**: Download the [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/) (or the ML-1M/10M set depending on your training preference).
* Extract the `.csv` files into `datasets/raw/ml-25m/`.
* Any other raw datasets mapped by the notebooks should be placed in `datasets/raw/`.

### 2. Generate Processed Data & Models
If the `models/` folder and `datasets/processed/` folder are empty, you need to run the data pipelines to generate the cleaned `.csv` files and trained `.h5`/`.pkl` model weights.

Open the `notebooks/` directory and run them sequentially:
1. `01_EDA_MovieLens1M.ipynb`
2. `02_Recommendation_System.ipynb`
3. `03_Sentiment_Analysis.ipynb`
4. `04_Popularity_Churn.ipynb`
5. `05_Genre_Classification.ipynb`

Executing these will correctly populate `datasets/processed/` with `movies_clean.csv`, `ratings_clean.csv`, etc., as well as output the required models into the `models/` directory.

### 3. Run the Web Application
Once the datasets and models are generated, you can launch the app perfectly.

```bash
# Install dependencies
pip install -r requirements.txt

# Start the Flask backend and UI
python app.py
```
Go to `http://localhost:5000` to interact with the MoodVerse AI.
