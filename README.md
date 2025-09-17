## Sentiment Analysis on IMDB Movie Reviews

**Purpose**

This project performs binary sentiment analysis (positive vs. negative) on IMDB movie reviews using a traditional machine learning approach with scikit-learn. A TF‑IDF representation is built from the text data and a Logistic Regression classifier is trained and evaluated on a dataset of 50,000 reviews.

**Key points**

- **Dataset**: IMDB reviews (`IMDB Dataset.csv`) with 50,000 rows and 2 columns: `review` (text) and `sentiment` (label: `positive` or `negative`).
- **Preprocessing**: Lowercasing, punctuation removal, and stopword removal using NLTK English stopwords.
- **Features**: `TfidfVectorizer` from scikit-learn.
- **Model**: `LogisticRegression` from scikit-learn.
- **Evaluation**: Accuracy, classification report, and confusion matrix on a 80/20 train/test split. In a sample run, the model achieved about ~0.90 accuracy on the test set.

### Project structure

- `Notebook.ipynb`: End-to-end workflow (load data → clean → vectorize → train → evaluate).
- `IMDB Dataset.csv`: The dataset used for training and evaluation.
- `requirements.txt`: Python dependencies to run the notebook.

### Getting started

1) Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Ensure NLTK stopwords are available. The notebook includes the following line and will download as needed:

```python
import nltk
nltk.download('stopwords')
```

If running in a restricted environment, pre-download the stopwords before executing the notebook.

4) Open and run the notebook:

```bash
jupyter notebook Notebook.ipynb
```

Run all cells to reproduce preprocessing, training, and evaluation.

### Notes

- The current workflow uses classic ML (TF‑IDF + Logistic Regression). You can swap in other linear classifiers (e.g., Linear SVM) or adjust vectorizer parameters (e.g., n-grams, min_df) to explore trade-offs.
- If you change preprocessing, clear and rerun the notebook to rebuild features and retrain the model.


