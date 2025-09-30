## Sentiment Analysis on IMDB Movie Reviews

**Purpose**

This project performs binary sentiment analysis (positive vs. negative) on IMDB movie reviews using two approaches: a classic machine learning pipeline with scikit-learn and a simple feed-forward neural network with PyTorch. Text is vectorized via TF‑IDF; models are trained and evaluated on 50,000 reviews.

**Key points**

- **Dataset**: IMDB reviews (`IMDB Dataset.csv`) with 50,000 rows and 2 columns: `review` (text) and `sentiment` (label: `positive` or `negative`).
- **Preprocessing**: Lowercasing, punctuation removal, and stopword removal using NLTK English stopwords.
- **Features**: `TfidfVectorizer` from scikit-learn.
- **Models**:
  - `LogisticRegression` (scikit-learn)
  - 2-layer feed-forward network (PyTorch) trained with Adam and BCE loss
- **Evaluation**: Accuracy, classification report, and confusion matrix on an 80/20 train/test split.

### Final results

- **Scikit-learn (Logistic Regression)**: accuracy ≈ 0.90
- **PyTorch (FFN)**: accuracy ≈ 0.89

Given the similar performance and lower complexity, the scikit-learn model is recommended for production scenarios in this project.

### Project structure

- `notebooks/Sentiment-Analysis.ipynb`: End-to-end workflow (load → clean → vectorize → train → evaluate) for both scikit-learn and PyTorch models.
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

3) Ensure NLTK stopwords are available. The notebook includes the following and will download as needed:

```python
import nltk
nltk.download('stopwords')
```

If running in a restricted environment, pre-download the stopwords before executing the notebook.

4) Open and run the notebook:

```bash
jupyter notebook notebooks/Sentiment-Analysis.ipynb
# or
jupyter lab notebooks/Sentiment-Analysis.ipynb
```

Run all cells to reproduce preprocessing, training, evaluation, and example real-time predictions for both models.

### Reproducing the results

- Place `IMDB Dataset.csv` at the project root (already included here).
- Execute the notebook cells in order. Key outputs to look for:
  - Vectorized feature matrix shape (e.g., `(50000, 180395)`).
  - Test metrics for scikit-learn and PyTorch (accuracy, classification report, confusion matrix).
  - Example predictions for custom input sentences.

### Notes

- You can explore other linear models (e.g., Linear SVM) or adjust TF‑IDF parameters (n-grams, min_df) for different trade-offs.
- The PyTorch section demonstrates handling sparse TF‑IDF data by converting batches to dense tensors, with `DataLoader` to manage memory usage.

