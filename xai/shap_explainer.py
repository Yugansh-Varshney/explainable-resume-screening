import shap
import numpy as np
import string
from models.embedding_model import EmbeddingModel

# -----------------------------
# Filters
# -----------------------------

# Words that should NEVER appear as strengths
GENERIC_WORDS = {
    "engineer", "engineering", "developer", "graduate",
    "skills", "experience", "background", "knowledge",
    "project", "projects", "work", "role"
}

# Allowed technical signals
TECHNICAL_KEYWORDS = {
    "python", "java", "sql", "pytorch", "tensorflow", "keras",
    "machine learning", "deep learning", "nlp", "computer vision",
    "yolo", "transformer", "transformers", "data science",
    "shap", "pandas", "numpy", "scikit-learn", "aws", "git"
}

STOPWORDS = {
    "and", "or", "the", "with", "to", "in",
    "of", "for", "a", "an", "is", "are"
}

# -----------------------------
# Utility functions
# -----------------------------

def clean_token(token: str):
    token = token.lower()
    token = token.replace("\n", " ")
    token = token.strip()
    token = token.strip(string.punctuation)

    if token in STOPWORDS or token == "":
        return None

    return token


def split_contributions(tokens, values):
    positives = []
    negatives = []

    for token, value in zip(tokens, values):
        if value > 0:
            positives.append((token, value))
        elif value < 0:
            negatives.append((token, value))

    return positives, negatives


# -----------------------------
# SHAP processing
# -----------------------------

def process_shap_output(tokens, values, top_k=5):
    cleaned_tokens = []
    cleaned_values = []

    for token, value in zip(tokens, values):
        token_cleaned = clean_token(token)

        if token_cleaned is None:
            continue

        # Block generic role words
        if token_cleaned in GENERIC_WORDS:
            continue

        # Keep only technical-looking signals
        if not any(
            keyword in token_cleaned
            for keyword in TECHNICAL_KEYWORDS
        ):
            continue

        cleaned_tokens.append(token_cleaned)
        cleaned_values.append(float(value))

    positives, negatives = split_contributions(
        cleaned_tokens,
        cleaned_values
    )

    positives = sorted(
        positives, key=lambda x: x[1], reverse=True
    )[:top_k]

    negatives = sorted(
        negatives, key=lambda x: x[1]
    )[:top_k]

    return positives, negatives


# -----------------------------
# Resume–JD Scorer (for SHAP)
# -----------------------------

class ResumeJDScorer:
    def __init__(self, job_description: str):
        self.embedder = EmbeddingModel()
        self.jd_embedding = self.embedder.encode(job_description)

    def score(self, resume_texts):
        scores = []
        for text in resume_texts:
            r_emb = self.embedder.encode(text)
            score = float(np.dot(r_emb, self.jd_embedding))
            scores.append(score)
        return scores
