import shap
import numpy as np
import string
from models.embedding_model import EmbeddingModel

class ResumeJDScorer:
    def __init__(self, job_description: str):
        self.embedder = EmbeddingModel()
        self.jd_embedding = self.embedder.encode(job_description)

    def score(self, resume_texts):
        """
        resume_texts: list[str]
        returns: list[float]
        """
        scores = []
        for text in resume_texts:
            r_emb = self.embedder.encode(text)
            score = float(np.dot(r_emb, self.jd_embedding))
            scores.append(score)
        return scores



def split_contributions(tokens, values):
    positives = []
    negatives = []

    for token, value in zip(tokens, values):
        if value > 0:
            positives.append((token, value))
        elif value < 0:
            negatives.append((token, value))

    return positives, negatives


STOPWORDS = {
    "and", "or", "the", "with", "to", "in",
    "of", "for", "a", "an", "is", "are"
}

def clean_token(token: str):
    token = token.lower()
    token = token.replace("\n", " ")
    token = token.strip()
    token = token.strip(string.punctuation)

    if token in STOPWORDS or token == "":
        return None

    return token


def process_shap_output(tokens, values, top_k=5):
    cleaned_tokens = []
    cleaned_values = []

    # clean tokens first
    for token, value in zip(tokens, values):
        token_cleaned = clean_token(token)
        if token_cleaned is not None:
            cleaned_tokens.append(token_cleaned)
            cleaned_values.append(float(value))

    # split into positive and negative
    positives, negatives = split_contributions(
        cleaned_tokens,
        cleaned_values
    )

    # sort by contribution strength
    positives = sorted(positives, key=lambda x: x[1], reverse=True)[:top_k]
    negatives = sorted(negatives, key=lambda x: x[1])[:top_k]

    return positives, negatives
