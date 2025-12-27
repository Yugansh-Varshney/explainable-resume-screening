import shap
import numpy as np
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
