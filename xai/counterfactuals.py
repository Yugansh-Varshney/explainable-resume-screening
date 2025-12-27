from models.embedding_model import EmbeddingModel
import numpy as np

class CounterfactualAnalyzer:
    def __init__(self, job_description: str):
        self.embedder = EmbeddingModel()
        self.jd_emb = self.embedder.encode(job_description)

    def score(self, text: str):
        r_emb = self.embedder.encode(text)
        return float(np.dot(r_emb, self.jd_emb))

    def add_skill(self, resume_text: str, skill: str):
        original_score = self.score(resume_text)
        modified_resume = resume_text + f" {skill}"
        new_score = self.score(modified_resume)

        return {
            "skill_added": skill,
            "original_score": round(original_score * 100, 2),
            "new_score": round(new_score * 100, 2),
            "delta": round((new_score - original_score) * 100, 2)
        }
