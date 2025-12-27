from models.embedding_model import EmbeddingModel
from models.resume_scorer import cosine_similarity
from xai.shap_explainer import ResumeJDScorer, process_shap_output
from xai.counterfactuals import CounterfactualAnalyzer


class ResumeInferencePipeline:
    def __init__(self, job_description: str):
        self.job_description = job_description

        self.embedder = EmbeddingModel()
        self.jd_embedding = self.embedder.encode(job_description)

        self.scorer = ResumeJDScorer(job_description)
        self.counterfactual = CounterfactualAnalyzer(job_description)

    def compute_score(self, resume_text: str):
        resume_emb = self.embedder.encode(resume_text)
        score = float(resume_emb @ self.jd_embedding)
        return score
    
    def interpret_score(self, score: float):
        if score >= 0.7:
            return "Strong Match"
        elif score >= 0.4:
            return "Partial Match"
        else:
            return "Weak Match"
    
    def explain(self, resume_text: str):
        import shap

        explainer = shap.Explainer(
            self.scorer.score,
            shap.maskers.Text()
        )

        shap_values = explainer([resume_text])

        tokens = shap_values.data[0]
        values = shap_values.values[0]

        positives, negatives = process_shap_output(tokens, values)

        return {
            "strengths": [t for t, _ in positives],
            "weak_signals": [t for t, _ in negatives]
        }
    
    def suggest_improvements(self, resume_text: str):
        skills_to_try = ["PyTorch", "Deep Learning", "AWS"]

        suggestions = []
        for skill in skills_to_try:
            result = self.counterfactual.add_skill(resume_text, skill)
            if result["delta"] > 1:
                suggestions.append(result)

        return suggestions
    
    def run(self, resume_text: str):
        score = self.compute_score(resume_text)

        result = {
            "score": round(score, 3),
            "match_level": self.interpret_score(score),
            "explanations": self.explain(resume_text),
            "suggested_improvements": self.suggest_improvements(resume_text)
        }

        return result


