import shap
from models.embedding_model import EmbeddingModel
from xai.shap_explainer import ResumeJDScorer, process_shap_output
from xai.counterfactuals import CounterfactualAnalyzer
from xai.groq_explainer import GroqExplainer
import os
from utils.validators import is_valid_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ResumeInferencePipeline:
    def __init__(self, job_description: str):
        self.job_description = job_description

        
        self.embedder = EmbeddingModel()
        self.jd_embedding = self.embedder.encode(job_description)

        self.scorer = ResumeJDScorer(job_description)
        self.counterfactual = CounterfactualAnalyzer(job_description)
        self.groq_explainer = GroqExplainer()

   
    def compute_score(self, resume_text: str) -> float:
        resume_emb = self.embedder.encode(resume_text)
        return float(resume_emb @ self.jd_embedding)

    def interpret_score(self, score: float) -> str:
        if score >= 0.7:
            return "Strong Match"
        elif score >= 0.4:
            return "Partial Match"
        else:
            return "Weak Match"

   
    def explain(self, resume_text: str) -> dict:
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

   
    def suggest_improvements(self, resume_text: str) -> list:
        skills_to_try = ["PyTorch", "Deep Learning", "AWS"]

        suggestions = []
        for skill in skills_to_try:
            result = self.counterfactual.add_skill(resume_text, skill)
            if result["delta"] > 1:
                suggestions.append(result)

        return suggestions

   
    def run(self, resume_text: str) -> dict:

        # 1. Input validation
        if not is_valid_text(self.job_description) or not is_valid_text(resume_text):
            return {
                "error": (
                    "Invalid input detected. Please provide a meaningful job description "
                    "and resume with sufficient detail."
                )
            }

        # 2. Compute score
        score = self.compute_score(resume_text)

        # 3. Build result dictionary
        structured_result = {
            "score": round(score, 3),
            "match_level": self.interpret_score(score),
            "explanations": self.explain(resume_text),
            "suggested_improvements": self.suggest_improvements(resume_text)
        }

        # 4. Strength gating (VERY IMPORTANT)
        if score < 0.4 or len(structured_result["explanations"]["strengths"]) < 2:
            structured_result["explanations"]["strengths"] = []

        # 5. Natural language explanation
        structured_result["natural_language_explanation"] = (
            self.groq_explainer.explain(structured_result)
        )

        return structured_result

    

