import shap
from xai.shap_explainer import ResumeJDScorer
from xai.shap_explainer import process_shap_output


resume_text = """
Machine learning engineer skilled in Python, PyTorch,
computer vision, and YOLO-based object detection.
"""

job_description = """
Looking for an AI engineer with strong Python,
PyTorch, and computer vision experience.
"""

scorer = ResumeJDScorer(job_description)

explainer = shap.Explainer(
    scorer.score,
    shap.maskers.Text()
)

shap_values = explainer([resume_text])

tokens = shap_values.data[0]
values = shap_values.values[0]

positives, negatives = process_shap_output(tokens, values)

result = {
    "top_positive_tokens": positives,
    "top_negative_tokens": negatives
}

print(result)
