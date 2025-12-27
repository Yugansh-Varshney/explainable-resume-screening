# xai/test_shap.py

import shap
from xai.shap_explainer import ResumeJDScorer

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

# Print top contributing tokens
tokens = shap_values.data[0]
values = shap_values.values[0]

token_contributions = list(zip(tokens, values))
token_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

print("Top contributing tokens:")
for tok, val in token_contributions[:10]:
    print(f"{tok:15} -> {val:.4f}")
