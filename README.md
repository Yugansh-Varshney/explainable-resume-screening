# Explainable Resume Analyzer (XAI + GenAI)

An end-to-end **Explainable AI system** that evaluates how well a resume matches a job description, explains *why* it received that score, and provides **actionable improvement suggestions**.

This project focuses on **interpretability, transparency, and responsible use of Generative AI**, rather than treating resume screening as a black box.

---

## Features

- **Semantic Resume–Job Description Matching**  
  Uses transformer-based embeddings to compute meaningful similarity instead of keyword matching.

- **Explainable AI (XAI)**  
  Identifies which resume terms contribute positively or weakly to the match using SHAP-based explanations.

- **Counterfactual Skill Suggestions**  
  Suggests specific skills that could improve alignment (e.g., “Adding PyTorch could improve the score by +12%”).

- **GenAI-Powered Natural Language Feedback**  
  Converts model outputs into clear, human-readable guidance using a large language model.

- **Input Validation & Guardrails**  
  Prevents misleading explanations for low-quality or invalid inputs.

- **Interactive Web Application**  
  Built using Streamlit for easy testing and demonstration.

---

## Why This Project Is Different

Most resume screening tools rely on keyword matching, produce opaque scores, and provide no actionable feedback.

This system:
- separates decision logic from explanation logic
- ensures explanations are grounded in model behavior
- avoids hallucinations and overconfident outputs
- prioritizes transparency and user trust

---

## High-Level Architecture

Resume / Job Description
↓
Text Embedding Model
↓
Semantic Similarity Scoring
↓
Explainability (SHAP)
↓
Counterfactual Skill Analysis
↓
GenAI Explanation Layer
↓
Streamlit Web Interface


---

## Tech Stack

- Python  
- Sentence Transformers  
- SHAP (Explainable AI)  
- NumPy  
- Groq API (LLM for explanations)  
- Streamlit  

---




## How to Run Locally

### 1. Clone the repository

git clone https://github.com/Yugansh-Varshney/explainable-resume-screening.git
cd explainable-resume-screening

 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Set environment variables

Create a .env file:

GROQ_API_KEY=your_api_key_here

5. Run the application
streamlit run app.py


Example Output

Match Score: 74%

Match Level: Strong Match

Strengths Detected: Python, Java, Data Structures, SQL

Suggested Improvements: Add PyTorch, Cloud experience

Explanation: Clear natural-language feedback explaining alignment and gaps

Responsible AI Considerations

Invalid or low-quality inputs are detected and blocked

Generic or non-technical words are not treated as strengths

LLMs are used only for explanation, not decision-making

The system avoids misleading or overconfident guidance

Future Improvements

Resume upload support (PDF / DOCX)

Skill grouping (Languages, ML, Backend, Cloud)

Confidence scoring (low / medium / high)

UI enhancements and analytics

Bias and fairness evaluation

Author

Yugansh
B.Tech (ITE)
Interests: Explainable AI, Agentic AI, Generative AI

Why This Matters

This project demonstrates strong ML fundamentals, explainability-first thinking, real-world system design, and responsible use of Generative AI.
It is designed not just to predict, but to explain and guide.
