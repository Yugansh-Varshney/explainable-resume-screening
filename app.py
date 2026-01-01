import streamlit as st
import os

from pipeline.inference import ResumeInferencePipeline
from utils.resume_parser import extract_resume_text

# Avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Explainable Resume Analyzer",
    page_icon="📄",
    layout="centered"
)

# -----------------------------
# App header
# -----------------------------
st.title("Explainable Resume Analyzer")
st.write(
    "Analyze how well a resume matches a job description, "
    "understand why it received that score, and see what can be improved."
)

st.divider()

# -----------------------------
# Job Description Input
# -----------------------------
st.subheader("Job Description")
job_description = st.text_area(
    "Paste the job description here",
    height=180
)

# -----------------------------
# Resume Input (Upload or Paste)
# -----------------------------
st.subheader("Resume")

uploaded_file = st.file_uploader(
    "Upload your resume (PDF or DOCX)",
    type=["pdf", "docx"]
)

resume_text = st.text_area(
    "Or paste resume text manually",
    height=180
)

st.divider()

analyze_button = st.button("Analyze Resume")

# -----------------------------
# Run Analysis
# -----------------------------
if analyze_button:
    # Decide resume source
    if uploaded_file:
        try:
            resume_text = extract_resume_text(uploaded_file)
        except Exception:
            st.error("Failed to extract text from the uploaded resume.")
            st.stop()

    # Basic validation
    if not job_description.strip() or not resume_text.strip():
        st.warning("Please provide both a job description and a resume.")
        st.stop()

    with st.spinner("Analyzing resume..."):
        pipeline = ResumeInferencePipeline(job_description)
        result = pipeline.run(resume_text)

    # -----------------------------
    # Handle invalid input
    # -----------------------------
    if "error" in result:
        st.error(result["error"])

    else:
        # -----------------------------
        # Results
        # -----------------------------
        st.subheader("Match Result")
        st.write(f"**Match Score:** {round(result['score'] * 100, 1)}%")
        st.write(f"**Match Level:** {result['match_level']}")

        st.divider()

        st.subheader("Strengths Detected")
        if result["explanations"]["strengths"]:
            for skill in result["explanations"]["strengths"]:
                st.write(f"- {skill}")
        else:
            st.write("No strong technical signals detected.")

        st.divider()

        st.subheader("Suggested Improvements")
        if result["suggested_improvements"]:
            for s in result["suggested_improvements"]:
                st.write(
                    f"- Adding **{s['skill_added']}** could improve the match "
                    f"by approximately **{round(s['delta'], 1)}%**"
                )
        else:
            st.write("No major improvement suggestions identified.")

        st.divider()

        st.subheader("Explanation")
        st.write(result["natural_language_explanation"])
