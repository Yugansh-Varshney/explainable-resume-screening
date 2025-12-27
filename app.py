import streamlit as st
from pipeline.inference import ResumeInferencePipeline
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="Explainable Resume Analyzer",
    page_icon="📄",
    layout="centered"
)

st.title("Explainable Resume Analyzer")
st.write(
    "This tool analyzes how well a resume matches a job description, "
    "explains the reasoning, and suggests improvements."
)

st.divider()

st.subheader("Job Description")
job_description = st.text_area(
    "Paste the job description here",
    height=150
)

st.subheader("Resume")
resume_text = st.text_area(
    "Paste the resume here",
    height=150
)

st.divider()

analyze_button = st.button("Analyze Resume")


if analyze_button:
    if not job_description.strip() or not resume_text.strip():
        st.warning("Please provide both a job description and a resume.")
    else:
        with st.spinner("Analyzing resume..."):
            pipeline = ResumeInferencePipeline(job_description)
            result = pipeline.run(resume_text)

        
        if "error" in result:
            st.error(result["error"])

        else:
            
            st.subheader("Match Result")
            st.write(f"**Match Score:** {round(result['score'] * 100, 1)}%")
            st.write(f"**Match Level:** {result['match_level']}")

            st.divider()

            st.subheader("Strengths Detected")
            if result["explanations"]["strengths"]:
                for skill in result["explanations"]["strengths"]:
                    st.write(f"- {skill}")
            else:
                st.write("No strong signals detected.")

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
