from pipeline.inference import ResumeInferencePipeline

resume = """
Machine learning engineer skilled in Python and computer vision.
"""

jd = """
Looking for an AI engineer with PyTorch, Python,
and deep learning experience.
"""

pipeline = ResumeInferencePipeline(jd)
result = pipeline.run(resume)

print("\n=== FINAL RESULT ===\n")
for k, v in result.items():
    print(f"{k}: {v}")
