from xai.counterfactuals import CounterfactualAnalyzer

resume = """
Machine learning engineer skilled in Python and computer vision.
"""

jd = """
Looking for an AI engineer with PyTorch, Python,
and deep learning experience.
"""

analyzer = CounterfactualAnalyzer(jd)

print(analyzer.add_skill(resume, "PyTorch"))
print(analyzer.add_skill(resume, "Deep Learning"))
print(analyzer.add_skill(resume, "React"))
