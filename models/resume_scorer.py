import numpy as np
from embedding_model import EmbeddingModel

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)

if __name__ == "__main__":
    resume_text = """
    hi, Looking for an AI engineer skilled in Python, deep learning,
    PyTorch, and computer vision techniques.
    """

    job_description = """
    Looking for an AI engineer skilled in Python, deep learning,
    PyTorch, and computer vision techniques.
    """

    embedder = EmbeddingModel()

    resume_embedding = embedder.encode(resume_text)
    jd_embedding = embedder.encode(job_description)

    similarity_score = cosine_similarity(resume_embedding, jd_embedding)

    print(f"Resume–JD Similarity Score: {round(similarity_score * 100, 2)}")
