import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqExplainer:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    def explain(self, result: dict):
        

        prompt = f"""
You are an AI resume analysis assistant.

Your role is to explain the evaluation of a resume to the candidate
in clear, honest, and neutral language.

Important rules:
- Speak directly to the candidate (use "your resume", "you")
- Do NOT speak as a recruiter or employer
- Do NOT exaggerate strengths
- If strong signals are weak or unclear, explicitly say so
- Avoid bullet points or numbered lists
- Write 1–2 short paragraphs in a calm, advisory tone

Evaluation summary:
- Match category: {result['match_level']}
- Match score: {round(result['score'] * 100, 1)}%
- Detected strengths: {', '.join(result['explanations']['strengths']) if result['explanations']['strengths'] else 'None clearly detected'}
- Possible improvement areas: {', '.join(
        [s['skill_added'] for s in result['suggested_improvements']]
    ) if result['suggested_improvements'] else 'None identified'}

Explain the result honestly.
If the resume lacks sufficient detail or clear strengths,
say so gently and suggest adding more concrete information.
Do NOT mention technical methods or internal mechanics.
"""




        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )

        return response.choices[0].message.content.strip()
