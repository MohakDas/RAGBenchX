import requests
import re
import json


def generate_eval_questions(document_text, num_questions=3):

    document_text = document_text[:1200]

    prompt = f"""
Generate {num_questions} question-answer pairs from the document.

Each pair MUST follow this format exactly:

{{
 "question": "...",
 "reference_answer": "..."
}}

Return only the objects. Do not number them.

Document:
{document_text}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2:1.5b",
            "prompt": prompt,
            "stream": False,
            "options": {"num_gpu": 0}
        }
    )

    result = response.json()

    if "response" not in result:
        return []

    raw_text = result["response"]

    # Extract every JSON object
    objects = re.findall(r'\{[^{}]*\}', raw_text)

    questions = []

    for obj in objects:
        try:
            parsed = json.loads(obj)
            if "question" in parsed and "reference_answer" in parsed:
                questions.append(parsed)
        except:
            continue

    return questions