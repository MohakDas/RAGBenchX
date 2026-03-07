import time
from sklearn.metrics.pairwise import cosine_similarity

def run_benchmark(pipeline, questions, eval_model):

    total_score = 0
    total_latency = 0
    detailed_results = []

    for item in questions:
        question = item["question"]
        reference = item["reference_answer"]

        start = time.time()
        answer = pipeline.ask(question)
        end = time.time()

        latency = end - start

        similarity = cosine_similarity(
            eval_model.encode([answer]),
            eval_model.encode([reference])
        )[0][0]

        total_score += similarity
        total_latency += latency

        detailed_results.append({
            "Question": question,
            "Similarity": round(similarity, 3),
            "Latency": round(latency, 3)
        })

    avg_score = total_score / len(questions)
    avg_latency = total_latency / len(questions)

    return avg_score, avg_latency, detailed_results