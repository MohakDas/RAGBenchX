from pipelineA_wrapper import RAGPipelineA
from evalA_data import evaluation_questions
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

pipeline = RAGPipelineA("data.txt")
eval_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_score(answer, reference):
    answer_embedding = eval_model.encode([answer])
    reference_embedding = eval_model.encode([reference])
    similarity = cosine_similarity(answer_embedding, reference_embedding)[0][0]
    return float(similarity)

total_score = 0
total_latency = 0

for item in evaluation_questions:
    question = item["question"]
    reference = item["reference_answer"]

    start_time = time.time()
    answer = pipeline.ask(question)
    end_time = time.time()

    latency = end_time - start_time
    score = semantic_score(answer, reference)

    total_score += score
    total_latency += latency

    print("\nQuestion:", question)
    print("Answer:", answer)
    print("Semantic Similarity Score:", round(score, 3))
    print("Latency (seconds):", round(latency, 3))

final_score = total_score / len(evaluation_questions)
avg_latency = total_latency / len(evaluation_questions)

print("\nFinal Average Similarity Score:", round(final_score, 3))
print("Average Latency:", round(avg_latency, 3), "seconds")