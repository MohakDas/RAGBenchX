import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer


class RAGPipelineA:
    def __init__(self, data_path):
        self.text = self.load_document(data_path)
        self.chunks = self.chunk_text(self.text, chunk_size=512)

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunk_embeddings = self.embedding_model.encode(self.chunks)

        dimension = self.chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.chunk_embeddings))

    def load_document(self, path):
        if path.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    def chunk_text(self, text, chunk_size=512):
        words = text.split()
        return [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        return [self.chunks[i] for i in indices[0]]

    def generate_answer(self, query, context):
        prompt = f"""
Answer ONLY using the context.

Context:
{context}

Question:
{query}

Answer:
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2:1.5b",
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]

    def ask(self, query):
        retrieved = self.retrieve(query)
        context = "\n".join(retrieved)
        return self.generate_answer(query, context)