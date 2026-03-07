RAGBenchX

RAGBenchX is a document-driven benchmarking framework designed to evaluate and compare different Retrieval Augmented Generation (RAG) pipelines. The system allows users to upload their own documents, automatically generates evaluation questions from the content, and benchmarks multiple RAG pipelines based on answer quality and response latency. The results are presented through an interactive Streamlit dashboard that highlights performance differences between pipelines.

The goal of this project is to provide a simple yet practical environment for experimenting with RAG architectures and understanding how changes in retrieval configuration—such as chunk size or context length—affect the quality and speed of generated responses.

Project Overview

Retrieval Augmented Generation combines information retrieval with language model generation. Instead of relying solely on a language model’s internal knowledge, a RAG system retrieves relevant context from a document and uses that information to produce more accurate answers.

This project implements two RAG pipelines with different chunking strategies and compares their performance using automatically generated evaluation queries.

Pipeline A uses smaller document chunks (512 tokens), while Pipeline B uses larger chunks (1024 tokens). This difference allows the system to study how context granularity impacts retrieval accuracy and answer generation.

How the System Works

The benchmarking workflow follows these steps:

A user uploads a document (PDF or TXT).

The document is parsed and split into chunks.

Each document chunk is converted into a dense vector embedding using sentence-transformer models, specifically all-MiniLM-L6-v2 for Pipeline A and BAAI/bge-small-en-v1.5 for Pipeline B, enabling semantic retrieval through vector similarity search.
These embeddings are stored in a FAISS vector index to allow fast similarity search.

The system automatically generates evaluation questions and reference answers from the document using a local LLM(qwen1.5:2b in our case) served through Ollama.

Each question is processed by both RAG pipelines.

Relevant document chunks are retrieved using vector similarity search.

The retrieved context is passed to the language model to generate an answer.

Generated answers are compared against the reference answers using cosine similarity between embeddings.

Response latency is also measured for each pipeline.

Results are visualized in a Streamlit dashboard.

This process allows direct comparison between different RAG configurations under the same evaluation conditions.

RAG Pipeline Configurations

The project currently benchmarks two pipelines.

Pipeline A
Chunk Size: 512
Embedding Model: sentence-transformers/all-MiniLM-L6-v2
Vector Store: FAISS
Retrieval: Top-K similarity search

Pipeline B
Chunk Size: 1024
Embedding Model: sentence-transformers/BAAI/bge-small-en-v1.5
Vector Store: FAISS
Retrieval: Top-K similarity search

The larger chunk size in Pipeline B typically provides broader context, while smaller chunks in Pipeline A may improve retrieval precision.

Evaluation Metrics

Two primary metrics are used to compare the pipelines.

Semantic Similarity
The generated answer from each pipeline is compared with the reference answer using cosine similarity between embedding vectors. This provides an approximate measure of how closely the generated response matches the expected answer.

Latency
The time required for each pipeline to generate an answer is measured. This helps analyze the trade-off between response quality and system speed.

Dashboard Visualization

The Streamlit interface displays:

• Benchmark summary metrics
• Pipeline comparison charts
• Similarity scores for both pipelines
• Latency measurements
• Pipeline configuration details

This makes it easy to visually compare different RAG setups.

Technology Stack

The system is built using the following tools and libraries:

Streamlit – interactive web interface
FAISS – vector similarity search
Sentence Transformers – embedding generation
Ollama – local LLM runtime
Plotly – performance visualization
Python – core implementation

The local language model used for question generation and answering can be any model supported by Ollama (qwen1.5:2b in our case).

Project Structure
RAGBenchX
│
├── app.py
│   Streamlit dashboard
│
├── benchmark.py
│   Benchmark execution logic
│
├── synthetic_eval.py
│   Synthetic question generation
│
├── pipelineA_wrapper.py
├── pipelineB_wrapper.py
│   RAG pipeline implementations
│
├── pipelineA.py
│   Pipeline utilities
│
├── evalA.py
├── evalB.py
├── evalA_data.py
│   Evaluation helpers
│
├── requirements.txt
└── README.md
Installation

Clone the repository:

git clone https://github.com/yourusername/RAGBenchX.git
cd RAGBenchX

Create a virtual environment:

python -m venv venv

Activate the environment:

Windows

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Running the Application

Start the Ollama server:

ollama serve

Make sure a model is installed:

ollama pull qwen2:1.5b

Then run the Streamlit app:

streamlit run app.py

Open the dashboard in your browser:

http://localhost:8501

Upload a document and start the benchmark.

Example Workflow

Upload a document

System generates evaluation questions

Both RAG pipelines answer each question

Similarity and latency metrics are computed

Results are displayed in the dashboard

Future Improvements

Several improvements could further extend this framework:

• Retrieval recall metrics (Recall@K)
• Faithfulness evaluation
• Context relevance scoring
• Support for additional embedding models
• Integration with evaluation frameworks such as RAGAS
• Multi-document benchmarking

License

MIT License
