## RAGBenchX

RAGBenchX is a document driven benchmarking framework designed to evaluate and compare different Retrieval Augmented Generation RAG pipelines. The system allows users to upload their own documents automatically generates evaluation questions from the content and benchmarks multiple RAG pipelines based on answer quality and response latency. The results are presented through an interactive Streamlit dashboard that highlights performance differences between pipelines.

The goal of this project is to provide a simple practical environment for experimenting with RAG architectures and understanding how changes in retrieval configuration such as chunk size or context length affect the quality and speed of generated responses.

## Project Overview

Retrieval Augmented Generation combines information retrieval with language model generation. Instead of relying only on a language model’s internal knowledge a RAG system retrieves relevant context from a document and uses that information to produce more accurate answers.

This project implements two RAG pipelines with different chunking strategies and compares their performance using automatically generated evaluation queries.

Pipeline A uses smaller document chunks of 512 tokens while Pipeline B uses larger chunks of 1024 tokens. This difference allows the system to study how context granularity impacts retrieval accuracy and answer generation.

## How the System Works

The benchmarking workflow follows these steps.

A user uploads a document in PDF or TXT format.

The document is parsed and split into chunks.

Each document chunk is converted into a dense vector embedding using sentence transformer models. Pipeline A uses all MiniLM L6 v2 while Pipeline B uses BAAI bge small en v1.5. These embeddings allow semantic retrieval through vector similarity search.

The embeddings are stored in a FAISS vector index to enable fast similarity search.

The system automatically generates evaluation questions and reference answers from the document using a local language model served through Ollama. In our implementation we used the Qwen model qwen1.5:2b.

Each question is processed by both RAG pipelines.

Relevant document chunks are retrieved using vector similarity search.

The retrieved context is passed to the language model which generates an answer.

Generated answers are compared against the reference answers using cosine similarity between embedding vectors.

Response latency is also measured for each pipeline.

The final results are visualized through an interactive Streamlit dashboard.

This process allows direct comparison between different RAG configurations under identical evaluation conditions.

## RAG Pipeline Configurations

The system benchmarks two retrieval pipelines.

Pipeline A
Chunk Size 512
Embedding Model sentence transformers all MiniLM L6 v2
Vector Store FAISS
Retrieval Top K similarity search

Pipeline B
Chunk Size 1024
Embedding Model sentence transformers BAAI bge small en v1.5
Vector Store FAISS
Retrieval Top K similarity search

The larger chunk size used in Pipeline B provides broader context while smaller chunks in Pipeline A may improve retrieval precision.

## Evaluation Metrics

Two metrics are used to compare pipeline performance.

*Semantic Similarity*
The generated answer from each pipeline is compared with the reference answer using cosine similarity between embedding vectors. This provides an approximate measure of how closely the generated response matches the expected answer.

*Latency*
The time required for each pipeline to generate an answer is measured. This helps evaluate the trade off between response quality and system speed.

## Dashboard Visualization

The Streamlit interface displays benchmark results through several visual components.

- Benchmark summary metrics
- Pipeline comparison charts
- Similarity scores for both pipelines
- Latency measurements
- Pipeline configuration details

These visualizations make it easier to compare different RAG setups and understand the impact of different retrieval configurations.

## Technology Stack

The system is built using the following tools and libraries.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge)
![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-FF6F00?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

- Streamlit for the interactive dashboard
- FAISS for vector similarity search
- Sentence Transformers for embedding generation
- Ollama for running local language models
- Plotly for performance visualization
- Python for the core implementation

The local language model used in this implementation is Qwen running through Ollama.

## Project Structure

```text
RAGBenchX/
├── app.py                  # Streamlit dashboard
├── benchmark.py            # Benchmark execution logic
├── synthetic_eval.py       # Synthetic question generation
├── pipelineA_wrapper.py    # RAG pipeline implementation A
├── pipelineB_wrapper.py    # RAG pipeline implementation B
├── pipelineA.py            # Pipeline utilities
├── evalA.py                # Evaluation helper
├── evalB.py                # Evaluation helper
├── evalA_data.py           # Evaluation data/helper
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation

```bash
git clone https://github.com/MohakDas/RAGBenchX.git
cd RAGBenchX
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

```bash
ollama serve
ollama pull qwen2:1.5b
streamlit run app.py
```

Open the dashboard in your browser, upload a document, and start the benchmark.

## Example Workflow

- Upload a document
- Generate evaluation questions
- Run both RAG pipelines
- Compute similarity and latency metrics
- Display results in the dashboard
## Screenshots

<p align="center">
    <img width="1285" height="421" alt="1" src="https://github.com/user-attachments/assets/998719b2-c68f-4121-9179-75e8b95cf2d9" />
    <img width="1272" height="438" alt="2" src="https://github.com/user-attachments/assets/c2679224-fc3d-48fe-b584-049e7fe05a09" />
    <img width="1278" height="548" alt="3" src="https://github.com/user-attachments/assets/88551309-e7d0-4339-8194-9e1b6539456b" />
    <img width="1298" height="381" alt="4" src="https://github.com/user-attachments/assets/93721992-3ebf-43b1-9107-d1ad62cf5d48" />


</p>

> PS: Results were obtained locally using smaller models. Larger models and stronger hardware would likely improve both scores and latency.

## Future Improvements

Several improvements could further extend this framework.

Retrieval recall metrics such as Recall at K
Faithfulness evaluation
Context relevance scoring
Support for additional embedding models
Integration with evaluation frameworks such as RAGAS
Multi document benchmarking

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
MIT License
