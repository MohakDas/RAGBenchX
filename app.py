import streamlit as st
import tempfile
import pandas as pd
import os
import plotly.graph_objects as go

from pipelineA_wrapper import RAGPipelineA
from pipelineB_wrapper import RAGPipelineB
from sentence_transformers import SentenceTransformer
from benchmark import run_benchmark
from synthetic_eval import generate_eval_questions


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RAGBenchX", layout="wide")

st.title("🚀 RAGBenchX – RAG Benchmark Dashboard")
st.divider()


# ---------------- SESSION STATE ----------------
if "pipeline_a" not in st.session_state:
    st.session_state.pipeline_a = None

if "pipeline_b" not in st.session_state:
    st.session_state.pipeline_b = None

if "document_text" not in st.session_state:
    st.session_state.document_text = None

if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a document (.txt or .pdf)",
    type=["txt", "pdf"]
)

if uploaded_file is not None:

    suffix = "." + uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.success("Document uploaded successfully")

    # Build pipelines once
    if not st.session_state.doc_loaded:

        with st.spinner("Building Pipeline A"):
            st.session_state.pipeline_a = RAGPipelineA(temp_path)

        with st.spinner("Building Pipeline B"):
            st.session_state.pipeline_b = RAGPipelineB(temp_path)

        st.session_state.document_text = st.session_state.pipeline_a.text
        st.session_state.doc_loaded = True

    st.divider()

    st.subheader("Run RAG Benchmark")

    num_q = st.slider("Number of evaluation questions", 1, 5, 3)

    if st.button("Start Benchmark"):

        # -------- Generate Questions --------
        with st.spinner("Generating evaluation questions"):
            eval_questions = generate_eval_questions(
                st.session_state.document_text,
                num_questions=num_q
            )

        if not eval_questions:
            st.error("Question generation failed")
            st.stop()

        eval_model = SentenceTransformer("all-MiniLM-L6-v2")

        # -------- Run Pipelines --------
        with st.spinner("Running pipelines"):

            avg_score_a, avg_latency_a, details_a = run_benchmark(
                st.session_state.pipeline_a,
                eval_questions,
                eval_model
            )

            avg_score_b, avg_latency_b, details_b = run_benchmark(
                st.session_state.pipeline_b,
                eval_questions,
                eval_model
            )

        # ---------------- SUMMARY ----------------
        st.divider()
        st.subheader("📊 Benchmark Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Pipeline A Score", round(avg_score_a, 3))
        c2.metric("Pipeline B Score", round(avg_score_b, 3))
        c3.metric("Pipeline A Latency (s)", round(avg_latency_a, 3))
        c4.metric("Pipeline B Latency (s)", round(avg_latency_b, 3))

        # ---------------- GRAPH ----------------
        st.divider()
        st.subheader("📈 Pipeline Performance Comparison")

        fig = go.Figure()

        # Similarity bars
        fig.add_trace(go.Bar(
            name="Similarity",
            x=["Pipeline A", "Pipeline B"],
            y=[avg_score_a, avg_score_b],
            marker_color="#1f77b4",
            text=[round(avg_score_a, 3), round(avg_score_b, 3)],
            textposition="outside"
        ))

        # Latency bars
        fig.add_trace(go.Bar(
            name="Latency (seconds)",
            x=["Pipeline A", "Pipeline B"],
            y=[avg_latency_a, avg_latency_b],
            marker_color="#ff7f0e",
            text=[round(avg_latency_a, 2), round(avg_latency_b, 2)],
            textposition="outside"
        ))

        fig.update_layout(
            barmode="group",
            title="RAG Pipeline Benchmark",
            xaxis_title="Pipeline",
            yaxis_title="Value",
            legend_title="Metric",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- RAG CONFIGURATION ----------------
        st.divider()
        st.subheader("⚙️ RAG Pipeline Configuration")

        config_data = {
            "Pipeline": ["Pipeline A", "Pipeline B"],
            "Chunk Size": ["512", "1024"],
            "Embedding Model": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/BAAI/bge-small-en-v1.5"
            ],
            "Vector Store": ["FAISS", "FAISS"],
            "Retriever": ["Top-K", "Top-K"]
        }

        config_df = pd.DataFrame(config_data)
        st.table(config_df)

        # ---------------- WINNER ----------------
        st.divider()

        if avg_score_a > avg_score_b:
            st.success("🏆 Pipeline A performs better")
        else:
            st.success("🏆 Pipeline B performs better")