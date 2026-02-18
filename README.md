# LLM-RAG Framework for Facility Management

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) framework developed as part of a Master's thesis in Building Science.

## Overview
The project integrates BIM metadata, Facility Management (FM) records, and O&M documentation into a unified, queryable knowledge environment. It supports hybrid SQL and vector-based retrieval to enable grounded, explainable responses for facility operations.

## Architecture
- BIM data exported to PostgreSQL
- Vector embeddings stored in ChromaDB
- Hybrid retrieval combining SQL filters and semantic search
- LLM-based response generation with evidence grounding
- Streamlit-based user interface

## Technologies
- Python
- Streamlit
- PostgreSQL
- ChromaDB
- SentenceTransformers
- OpenAI-compatible LLMs

## Repository Structure
