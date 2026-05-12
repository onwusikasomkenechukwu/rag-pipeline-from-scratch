# RAG from Scratch

A lightweight Retrieval-Augmented Generation project built to understand the full document-to-answer pipeline behind modern “chat with your documents” systems.

This project loads a webpage, splits it into retrievable chunks, embeds those chunks into a vector database, retrieves relevant context for a user question, and passes that context into an LLM to generate a grounded answer.

## What This Project Does

This notebook implements a basic RAG pipeline:

```text
Raw Web Document
    ↓
Document Loading
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
Vector Database Storage
    ↓
Semantic Retrieval
    ↓
Prompt Construction
    ↓
LLM Answer Generation
