# RAG From Scratch

A lightweight Retrieval-Augmented Generation pipeline built with LangChain, Chroma, OpenAI embeddings, and web-loaded documents.

This project demonstrates the core architecture behind “chat with your documents” systems: load external text, split it into chunks, embed those chunks, store them in a vector database, retrieve relevant context for a query, and pass that context into an LLM for grounded answer generation.

## Overview

Large language models are powerful, but they do not automatically know about private, recent, or user-provided documents. Retrieval-Augmented Generation solves this by giving the model relevant context at query time.

This notebook builds a simple RAG pipeline around a web article and answers questions using retrieved evidence from that article.

The flow is:

```text
Web Document
    ↓
Document Loading
    ↓
Text Chunking
    ↓
Embedding
    ↓
Vector Store
    ↓
Semantic Retrieval
    ↓
Prompt + LLM
    ↓
Grounded Answer
````

## Features

* Loads web documents using `WebBaseLoader`
* Parses specific HTML sections with BeautifulSoup
* Splits long text into overlapping chunks
* Creates embeddings with OpenAI embeddings
* Stores document vectors in Chroma
* Retrieves semantically relevant chunks for a user query
* Uses a local prompt template instead of deprecated LangChain Hub imports
* Generates concise answers grounded in retrieved context
* Removes unsafe hardcoded API-key handling

## Tech Stack

* Python
* LangChain
* LangChain Community
* LangChain OpenAI
* ChromaDB
* BeautifulSoup4
* OpenAI API

## Installation

Create and activate a virtual environment first.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -U langchain langchain-community langchain-core langchain-openai langchain-text-splitters chromadb beautifulsoup4
```

## Environment Variables

You need an OpenAI API key.

Create a `.env` file or set the environment variable directly.

```bash
OPENAI_API_KEY=your_api_key_here
```

Do **not** hardcode your API key inside the notebook.

On Windows Command Prompt:

```bash
set OPENAI_API_KEY=your_api_key_here
```

On PowerShell:

```bash
$env:OPENAI_API_KEY="your_api_key_here"
```

On macOS/Linux:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Usage

Open the notebook:

```bash
jupyter notebook rag_from_scratch_1-4_cleaned.ipynb
```

Run the notebook cells in order.

The final chain can answer questions like:

```python
rag_chain.invoke("What is Task Decomposition?")
```

The system will:

1. Search the vector database for relevant chunks.
2. Format those chunks as context.
3. Insert the context and question into a prompt.
4. Send the prompt to the LLM.
5. Return a concise grounded answer.

## Core Code Structure

### 1. Load Documents

```python
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()
```

### 2. Split Text

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(docs)
```

### 3. Create Vector Store

```python
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()
```

### 4. Build Prompt

```python
template = """
You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know the answer, say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
"""
```

### 5. Create RAG Chain

```python
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

## What I Learned

This project helped me understand the building blocks of retrieval-grounded LLM systems. Instead of treating an LLM as a standalone answer engine, RAG adds an external memory layer that can be searched at runtime.

The most important idea is that the model does not need to memorize everything. It needs a reliable way to retrieve the right information and reason over it.

This project also exposed some of the practical issues in modern AI tooling, including package version changes, deprecated imports, environment configuration, and safe API-key management.

## Future Improvements

Possible extensions include:

* Add support for PDFs
* Add support for multiple websites
* Save and reload the Chroma vector database
* Add metadata filtering
* Build a Streamlit or FastAPI interface
* Compare different chunk sizes and overlap settings
* Add citation-style source tracking
* Evaluate answer quality across multiple retrieval settings

## Status

This is a small portfolio project built for learning and experimentation. It is not intended to be a production RAG system, but it demonstrates the core pipeline behind many real-world AI search and document QA tools.
