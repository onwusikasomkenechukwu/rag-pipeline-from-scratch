````markdown
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
````

Instead of asking an LLM to answer from pretrained knowledge alone, the system retrieves relevant source material first and uses that retrieved context to produce a more grounded response.

## Why I Built This

I built this as a small AI systems project to understand how retrieval-based LLM applications work under the hood. RAG is the foundation behind many modern AI tools, including internal knowledge assistants, document search systems, research copilots, customer support bots, and coding assistants.

The goal was not to build a large production system, but to understand the core architecture clearly and make the pipeline runnable from end to end.

## Features

* Loads web documents using LangChain’s `WebBaseLoader`
* Extracts relevant article content using BeautifulSoup
* Splits long documents into overlapping chunks
* Generates embeddings using OpenAI embeddings
* Stores document vectors in Chroma
* Retrieves semantically relevant chunks for a user query
* Builds a local prompt template for grounded question answering
* Uses an OpenAI chat model to generate concise answers
* Avoids deprecated LangChain `hub.pull()` usage
* Uses safer API key handling instead of hardcoding credentials

## Tech Stack

* Python
* LangChain
* LangChain Community
* LangChain OpenAI
* LangChain Text Splitters
* ChromaDB
* BeautifulSoup4
* OpenAI API

## Project Structure

```text
rag-from-scratch/
│
├── rag_from_scratch_1-4.ipynb          # Main notebook
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/rag-from-scratch.git
cd rag-from-scratch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

A possible `requirements.txt` file:

```text
langchain
langchain-community
langchain-core
langchain-openai
langchain-text-splitters
chromadb
beautifulsoup4
openai
```

## API Key Setup

This project requires an OpenAI API key.

Do **not** hardcode your API key directly inside the notebook.

Set it as an environment variable instead:

### macOS/Linux

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Windows Command Prompt

```cmd
set OPENAI_API_KEY=your-api-key-here
```

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

Then run the notebook normally.

## How It Works

### 1. Load the Source Document

The pipeline begins by loading a web article:

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

This extracts the main article content instead of scraping the entire webpage.

### 2. Split the Text

Long documents are split into smaller overlapping chunks:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(docs)
```

Chunking makes retrieval more precise and helps preserve context across boundaries.

### 3. Create Embeddings and Store Them

Each chunk is converted into an embedding vector and stored in Chroma:

```python
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()
```

This creates a searchable semantic index over the document.

### 4. Retrieve Context

When a user asks a question, the retriever finds the most relevant chunks from the vector database.

```python
retriever = vectorstore.as_retriever()
```

### 5. Generate an Answer

The retrieved chunks are inserted into a prompt and passed to an LLM:

```python
rag_chain.invoke("What is Task Decomposition?")
```

The model then answers using the provided context.

## Example Question

```python
rag_chain.invoke("What is Task Decomposition?")
```

Example output:

```text
Task decomposition is the process of breaking a complex task into smaller, more manageable steps. In LLM agents, this can be done through prompting, planning methods, or external tools that help the model reason through intermediate actions. It allows the agent to solve difficult problems by handling one step at a time.
```

## What I Learned

This project helped me understand the basic components of RAG systems:

* why chunking strategy matters
* how embeddings represent semantic meaning
* how vector databases enable similarity search
* how retrieval improves factual grounding
* how prompt templates structure LLM responses
* how fast-moving AI libraries like LangChain can break older code
* why safe credential handling matters in AI projects

## Limitations

This is a minimal educational project, not a production RAG system.

Current limitations include:

* single-document retrieval
* no persistent Chroma database directory
* no reranking step
* no citation formatting in responses
* no evaluation framework for answer quality
* no frontend or API wrapper
* no support for PDFs or uploaded files yet

## Possible Future Improvements

Future versions could include:

* persistent vector storage
* support for multiple documents
* PDF ingestion
* metadata filtering
* retrieved-source citations
* reranking with a cross-encoder
* a Streamlit or React frontend
* evaluation with test questions
* comparison across embedding models
* local LLM support

## Status

Completed as a small portfolio project for learning and experimentation.

This project is not intended to replace my main research work in embedded ML and biomedical sensing, but it serves as a useful systems exercise in retrieval-grounded language model applications.

## Author

**Somkenechukwu Onwusika**
Electrical Engineering, Howard University
Portfolio: [https://onwusikasomkenechukwu.github.io/](https://onwusikasomkenechukwu.github.io/)

```
```
