# 🤖 RAG Document Assistant

A Retrieval-Augmented Generation (RAG) based document question-answering application built with **Python, LangChain, Mistral AI, ChromaDB, and Streamlit**.

The application allows users to upload a PDF document and ask questions about its content. Instead of relying only on the LLM's existing knowledge, the system retrieves relevant information from the uploaded document and provides an answer based on that context.

## 🌐 Live Demo

Try the application here:

https://rag-document-assistant-system.streamlit.app/

---

## 📌 Project Overview

Large Language Models can generate powerful answers, but they may not know the information contained inside a user's private documents.

This project solves that problem using **Retrieval-Augmented Generation (RAG)**.

The system follows this workflow:

PDF
↓
Text Extraction
↓
Text Chunking
↓
Mistral Embeddings
↓
ChromaDB
↓
MMR Retrieval
↓
Relevant Context
↓
Mistral LLM
↓
AI Answer

The user can upload a document and ask natural-language questions about it.

---

## ✨ Features

- 📄 Upload PDF documents
- 🔍 Ask questions about uploaded documents
- ✂️ Automatic text chunking
- 🔢 Generate vector embeddings using Mistral
- 🗄️ Store embeddings in ChromaDB
- 🧠 Retrieve relevant document chunks
- 🎯 MMR-based retrieval for diverse relevant results
- 🤖 Generate answers using Mistral LLM
- 🌐 Streamlit web interface
- 🔐 API keys handled using environment variables / Streamlit Secrets
- 🚀 Deployed on Streamlit Cloud

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines information retrieval with a Large Language Model.

Instead of directly asking the LLM a question:

User Question
→ LLM
→ Answer

RAG works like:

User Question
→ Retrieve relevant document information
→ Provide retrieved information to LLM
→ Generate grounded answer

This helps the model answer questions based on the contents of the user's document.

---

# 🏗️ Architecture

```text
                ┌───────────────────┐
                │    PDF Upload     │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │   PDF Loader      │
                │   PyPDFLoader     │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │   Text Chunking   │
                │ Recursive Splitter│
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Mistral Embeddings│
                │   mistral-embed   │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │    ChromaDB       │
                │  Vector Database  │
                └─────────┬─────────┘
                          │
                          │
User Question ────────────┤
                          ▼
                ┌───────────────────┐
                │   MMR Retriever   │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Relevant Context  │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │   Mistral LLM     │
                │ mistral-small-2506│
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │    AI Answer      │
                └───────────────────┘
